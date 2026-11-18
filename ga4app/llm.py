"""OpenRouter streaming client for the AI narrator: a thin httpx wrapper, no SDK.

``stream_chat`` posts one chat completion with ``stream: true`` and turns the server-sent events
into ``StreamEvent`` objects. Keep-alive comments (``: OPENROUTER PROCESSING``) are skipped,
``data: [DONE]`` ends the stream, a chunk with a top-level ``error`` object is a mid-stream failure,
and the final usage-only chunk is accounting. 429/502/503 are retried once; any other non-200 status
raises ``OpenRouterError``. Sampling parameters are never sent, because Claude Opus 5.5 rejects them.
``GA4APP_LLM_FAKE=1`` swaps in ``fake_transport()`` so tests and AppTest never touch the network.
The API key is never logged, cached or put in a message.
"""

from __future__ import annotations

import email.utils
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import httpx

from . import config

RETRY_STATUSES = frozenset({429, 502, 503})
MAX_RETRY_WAIT_S = 10.0
DEFAULT_RETRY_WAIT_S = 1.0
SAMPLING_PARAMS = ("temperature", "top_p", "top_k")  # never sent

# Offline canned stream used when GA4APP_LLM_FAKE=1 (AppTest) and in unit tests.
FAKE_MODEL = "ga4app/offline-fake"
FAKE_TEXT_CHUNKS = (
    "**Verdict:** Feature the top hidden gem [G1] and the top winner [W1]. ",
    "Retire [X1] from the homepage - it drives views but almost no purchases.\n\n",
    "- Offline test response: no model was called.\n"
    "- [G1] and [W1] lead their quadrants on adjusted conversion; [X1] has the most forgone purchases.\n\n"
    "Watch-outs: canned text for automated tests only.",
)
FAKE_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}

_STATUS_HINTS = {
    400: "bad model id or request",
    401: "invalid API key",
    402: "out of OpenRouter credits",
    403: "blocked by moderation",
    408: "request timed out",
    429: "rate limited",
    502: "model provider error",
    503: "no provider available",
}
_DETAIL_STATUSES = frozenset({400, 403})  # the server's message names the bad model id / flagged input
_KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_\-]{6,}")

_sleep = time.sleep  # module-level so tests can skip the real Retry-After wait


# --------------------------------------------------------------------------- types


@dataclass
class StreamEvent:
    """One parsed stream item: a text delta, accounting data, an error, or the end of the stream."""

    kind: Literal["text", "meta", "error", "done"]
    text: str = ""
    model: str | None = None
    usage: dict | None = None
    finish_reason: str | None = None
    status: int | None = None
    code: str | None = None


class OpenRouterError(RuntimeError):
    """OpenRouter refused or failed a request. ``status`` is the HTTP status (None for local checks)."""

    def __init__(self, status: int | None, code: str | int | None, message: str) -> None:
        self.status = status
        self.code = None if code is None else str(code)
        self.message = message
        label = f"HTTP {status}" if status else "error"
        super().__init__(f"OpenRouter {label}: {message}")

    @property
    def reason(self) -> str:
        """Short plain-language cause for UI notices."""
        if self.status is None:
            return _clip(self.message, 160)
        hint = _STATUS_HINTS.get(self.status)
        head = f"{hint}, HTTP {self.status}" if hint else f"HTTP {self.status}"
        if hint and self.status not in _DETAIL_STATUSES:
            return head
        return f"{head}: {_clip(self.message, 140)}"


# --------------------------------------------------------------------------- key + request


def resolve_api_key(session_value: str | None = None) -> tuple[str | None, str]:
    """Return ``(key, source)``: the session input, then ``st.secrets``, then the environment.

    ``source`` is ``"session"``, ``"secrets"``, ``"env"`` or ``"none"``.
    """
    key = _clean_key(session_value)
    if key:
        return key, "session"
    key = _clean_key(_secrets_value(config.OPENROUTER_KEY_ENV))
    if key:
        return key, "secrets"
    key = _clean_key(os.environ.get(config.OPENROUTER_KEY_ENV))
    if key:
        return key, "env"
    return None, "none"


def _secrets_value(name: str) -> Any:
    try:
        import streamlit as st

        return st.secrets[name]
    except Exception:  # no secrets.toml, key missing, or streamlit unavailable: skip this source
        return None


def _clean_key(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def fake_mode() -> bool:
    """True when ``GA4APP_LLM_FAKE=1``: use the canned offline transport."""
    return os.environ.get(config.LLM_FAKE_ENV, "").strip() == "1"


def default_fallbacks(model: str) -> tuple[str, ...]:
    """Fallback models for ``model``: ``config.DEFAULT_FALLBACKS`` for paid models, none for free ones."""
    if config.is_free_model(model):
        return ()
    return tuple(m for m in config.DEFAULT_FALLBACKS if m != model)


def build_request(
    messages: Sequence[dict],
    *,
    model: str,
    fallbacks: Iterable[str] = (),
    max_tokens: int = config.LLM_MAX_TOKENS,
    reasoning_effort: str | None = config.LLM_REASONING_EFFORT,
) -> dict:
    """Chat-completions body. ``models`` (fallbacks) is omitted when empty and for free models."""
    model = (model or "").strip()
    if not model:
        raise ValueError("a model id is required")
    body: dict[str, Any] = {"model": model}
    cleaned = (m.strip() for m in fallbacks or () if isinstance(m, str))
    extra = [m for m in dict.fromkeys(cleaned) if m and m != model]
    if extra and not config.is_free_model(model):
        body["models"] = extra
    body["messages"] = [dict(m) for m in messages]
    body["stream"] = True
    body["max_tokens"] = int(max_tokens)  # covers hidden reasoning plus the visible answer
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
    return body


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": config.OPENROUTER_REFERER,
        "X-OpenRouter-Title": _ascii(config.APP_TITLE),
    }


# --------------------------------------------------------------------------- streaming


def stream_chat(
    messages: Sequence[dict],
    *,
    api_key: str,
    model: str,
    fallbacks: Iterable[str] = (),
    max_tokens: int = config.LLM_MAX_TOKENS,
    reasoning_effort: str | None = config.LLM_REASONING_EFFORT,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = config.LLM_TIMEOUT_S,
) -> Iterator[StreamEvent]:
    """Stream one completion as ``StreamEvent``s; closing the generator closes the connection.

    Raises ``OpenRouterError`` for a non-200 status (after one retry on 429/502/503); network
    failures surface as ``httpx.HTTPError``.
    """
    key = (api_key or "").strip()
    if not key:
        raise OpenRouterError(None, "no_key", "no OpenRouter API key")
    body = build_request(
        messages, model=model, fallbacks=fallbacks, max_tokens=max_tokens, reasoning_effort=reasoning_effort
    )
    if transport is None and fake_mode():
        transport = fake_transport()
    timeout = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s))
    with httpx.Client(transport=transport, timeout=timeout) as client:
        for attempt in range(2):
            with client.stream("POST", config.OPENROUTER_URL, json=body, headers=_headers(key)) as resp:
                if 200 <= resp.status_code < 300:
                    yield from _iter_events(resp)
                    return
                resp.read()
                if attempt > 0 or resp.status_code not in RETRY_STATUSES:
                    raise _error_from_response(resp, key)
                delay = _retry_delay(resp.headers.get("retry-after"))
            _sleep(delay)


def _retry_delay(value: str | None) -> float:
    """Seconds to wait from a Retry-After header (seconds or HTTP date), capped at 10 s."""
    if not value:
        return DEFAULT_RETRY_WAIT_S
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return DEFAULT_RETRY_WAIT_S
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if seconds != seconds:  # NaN
        return DEFAULT_RETRY_WAIT_S
    return min(max(seconds, 0.0), MAX_RETRY_WAIT_S)


def _error_from_response(resp: httpx.Response, api_key: str) -> OpenRouterError:
    code: Any = None
    message = ""
    try:
        data = resp.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            message = str(err.get("message") or "")
        elif isinstance(err, str):
            message = err
        elif data.get("message"):
            message = str(data["message"])
    if not message:
        message = resp.reason_phrase or f"HTTP {resp.status_code}"
    return OpenRouterError(resp.status_code, code, _clip(_redact(message, api_key), 300))


def _iter_events(resp: httpx.Response) -> Iterator[StreamEvent]:
    model: str | None = None
    usage: dict | None = None
    finish: str | None = None
    done = False
    for data in _sse_data(resp.iter_lines()):
        if data.strip() == "[DONE]":
            done = True
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("model"), str) and chunk["model"]:
            model = chunk["model"]  # the model that actually served the request (fallbacks)
        chunk_usage = chunk.get("usage") if isinstance(chunk.get("usage"), dict) else None
        if chunk_usage is not None:
            usage = chunk_usage
        choice = _first_choice(chunk)
        reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        if reason:
            finish = reason
        if chunk.get("error") or reason == "error":
            yield _error_event(chunk.get("error"), model, usage)
            return
        text = _delta_text(choice)
        if text:
            yield StreamEvent("text", text=text, model=model, finish_reason=reason)
        else:
            # Role-only first chunk or the trailing usage chunk: accounting, not a second terminal event.
            yield StreamEvent("meta", model=model, usage=chunk_usage, finish_reason=reason)
    if done or finish:
        yield StreamEvent("done", model=model, usage=usage, finish_reason=finish)
    else:
        yield StreamEvent(
            "error", text="the stream ended before the answer finished", model=model, usage=usage,
            code="incomplete_stream",
        )


def _sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the data payload of each server-sent event (multi-line data joined with newlines)."""
    buf: list[str] = []
    for line in lines:
        if not line:
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue  # keep-alive comment
        field, _, value = line.partition(":")
        if field != "data":
            continue
        value = value[1:] if value.startswith(" ") else value
        if value.strip() == "[DONE]":
            if buf:  # a previous event missing its blank-line terminator
                yield "\n".join(buf)
                buf = []
            yield value  # end marker: do not wait for a trailing blank line
            continue
        buf.append(value)
    if buf:
        yield "\n".join(buf)


def _first_choice(chunk: dict) -> dict:
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _delta_text(choice: dict) -> str:
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


def _error_event(err: Any, model: str | None, usage: dict | None) -> StreamEvent:
    code: Any = None
    message: Any = None
    if isinstance(err, dict):
        code, message = err.get("code"), err.get("message")
    elif isinstance(err, str):
        message = err
    status = None
    if isinstance(code, int) and not isinstance(code, bool):
        status = code
    elif isinstance(code, str) and code.isdigit():
        status = int(code)
    return StreamEvent(
        "error",
        text=_clip(_redact(str(message or "the provider reported an error"), None), 300),
        model=model,
        usage=usage,
        finish_reason="error",
        status=status,
        code=None if code is None else str(code),
    )


# --------------------------------------------------------------------------- fake transport


def fake_transport(respond: Callable[[dict], Iterable[str]] | None = None) -> httpx.MockTransport:
    """Canned OpenRouter SSE stream: a comment, the content chunks, usage, [DONE].

    ``respond`` maps the request body (a dict) to the text chunks to stream; the default streams
    ``FAKE_TEXT_CHUNKS`` (3 verdict chunks citing [G1]/[W1]/[X1]). Mirrors OpenRouter by rejecting
    sampling parameters with a 400.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.read() or b"{}")
        except json.JSONDecodeError:
            body = {}
        bad = sorted(set(body) & set(SAMPLING_PARAMS)) if isinstance(body, dict) else []
        if bad:
            error = {"code": 400, "message": f"unsupported parameter(s): {', '.join(bad)}"}
            return httpx.Response(400, json={"error": error})
        chunks = tuple(respond(body if isinstance(body, dict) else {})) if respond is not None else FAKE_TEXT_CHUNKS
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_fake_sse(chunks))

    return httpx.MockTransport(handler)


def _fake_sse(chunks: Sequence[str]) -> bytes:
    def event(delta: dict, finish: str | None, usage: dict | None = None) -> str:
        chunk: dict[str, Any] = {
            "id": "gen-offline-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": FAKE_MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}\n\n"

    last = len(chunks) - 1
    parts = [": OPENROUTER PROCESSING\n\n"]
    for i, text in enumerate(chunks):
        delta = {"role": "assistant", "content": text} if i == 0 else {"content": text}
        parts.append(event(delta, "stop" if i == last else None))
    parts.append(event({"content": ""}, "stop", FAKE_USAGE))
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


# --------------------------------------------------------------------------- helpers


def _redact(text: str, api_key: str | None) -> str:
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return _KEY_PATTERN.sub("[redacted]", text)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _ascii(text: str) -> str:
    """Header-safe text: httpx encodes header values as ASCII."""
    text = text.replace("–", "-").replace("—", "-").replace("·", "-")
    return text.encode("ascii", "ignore").decode("ascii")

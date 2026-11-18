"""OpenRouter client tests: SSE parsing, errors, retries, request body and key resolution (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from ga4app import config, llm

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
KEY = "sk-or-v1-testkey0123456789abcdef"


def _chunk(content: str | None = None, *, model: str = "anthropic/claude-sonnet-5.5", finish=None, usage=None, **extra) -> str:
    delta = {} if content is None else {"content": content}
    chunk = {"id": "gen-1", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


USAGE = {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160, "cost": 0.0012}
GOOD_SSE = (
    ": OPENROUTER PROCESSING\n\n"
    + _chunk("", finish=None).replace('"delta": {"content": ""}', '"delta": {"role": "assistant", "content": ""}')
    + _chunk("Hello ")
    + ": OPENROUTER PROCESSING\n\n"
    + _chunk("[G1] ")
    + _chunk("world.", finish="stop")
    + _chunk("", finish="stop", usage=USAGE)
    + "data: [DONE]\n\n"
)


def _sse_transport(body: str, seen: list | None = None, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=body.encode())

    return httpx.MockTransport(handler)


def _run(transport: httpx.BaseTransport, **kw) -> list[llm.StreamEvent]:
    args = {"api_key": KEY, "model": config.DEFAULT_MODEL, "transport": transport}
    args.update(kw)
    return list(llm.stream_chat(MESSAGES, **args))


@pytest.fixture(autouse=True)
def _no_fake_no_sleep(monkeypatch):
    monkeypatch.delenv(config.LLM_FAKE_ENV, raising=False)
    waits: list[float] = []
    monkeypatch.setattr(llm, "_sleep", waits.append)
    return waits


# --------------------------------------------------------------------------- SSE parsing


def test_sse_parsing_text_model_usage_done():
    events = _run(_sse_transport(GOOD_SSE))
    text = "".join(e.text for e in events if e.kind == "text")
    assert text == "Hello [G1] world."
    assert events[-1].kind == "done"
    assert events[-1].model == "anthropic/claude-sonnet-5.5"  # the fallback that actually served it
    assert events[-1].usage == USAGE
    assert events[-1].finish_reason == "stop"
    assert [e.kind for e in events].count("done") == 1
    assert not any(e.kind == "error" for e in events)
    # The role-only first chunk and the usage-only last chunk are accounting, not text.
    metas = [e for e in events if e.kind == "meta"]
    assert len(metas) == 2 and metas[-1].usage == USAGE


def test_sse_crlf_and_missing_blank_line_before_done():
    body = (_chunk("a") + _chunk("b", finish="stop")).replace("\n\n", "\r\n\r\n")
    body = body + "data: [DONE]"  # no trailing newline at all
    events = _run(_sse_transport(body))
    assert "".join(e.text for e in events if e.kind == "text") == "ab"
    assert events[-1].kind == "done"


def test_mid_stream_error_chunk_gives_error_event():
    err = {"code": "server_error", "message": "Provider disconnected unexpectedly"}
    body = _chunk("partial ") + _chunk("", finish="error", error=err) + _chunk("never")
    events = _run(_sse_transport(body))
    assert [e.kind for e in events] == ["text", "error"]
    assert events[-1].code == "server_error"
    assert "Provider disconnected" in events[-1].text
    assert events[-1].finish_reason == "error"


def test_mid_stream_numeric_error_code_sets_status():
    body = _chunk(None, finish="error", error={"code": 502, "message": "upstream died"})
    events = _run(_sse_transport(body))
    assert events[-1].kind == "error" and events[-1].status == 502


def test_stream_ending_without_done_or_finish_is_an_error():
    events = _run(_sse_transport(_chunk("cut off")))
    assert events[-1].kind == "error" and events[-1].code == "incomplete_stream"


def test_non_json_data_lines_are_skipped():
    body = "data: not json\n\n" + _chunk("ok", finish="stop") + "data: [DONE]\n\n"
    events = _run(_sse_transport(body))
    assert "".join(e.text for e in events if e.kind == "text") == "ok"
    assert events[-1].kind == "done"


# --------------------------------------------------------------------------- HTTP errors + retries


def _status_transport(statuses: list[int], calls: list, *, retry_after: str | None = "0") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        if status == 200:
            return httpx.Response(200, content=(_chunk("ok", finish="stop") + "data: [DONE]\n\n").encode())
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        body = {"error": {"code": status, "message": f"failure {status} for key {KEY}"}}
        return httpx.Response(status, json=body, headers=headers)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("status", [400, 401, 402, 408])
def test_non_retryable_errors_raise_once(status):
    calls: list = []
    with pytest.raises(llm.OpenRouterError) as info:
        _run(_status_transport([status], calls))
    assert info.value.status == status
    assert info.value.code == str(status)
    assert len(calls) == 1
    assert KEY not in str(info.value) and KEY not in info.value.reason  # the echoed key is redacted
    assert f"failure {status}" in info.value.message


@pytest.mark.parametrize("status", [429, 502, 503])
def test_retryable_errors_retry_once_then_raise(status, _no_fake_no_sleep):
    calls: list = []
    with pytest.raises(llm.OpenRouterError) as info:
        _run(_status_transport([status], calls, retry_after="3"))
    assert info.value.status == status
    assert len(calls) == 2
    assert _no_fake_no_sleep == [3.0]


def test_429_then_success(_no_fake_no_sleep):
    calls: list = []
    events = _run(_status_transport([429, 200], calls, retry_after=None))
    assert len(calls) == 2
    assert "".join(e.text for e in events if e.kind == "text") == "ok"
    assert _no_fake_no_sleep == [llm.DEFAULT_RETRY_WAIT_S]


def test_retry_after_is_capped_and_parsed():
    assert llm._retry_delay("30") == llm.MAX_RETRY_WAIT_S
    assert llm._retry_delay("2.5") == 2.5
    assert llm._retry_delay("-4") == 0.0
    assert llm._retry_delay("garbage") == llm.DEFAULT_RETRY_WAIT_S
    assert llm._retry_delay("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # a past date
    assert llm._retry_delay(None) == llm.DEFAULT_RETRY_WAIT_S


def test_error_reason_is_plain_language():
    assert llm.OpenRouterError(402, 402, "Insufficient credits").reason == "out of OpenRouter credits, HTTP 402"
    assert "bad-model" in llm.OpenRouterError(400, 400, "bad-model is not a valid model ID").reason
    err = llm.OpenRouterError(401, None, "No auth credentials found")
    assert err.status == 401 and err.code is None and "invalid API key" in err.reason


def test_error_body_without_json():
    def handler(request):
        return httpx.Response(401, content=b"<html>nope</html>")

    with pytest.raises(llm.OpenRouterError) as info:
        _run(httpx.MockTransport(handler))
    assert info.value.status == 401 and info.value.message


def test_missing_key_raises_without_a_request():
    calls: list = []
    with pytest.raises(llm.OpenRouterError) as info:
        _run(_status_transport([200], calls), api_key="  ")
    assert info.value.status is None and info.value.code == "no_key"
    assert calls == []


def test_network_errors_surface_as_httpx_errors():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(httpx.ConnectError):
        _run(httpx.MockTransport(handler))


def test_closing_the_generator_closes_the_connection():
    class Body(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield (_chunk("one ") + _chunk("two ")).encode()
            yield (_chunk("three", finish="stop") + "data: [DONE]\n\n").encode()

        def close(self):
            Body.closed = True

    def handler(request):
        return httpx.Response(200, stream=Body())

    gen = llm.stream_chat(MESSAGES, api_key=KEY, model=config.DEFAULT_MODEL, transport=httpx.MockTransport(handler))
    first = next(gen)
    assert first.kind == "text" and first.text == "one "
    gen.close()
    assert Body.closed


# --------------------------------------------------------------------------- request body + headers


def test_request_body_and_headers():
    seen: list[httpx.Request] = []
    _run(_sse_transport(GOOD_SSE, seen), fallbacks=config.DEFAULT_FALLBACKS)
    req = seen[0]
    assert req.method == "POST" and str(req.url) == config.OPENROUTER_URL
    body = json.loads(req.content)
    assert body["model"] == config.DEFAULT_MODEL
    assert body["models"] == list(config.DEFAULT_FALLBACKS)
    assert body["stream"] is True
    assert body["max_tokens"] == config.LLM_MAX_TOKENS
    assert body["reasoning"] == {"effort": config.LLM_REASONING_EFFORT, "exclude": True}
    assert body["messages"] == MESSAGES
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in body
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert req.headers["content-type"] == "application/json"
    assert req.headers["http-referer"] == config.OPENROUTER_REFERER
    assert req.headers["x-openrouter-title"] == config.APP_TITLE


def test_free_model_and_empty_fallbacks_send_no_models_key():
    body = llm.build_request(MESSAGES, model="openrouter/free", fallbacks=config.DEFAULT_FALLBACKS)
    assert "models" not in body
    body = llm.build_request(MESSAGES, model="google/gemma-4-31b-it:free", fallbacks=["x/y"])
    assert "models" not in body
    body = llm.build_request(MESSAGES, model=config.DEFAULT_MODEL)
    assert "models" not in body
    assert set(body) == {"model", "messages", "stream", "max_tokens", "reasoning"}


def test_build_request_cleans_fallbacks_and_requires_a_model():
    body = llm.build_request(
        MESSAGES, model="a/b", fallbacks=["a/b", " c/d ", "c/d", ""], max_tokens=123, reasoning_effort=None
    )
    assert body["models"] == ["c/d"]
    assert body["max_tokens"] == 123 and "reasoning" not in body
    with pytest.raises(ValueError):
        llm.build_request(MESSAGES, model="  ")


def test_default_fallbacks():
    assert llm.default_fallbacks("anthropic/claude-opus-5.5") == config.DEFAULT_FALLBACKS
    assert llm.default_fallbacks("anthropic/claude-sonnet-5.5") == ()
    assert llm.default_fallbacks("openrouter/free") == ()


# --------------------------------------------------------------------------- fake transport


def test_fake_transport_stream():
    events = _run(llm.fake_transport())
    text = "".join(e.text for e in events if e.kind == "text")
    for code in ("[G1]", "[W1]", "[X1]"):
        assert code in text
    assert text.startswith("**Verdict:**")
    assert events[-1].kind == "done" and events[-1].model == llm.FAKE_MODEL
    assert events[-1].usage == llm.FAKE_USAGE
    assert sum(e.kind == "text" for e in events) == 3


def test_fake_mode_env_selects_fake_transport(monkeypatch):
    monkeypatch.setenv(config.LLM_FAKE_ENV, "1")
    assert llm.fake_mode()
    events = list(llm.stream_chat(MESSAGES, api_key=KEY, model=config.DEFAULT_MODEL))
    assert events[-1].model == llm.FAKE_MODEL


def test_fake_transport_streams_the_responders_chunks():
    seen = []

    def respond(body: dict) -> list[str]:
        seen.append(body)
        return ['{"notes": ', "[]}"]

    events = _run(llm.fake_transport(respond=respond))
    assert [e.text for e in events if e.kind == "text"] == ['{"notes": ', "[]}"]
    assert events[-1].kind == "done" and events[-1].usage == llm.FAKE_USAGE
    assert seen and seen[0]["model"] and seen[0]["messages"]  # the request body reaches the responder
    default = _run(llm.fake_transport())
    assert tuple(e.text for e in default if e.kind == "text") == llm.FAKE_TEXT_CHUNKS


def test_fake_transport_with_no_chunks_still_finishes():
    events = _run(llm.fake_transport(respond=lambda body: []))
    assert not [e for e in events if e.kind == "text"] and events[-1].kind == "done"


def test_fake_transport_rejects_sampling_params():
    with httpx.Client(transport=llm.fake_transport()) as client:
        resp = client.post(config.OPENROUTER_URL, json={"model": "m", "temperature": 0.2})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- key resolution


class _Secrets:
    def __init__(self, data: dict | None):
        self.data = data

    def __getitem__(self, key):
        if self.data is None:
            raise FileNotFoundError("No secrets found")
        return self.data[key]


def test_resolve_api_key_order(monkeypatch):
    import streamlit

    monkeypatch.setattr(streamlit, "secrets", _Secrets({config.OPENROUTER_KEY_ENV: " sk-or-secret-key "}))
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-env-key")
    assert llm.resolve_api_key(" sk-or-session ") == ("sk-or-session", "session")
    assert llm.resolve_api_key("   ") == ("sk-or-secret-key", "secrets")
    assert llm.resolve_api_key(None) == ("sk-or-secret-key", "secrets")

    monkeypatch.setattr(streamlit, "secrets", _Secrets({}))  # file present, key missing -> KeyError
    assert llm.resolve_api_key() == ("sk-or-env-key", "env")

    monkeypatch.delenv(config.OPENROUTER_KEY_ENV)
    assert llm.resolve_api_key() == (None, "none")


def test_resolve_api_key_tolerates_no_secrets_file(monkeypatch):
    import streamlit

    monkeypatch.setattr(streamlit, "secrets", _Secrets(None))
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-env-key")
    assert llm.resolve_api_key() == ("sk-or-env-key", "env")
    monkeypatch.delenv(config.OPENROUTER_KEY_ENV)
    assert llm.resolve_api_key() == (None, "none")


def test_resolve_api_key_ignores_non_string_secrets(monkeypatch):
    import streamlit

    monkeypatch.setattr(streamlit, "secrets", _Secrets({config.OPENROUTER_KEY_ENV: {"nested": "x"}}))
    monkeypatch.delenv(config.OPENROUTER_KEY_ENV, raising=False)
    assert llm.resolve_api_key() == (None, "none")


def test_resolve_api_key_with_real_streamlit_secrets_object(monkeypatch, tmp_path):
    """The real st.secrets with no secrets file raises; resolve_api_key must skip it."""
    import streamlit
    from streamlit.runtime.secrets import Secrets

    monkeypatch.setattr(streamlit, "secrets", Secrets())
    monkeypatch.chdir(tmp_path)  # no .streamlit/secrets.toml here
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "sk-or-env-key")
    key, source = llm.resolve_api_key()
    assert source in ("env", "secrets")  # "secrets" only if the user's global ~/.streamlit file has one
    assert key

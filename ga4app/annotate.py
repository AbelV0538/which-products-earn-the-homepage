"""AI chart notes: the anchors a note may point at, validation with a number check, and the
rule-based notes.

The split of work is the safety rule. The model (or the rule-based template) chooses **what to
point at and what to say**, from the anchors this module offers: an anchor is an ID (a product
code such as ``G1``) plus the facts behind it, rounded the way ``narrative`` rounds them. The chart
function decides **where the note goes**, because it owns the coordinates. ``validate_notes``
checks **what the note says**: the anchor must be one that was offered, and every number in the
text must be a rounding of a number in that anchor's facts or in the chart's scope.

Anchors depend only on the data and the data controls (sidebar filters, the high-traffic cutoff),
never on display controls (Log/Linear, bubble size, colour), so switching a display control
re-places the same notes. ``ai_notes`` asks the model through ``llm.stream_chat`` (the verdict's
model, fallbacks and key) and falls back to ``rule_based_notes`` on any failure. No Streamlit
here; ``ui.chart_notes_controls`` draws the controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from . import config, llm, narrative
from .types import ChartNote, NotesResult

NOTES_PROMPT_VERSION = "notes-2026-11-17.1"

NOTE_KINDS = ("highlight", "caution")
COUNTING_WORDS = frozenset({1, 2, 3, 4, 5})  # "top 3" needs no source number
SEP = config.NOTE_ANCHOR_SEP  # anchor IDs of bars and cells: "Apparel·cyber_week", "G1·mobile"

CHART_SCATTER = "feature_picks_scatter"
CHART_BARS = "holiday_shift_bars"
CHART_HEATMAP = "device_split_heatmap"

QUESTIONS = {
    CHART_SCATTER: "Which products earn a holiday homepage slot, and which waste one?",
    CHART_BARS: "Which categories convert better or worse in the holiday window than in the base window?",
    CHART_HEATMAP: "Which homepage candidates convert better on mobile or on desktop?",
}

SCATTER_ANCHORS_MAX = 15  # the verdict payload's coded products: 5 per actionable quadrant
SCATTER_FACTS = (
    "name", "category", "views", "purchases", "conversion_pct", "adjusted_conversion_pct", "interval_pct",
    "vs_avg", "confidence", "lift_per_1k", "forgone_purchases", "views_undercount", "device_lean",
)
SCATTER_SCOPE = ("window", "filters", "avg_conversion_pct", "views_threshold")

SYSTEM_PROMPT = """\
You are a merchandising analyst for the Google Merchandise Store, annotating one chart for the \
Head of Merchandising. They must decide which products get the few holiday homepage feature slots \
(Black Friday, Cyber Monday, the December gifting push), ranked by view-to-purchase efficiency, \
not raw traffic.

Task
- The user message is JSON: the chart, the question it answers, the scope (numbers that apply to \
the whole chart) and the anchors: the only bubbles, bars or cells you may point at, each with an \
id and its facts.
- Pick at most 4 anchors that matter most for the homepage-slot decision. Fewer is fine.
- For each, write a note: "anchor" (an id copied exactly from anchors), "kind" ("highlight" for a \
reason to feature, "caution" for a reason to doubt or retire), "label" (5 words or fewer, drawn \
on the chart) and "detail" (one sentence of 25 words or fewer, shown under the chart).

Grounding
- Copy every number exactly as it is written in that anchor's facts or in scope; you may round \
it to fewer decimals. Never compute new numbers: no differences, sums, ratios or percent changes \
that are not already there.
- Use only the JSON. Never invent products, causes, stock levels or margins.
- If any anchor has views_undercount, cannibal_pair or confidence "uncertain", include at least \
one caution about such an anchor.
- Name a product in detail with its code in square brackets, for example "Google Tee [G1]". Put \
nothing else in square brackets.

Definitions
- conversion_pct = purchases per product view, in percent. A product view means the product \
appeared in a product-page view event, so rates are low (under 1% on average).
- adjusted_conversion_pct is the shrunk (empirical-Bayes) rate used for ranking; interval_pct is \
its 80% interval; vs_avg is it divided by scope.avg_conversion_pct.
- quadrant: Winners (high traffic, high conversion: keep), Hidden gems (low traffic, high \
conversion: promote), Wasted exposure (high traffic, low conversion: retire). lift_per_1k = extra \
purchases per 1,000 extra views; forgone_purchases = purchases lost against an average product.
- views_undercount: checkout events reach half its product views, so its views look \
under-counted and its conversion overstated. cannibal_pair: on its spike days a related product \
sold less, so featuring both at once may move sales rather than add them.
- Holiday bars: conversion_pct in the focus window (interval_pct is its 95% interval) against \
base_conversion_pct in the base window; lift_pct is the relative change, significant means \
p_value < scope.significance_level; lift_vs_rest_pct is the change beyond the rest of the store.
- Device cells: x_usual is the product's conversion on that device relative to its usual rate \
across devices (1 = as usual); lean is Mobile-first, Desktop-first, Both or Not enough data.

Reply with JSON only, no prose and no code fences:
{"notes": [{"anchor": "G1", "kind": "highlight", "label": "...", "detail": "..."}]}"""

_CODE = re.compile(r"\b[A-Z]\d{1,3}\b")  # product codes (G1, W12) and uncoded heatmap rows (R3)
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_FENCE = re.compile(r"```[A-Za-z]*")


@dataclass
class AnchorSet:
    """What one chart offers the note writer.

    ``anchors`` are ``{"id": ..., **facts}`` dicts in display order; ``scope`` holds the chart-wide
    numbers (window, filters, the average) that any note may quote; ``empty`` is the caption
    shown when there is nothing to point out. ``keys`` maps an anchor ID to its product key for
    the page (the inspector); it is never sent to the model.
    """

    chart: str
    scope: dict[str, Any]
    anchors: list[dict[str, Any]] = field(default_factory=list)
    empty: str = "Nothing on this chart to point out."
    keys: dict[str, str] = field(default_factory=dict)

    @property
    def question(self) -> str:
        return QUESTIONS.get(self.chart, "")

    @property
    def by_id(self) -> dict[str, dict[str, Any]]:
        return {str(a["id"]): a for a in self.anchors}

    def ids(self) -> list[str]:
        return [str(a["id"]) for a in self.anchors]


# --------------------------------------------------------------------------- anchors: Feature Picks


def scatter_anchors(payload: Mapping[str, Any], products: pd.DataFrame,
                    conflict_keys: Iterable[str] = ()) -> AnchorSet:
    """Anchors for the Feature Picks scatter, reusing the verdict payload (``narrative.build_payload``).

    The Winners, Hidden gems and Wasted exposure entries (up to 15, payload order) become anchors
    with their facts, ``quadrant`` and, for products in a cannibalization conflict pair,
    ``cannibal_pair: true``. Products that are uncoded or not eligible in ``products`` are never
    offered, so a note can only point at a bubble that carries its code.
    """
    eligible = products
    if "eligible" in products.columns:
        eligible = products[products["eligible"].astype("boolean").fillna(False).astype(bool)]
    key_of = {}
    if {"code", "product_key"} <= set(eligible.columns):
        key_of = {narrative._text(c): narrative._text(k) for c, k in zip(eligible["code"], eligible["product_key"])
                  if narrative._text(c)}
    conflicts = {str(k) for k in conflict_keys}
    anchors, keys = [], {}
    for quadrant, section in narrative._SECTIONS:
        for entry in payload.get(section) or []:
            code = str(entry.get("code") or "")
            if code not in key_of:
                continue
            anchor: dict[str, Any] = {"id": code}
            anchor.update({k: entry[k] for k in SCATTER_FACTS if entry.get(k) is not None})
            anchor["quadrant"] = quadrant
            if key_of[code] in conflicts:
                anchor["cannibal_pair"] = True
            anchors.append(anchor)
            keys[code] = key_of[code]
    anchors = anchors[:SCATTER_ANCHORS_MAX]
    scope_in = payload.get("scope") or {}
    scope = {k: scope_in[k] for k in SCATTER_SCOPE if scope_in.get(k) is not None}
    return AnchorSet(CHART_SCATTER, scope, anchors,
                     empty="No product has a code in this scope, so there is nothing to point out.",
                     keys={a["id"]: keys[a["id"]] for a in anchors})


# --------------------------------------------------------------------------- anchors: Holiday Shift


def _flag_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def bar_anchor(group: str, period: str) -> str:
    """The anchor ID of one Holiday Shift bar: ``"Apparel·cyber_week"``."""
    return f"{group}{SEP}{period}"


def holiday_anchors(groups: pd.DataFrame | None, *, filters: str = "") -> AnchorSet:
    """Anchors for the Holiday Shift bars (``metrics.period_comparison`` by category, first window
    = base). One per focus-window bar that was tested: base-window bars and ``low_sample`` bars
    (too few purchases to test) are never offered. ``filters`` is the sidebar scope as text."""
    empty = AnchorSet(CHART_BARS, {}, empty="No significant shift to point out.")
    if not isinstance(groups, pd.DataFrame) or groups.empty or not {"group", "period", "cr"} <= set(groups.columns):
        return empty
    df = groups.reset_index(drop=True)
    base_mask = _flag_mask(df, "is_base") if "is_base" in df.columns else df["period"] == df["period"].iloc[0]
    base = df[base_mask]
    base_cr = dict(zip(base["group"].map(narrative._text), base["cr"]))
    base_label = narrative._text(base["period_label"].iloc[0]) if len(base) and "period_label" in df else ""
    base_key = narrative._text(base["period"].iloc[0]) if len(base) else ""
    focus = df[~base_mask & ~_flag_mask(df, "low_sample")]
    anchors = []
    for r in focus.to_dict("records"):
        group, period = narrative._text(r.get("group")), narrative._text(r.get("period"))
        p_value = narrative._num(r.get("p_value"))
        lift = narrative._num(r.get("lift"))
        if not group or p_value is None or lift is None:
            continue
        anchor: dict[str, Any] = {
            "id": bar_anchor(group, period),
            "category": group,
            "window": narrative._text(r.get("period_label")) or period,
            "conversion_pct": narrative._pct(r.get("cr")),
            "base_conversion_pct": narrative._pct(base_cr.get(group)),
            "interval_pct": [narrative._pct(r.get("ci_low")), narrative._pct(r.get("ci_high"))],
            "lift_pct": round(100.0 * lift, 1),
            "p_value": narrative._round(p_value, 4),
            "significant": bool(p_value < narrative.SHIFT_ALPHA),
        }
        rest, p_rest = narrative._num(r.get("lift_vs_rest")), narrative._num(r.get("p_value_vs_rest"))
        if rest is not None:
            anchor["lift_vs_rest_pct"] = round(100.0 * rest, 1)
        if p_rest is not None:
            anchor["p_value_vs_rest"] = narrative._round(p_rest, 4)
        anchors.append(anchor)
    focus_key = narrative._text(focus["period"].iloc[0]) if len(focus) else ""
    scope: dict[str, Any] = {"base": base_label, "focus": anchors[0]["window"] if anchors else ""}
    for key, name in ((base_key, "base_dates"), (focus_key, "focus_dates")):
        if key in config.PERIODS:
            scope[name] = config.PERIODS[key].short()
    if filters:
        scope["filters"] = filters
    scope["significance_level"] = narrative.SHIFT_ALPHA
    return AnchorSet(CHART_BARS, scope, anchors, empty=empty.empty)


# --------------------------------------------------------------------------- anchors: Device Split


def heatmap_anchors(cells: pd.DataFrame | None, lean: pd.DataFrame | None = None, *,
                    scope: Mapping[str, Any] | None = None) -> AnchorSet:
    """Anchors for the Device Split heatmap, from ``charts.heatmap_cells`` (the numbers the cells
    show; anchor IDs ``G1·mobile``, or ``R3·mobile`` for an uncoded row) and ``metrics.device_lean``.

    Never offered: sparse tablet cells (drawn as "–"), and cells within +-19% of the product's
    usual rate (``abs(log2(rel)) < config.HEATMAP_NOTE_MIN_LOG2``) unless the product has a strong
    lean. At most ``config.HEATMAP_ANCHORS_MAX`` cells: strong-lean products' cells first, then the
    strongest, listed in reading order.
    """
    empty = "No product converts clearly differently by device here, so there is nothing to point out."
    need = {"anchor", "product_key", "device", "row", "col", "rel", "sparse"}
    if not isinstance(cells, pd.DataFrame) or cells.empty or not need <= set(cells.columns):
        return AnchorSet(CHART_HEATMAP, dict(scope or {}), empty=empty)
    lean_of: dict[str, str] = {}
    if isinstance(lean, pd.DataFrame) and {"product_key", "lean"} <= set(lean.columns):
        lean_of = {narrative._text(k): narrative._text(v) for k, v in zip(lean["product_key"], lean["lean"])}
    picked = []
    for r in cells.to_dict("records"):
        rel = narrative._num(r.get("rel"))
        if narrative._flag(r.get("sparse")) or rel is None or rel <= 0:
            continue
        key = narrative._text(r.get("product_key"))
        call = lean_of.get(key, "")
        strong = call in narrative.STRONG_LEANS
        strength = abs(math.log2(rel))
        if strength < config.HEATMAP_NOTE_MIN_LOG2 and not strong:
            continue
        anchor: dict[str, Any] = {
            "id": narrative._text(r.get("anchor")),
            "name": narrative._display_name({"item_name": r.get("item_name"), "product_key": key}),
            "device": narrative._text(r.get("device")),
            "conversion_pct": narrative._pct(r.get("shown", r.get("cr_raw"))),
            "x_usual": round(rel, 2),
            "views": narrative._int(r.get("views")),
            "purchases": narrative._int(r.get("purchases")),
        }
        if call:
            anchor["lean"] = call
        picked.append((not strong, -strength, int(r["row"]), int(r["col"]), anchor, key))
    picked.sort(key=lambda t: t[:4])
    kept = sorted(picked[: config.HEATMAP_ANCHORS_MAX], key=lambda t: (t[2], t[3]))
    return AnchorSet(CHART_HEATMAP, dict(scope or {}), [t[4] for t in kept], empty=empty,
                     keys={t[4]["id"]: t[5] for t in kept})


# --------------------------------------------------------------------------- the model call


def _request_json(anchors: AnchorSet) -> dict[str, Any]:
    return {"chart": anchors.chart, "question": anchors.question, "scope": anchors.scope, "anchors": anchors.anchors}


def build_messages(anchors: AnchorSet) -> list[dict]:
    """System prompt plus the chart, question, scope and anchors as compact JSON."""
    data = json.dumps(_request_json(anchors), ensure_ascii=False, separators=(",", ":"), default=str)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Annotate this chart. JSON:\n{data}"},
    ]


def cache_key(anchors: AnchorSet, model: str) -> str:
    """sha256 of the prompt version, model, chart and canonical JSON of scope + anchors. Display
    controls never enter the anchors, so switching one reuses the cached notes."""
    canonical = json.dumps({"scope": anchors.scope, "anchors": anchors.anchors}, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), default=str)
    text = f"{NOTES_PROMPT_VERSION}\n{model}\n{anchors.chart}\n{canonical}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ai_notes(
    anchors: AnchorSet,
    *,
    api_key: str | None,
    model: str,
    fallbacks: Sequence[str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> NotesResult:
    """Ask the model for notes and validate them. Not streamed: the text events are joined.

    Uses ``llm.stream_chat`` as the verdict does (``fallbacks=None`` means
    ``llm.default_fallbacks(model)``; no ``response_format``, the reply is parsed leniently). An
    ``OpenRouterError``, a network error or an empty reply gives the rule-based notes with
    ``error`` set to the reason. ``GA4APP_LLM_FAKE=1`` answers with ``fake_respond``.
    """
    model = (model or "").strip()
    if transport is None and llm.fake_mode():
        transport = llm.fake_transport(respond=fake_respond)
    served, usage, finish = None, None, None
    parts: list[str] = []
    try:
        if not model:
            raise llm.OpenRouterError(None, "no_model", "no model selected")
        if not anchors.anchors:
            return rule_based_notes(anchors)  # nothing to point at: no call
        chunks = llm.stream_chat(
            build_messages(anchors),
            api_key=api_key or "",
            model=model,
            fallbacks=llm.default_fallbacks(model) if fallbacks is None else tuple(fallbacks),
            transport=transport,
        )
        try:
            for ev in chunks:
                served = ev.model or served
                usage = ev.usage or usage
                finish = ev.finish_reason or finish
                if ev.kind == "text":
                    parts.append(ev.text)
                elif ev.kind == "error":
                    raise llm.OpenRouterError(ev.status, ev.code, ev.text or "the stream failed")
        finally:
            chunks.close()
        text = "".join(parts)
        if not text.strip():
            raise llm.OpenRouterError(None, "empty", f"the model returned no text (finish reason: {finish or 'none'})")
    except (llm.OpenRouterError, httpx.HTTPError) as exc:
        rule = rule_based_notes(anchors)
        return NotesResult(rule.notes, "rule", served or model or None, usage, [], narrative._failure_reason(exc))
    return validate_notes(text, anchors, model=served or model, usage=usage)


def fake_respond(body: Mapping[str, Any]) -> list[str]:
    """Offline answer for ``GA4APP_LLM_FAKE=1``: notes for the first two anchors of the request's
    user message (a highlight and a caution). They quote no numbers, so they pass the checks on
    any data."""
    user = next((m.get("content", "") for m in body.get("messages") or [] if m.get("role") == "user"), "")
    request = parse_json_object(user) or {}
    notes = []
    for i, anchor in enumerate((request.get("anchors") or [])[:2]):
        notes.append({
            "anchor": anchor.get("id"),
            "kind": "highlight" if i == 0 else "caution",
            "label": "Offline note" if i == 0 else "Offline caution",
            "detail": "Offline test note: no model was called.",
        })
    reply = json.dumps({"notes": notes})
    return [reply[: len(reply) // 2], reply[len(reply) // 2 :]]


# --------------------------------------------------------------------------- number check


def _numbers_in(node: Any) -> list[float]:
    """Every number in ``node`` as an absolute value: numeric leaves, and numbers written inside
    strings (window dates, "min 200 views"). Booleans are not numbers."""
    out: list[float] = []
    if isinstance(node, bool) or node is None:
        return out
    if isinstance(node, (int, float)):
        if math.isfinite(float(node)):
            out.append(abs(float(node)))
    elif isinstance(node, str):
        out.extend(float(t.replace(",", "")) for t in _NUMBER.findall(node))
    elif isinstance(node, Mapping):
        for v in node.values():
            out.extend(_numbers_in(v))
    elif isinstance(node, (list, tuple)):
        for v in node:
            out.extend(_numbers_in(v))
    return out


def _is_rounding(token: str, allowed: Sequence[float]) -> bool:
    """True when ``token`` (e.g. "2.4", "1,234") is a rounding of a value in ``allowed``:
    written with d decimals, abs(x - v) <= 0.5 * 10^-d. Integers 1-5 are counting words."""
    text = token.replace(",", "")
    x = float(text)
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    if decimals == 0 and x in COUNTING_WORDS:
        return True
    tol = 0.5 * 10.0**-decimals + 1e-9
    return any(abs(x - v) <= tol for v in allowed)


def unmatched_numbers(text: str, facts: Mapping[str, Any], scope: Mapping[str, Any] | None = None) -> list[str]:
    """Numbers in ``text`` that are not a rounding of any number in ``facts`` or ``scope``.

    Product codes (``[G1]``) and the anchor ID are removed first. Signs are ignored ("drops 23%"
    for a -23.4 lift passes); direction words are not checked.
    """
    anchor_id = str(facts.get("id") or "")
    cleaned = text.replace(anchor_id, " ") if anchor_id else text
    cleaned = _CODE.sub(" ", cleaned)
    allowed = _numbers_in({k: v for k, v in facts.items() if k != "id"}) + _numbers_in(scope or {})
    return [t for t in _NUMBER.findall(cleaned) if not _is_rounding(t, allowed)]


# --------------------------------------------------------------------------- validation


def parse_json_object(text: str) -> dict | None:
    """The first JSON object in ``text`` (after any prose), or None."""
    start = (text or "").find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_reply(text: str) -> list | None:
    """The notes list from a model reply: code fences removed, then the first JSON value that
    starts at a ``{`` (``{"notes": [...]}``) or a ``[`` (a bare list). None if nothing parses."""
    t = _FENCE.sub("", text or "")
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
    except json.JSONDecodeError:
        end = max(t.rfind("}"), t.rfind("]"))
        try:
            obj = json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("notes"), list):
            return obj["notes"]
        if "anchor" in obj:
            return [obj]
    return None


def _clip(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters at a word boundary, ending with "..."."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 3]
    if " " in cut[limit // 2 :]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "..."


def _as_draft(item: Any) -> Any:
    if isinstance(item, ChartNote):
        return {"anchor": item.anchor, "kind": item.kind, "label": item.label, "detail": item.detail}
    return item


def validate_notes(
    reply: str | Sequence[Any],
    anchors: AnchorSet,
    *,
    source: str = "ai",
    model: str | None = None,
    usage: dict | None = None,
) -> NotesResult:
    """Check notes and return what may be drawn. Every note goes through it, rule-based ones too.

    ``reply`` is the model's text, or a list of note dicts / ``ChartNote``. Steps: parse (lenient),
    shape (``anchor``, ``label`` and ``detail`` must be strings; an unknown ``kind`` becomes
    ``highlight``), anchor (offered, not a duplicate, the first ``config.NOTES_MAX``), the number
    check, then labels and details are shortened. Notes are numbered 1.. in order.

    For ``source="ai"``: a reply that cannot be read, or one with no surviving note, gives the
    rule-based notes with ``error`` set and ``dropped`` listing ``(anchor, reason)``.
    """
    if isinstance(reply, str):
        items = parse_reply(reply)
        if items is None:
            return _fallback(anchors, "AI notes could not be read", [], model, usage)
    else:
        items = [_as_draft(x) for x in reply]

    offered = anchors.by_id
    dropped: list[tuple[str, str]] = []
    staged: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping) or not all(
            isinstance(item.get(k), str) and item.get(k).strip() for k in ("anchor", "label", "detail")
        ):
            anchor = item.get("anchor") if isinstance(item, Mapping) else None
            dropped.append((str(anchor or ""), "not a note with an anchor, a label and a detail"))
            continue
        anchor = item["anchor"].strip()
        if anchor not in offered:
            dropped.append((anchor, "unknown anchor"))
            continue
        if anchor in seen:
            dropped.append((anchor, "duplicate anchor"))
            continue
        seen.add(anchor)
        if len(staged) >= config.NOTES_MAX:
            dropped.append((anchor, f"more than {config.NOTES_MAX} notes"))
            continue
        kind = str(item.get("kind") or "").strip().lower()
        staged.append({"anchor": anchor, "kind": kind if kind in NOTE_KINDS else "highlight",
                       "label": " ".join(item["label"].split()), "detail": " ".join(item["detail"].split())})

    kept: list[ChartNote] = []
    for note in staged:
        bad = unmatched_numbers(f"{note['label']} {note['detail']}", offered[note["anchor"]], anchors.scope)
        if bad:
            dropped.append((note["anchor"], f"{bad[0]} is not a number from its data"))
            continue
        kept.append(ChartNote(note["anchor"], note["kind"], _clip(note["label"], config.NOTE_LABEL_MAX),
                              _clip(note["detail"], config.NOTE_DETAIL_MAX), len(kept) + 1))

    if source == "ai" and not kept:
        why = "no AI note passed the checks" if items else "the model returned no notes"
        return _fallback(anchors, why, dropped, model, usage)
    return NotesResult(kept, "ai" if source == "ai" else "rule", model, usage, dropped, None)


def _fallback(anchors: AnchorSet, error: str, dropped: list[tuple[str, str]], model: str | None,
              usage: dict | None) -> NotesResult:
    rule = rule_based_notes(anchors)
    return NotesResult(rule.notes, "rule", model, usage, dropped, error)


# --------------------------------------------------------------------------- rule-based notes


def _fmt(value: Any) -> str:
    """A fact as text that passes the number check: integers with thousands separators, floats
    with their stored decimals ("2.37", "12", "0.123")."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{x:,.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _cite(anchor: Mapping[str, Any]) -> str:
    return f"{anchor.get('name') or anchor['id']} [{anchor['id']}]"


def _rate_phrase(anchor: Mapping[str, Any]) -> str:
    vs = anchor.get("vs_avg")
    if vs is not None:
        return f"{_fmt(vs)}× the average"
    return f"{_fmt(anchor.get('conversion_pct'))}%"


def _featured(anchors: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Winners and Hidden gems in the verdict's Feature order: confidence "above" first, then by
    adjusted conversion (confidence "below" never featured), top ``narrative.FEATURE_COUNT``."""
    pool = [a for a in anchors if a.get("quadrant") in ("Winners", "Hidden gems") and a.get("confidence") != "below"]
    pool.sort(key=lambda a: (a.get("confidence") != "above", -(a.get("adjusted_conversion_pct") or 0.0),
                             -(a.get("views") or 0), str(a["id"])))
    return pool[: narrative.FEATURE_COUNT]


def _scatter_drafts(anchors: AnchorSet) -> list[dict[str, str]]:
    by_id = anchors.by_id
    drafts: list[dict[str, str]] = []
    for code, label in (("G1", "Best hidden gem"), ("W1", "Top winner")):
        a = by_id.get(code)
        if a is not None:
            drafts.append({"anchor": code, "kind": "highlight", "label": label,
                           "detail": f"{_cite(a)} converts at {_rate_phrase(a)} on {_fmt(a.get('views'))} views."})
    x = by_id.get("X1")
    if x is not None and x.get("forgone_purchases") is not None:
        drafts.append({"anchor": "X1", "kind": "caution", "label": "Most purchases forgone",
                       "detail": f"{_cite(x)} drew {_fmt(x.get('views'))} views at {_fmt(x.get('conversion_pct'))}% "
                                 f"conversion: {_fmt(x['forgone_purchases'])} purchases forgone."})
    flagged = next((a for a in _featured(anchors.anchors) if a.get("views_undercount")), None)
    if flagged is not None:
        caution = {"anchor": str(flagged["id"]), "kind": "caution", "label": "Views may be under-counted",
                   "detail": f"{_cite(flagged)} converts at {_rate_phrase(flagged)}, but its checkout events reach half "
                             "its product views, so its views look under-counted and the rate overstated."}
        at = next((i for i, d in enumerate(drafts) if d["anchor"] == caution["anchor"]), None)
        if at is None:
            drafts.append(caution)
        else:
            drafts[at] = caution  # the pick's own note becomes the caution
    return drafts


def _bar_drafts(anchors: AnchorSet) -> list[dict[str, str]]:
    """The largest significant rise (highlight) and the largest significant drop (caution)."""
    sig = [a for a in anchors.anchors if a.get("significant") and a.get("lift_pct") is not None]
    rises = sorted((a for a in sig if a["lift_pct"] > 0), key=lambda a: (-a["lift_pct"], a["id"]))
    drops = sorted((a for a in sig if a["lift_pct"] < 0), key=lambda a: (a["lift_pct"], a["id"]))
    alpha = _fmt(anchors.scope.get("significance_level", narrative.SHIFT_ALPHA))
    drafts = []
    for pool, kind, label, word in ((rises, "highlight", "Biggest rise", "better"), (drops, "caution", "Biggest drop", "worse")):
        if pool:
            a = pool[0]
            drafts.append({
                "anchor": str(a["id"]), "kind": kind, "label": label,
                # The lift rounded like the bar's own label ("+108%"); the check accepts it as a rounding.
                "detail": f"{a['category']} converts {math.floor(abs(a['lift_pct']) + 0.5)}% {word} during {a['window']} "
                          f"({_fmt(a['base_conversion_pct'])}% to {_fmt(a['conversion_pct'])}%), significant at p < {alpha}.",
            })
    return drafts


_FAVOURED = {"Mobile-first": "mobile", "Desktop-first": "desktop"}


def _cell_product(anchor: Mapping[str, Any]) -> str:
    return str(anchor["id"]).split(SEP, 1)[0]


def _cell_detail(a: Mapping[str, Any]) -> str:
    return (f"{a['name']} converts at {_fmt(a['x_usual'])}× its usual rate on {a['device']}: "
            f"{_fmt(a.get('conversion_pct'))}% on {_fmt(a.get('views'))} views.")


def _heatmap_drafts(anchors: AnchorSet) -> list[dict[str, str]]:
    """The favoured-device cell of up to 2 strong-lean products, then the strongest remaining cell
    (of another product when there is one)."""
    def strength(a: Mapping[str, Any]) -> float:
        return abs(math.log2(a["x_usual"])) if (a.get("x_usual") or 0) > 0 else 0.0

    cells = [a for a in anchors.anchors if (a.get("x_usual") or 0) > 0]
    favoured = [a for a in cells if _FAVOURED.get(a.get("lean", "")) == a.get("device")]
    favoured.sort(key=lambda a: (-strength(a), str(a["id"])))
    drafts = [{"anchor": str(a["id"]), "kind": "highlight", "label": f"{a['lean']} hero", "detail": _cell_detail(a)}
              for a in favoured[:2]]
    used_cells = {d["anchor"] for d in drafts}
    used_products = {d["anchor"].split(SEP, 1)[0] for d in drafts}
    rest = sorted((a for a in cells if a["id"] not in used_cells), key=lambda a: (-strength(a), str(a["id"])))
    other = [a for a in rest if _cell_product(a) not in used_products] or rest
    if other and strength(other[0]) > 0:
        a = other[0]
        up = a["x_usual"] > 1
        drafts.append({"anchor": str(a["id"]), "kind": "highlight" if up else "caution",
                       "label": f"{'Strong' if up else 'Weak'} on {a['device']}", "detail": _cell_detail(a)})
    return drafts


_DRAFTS = {
    CHART_SCATTER: _scatter_drafts,
    CHART_BARS: _bar_drafts,
    CHART_HEATMAP: _heatmap_drafts,
}


def rule_based_notes(anchors: AnchorSet) -> NotesResult:
    """Deterministic notes from the same anchors (the default, and the fallback for AI notes).
    They go through ``validate_notes`` like any other notes."""
    build = _DRAFTS.get(anchors.chart)
    drafts = build(anchors) if build is not None else []
    return validate_notes(drafts, anchors, source="rule")

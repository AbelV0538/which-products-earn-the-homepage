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
re-places the same notes. No Streamlit here; ``ui.chart_notes_controls`` draws the controls.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config, narrative
from .types import ChartNote, NotesResult

NOTE_KINDS = ("highlight", "caution")
COUNTING_WORDS = frozenset({1, 2, 3, 4, 5})  # "top 3" needs no source number
SEP = "·"  # anchor ID separator: "Apparel·cyber_week", "G1·mobile"

CHART_SCATTER = "feature_picks_scatter"

QUESTIONS = {
    CHART_SCATTER: "Which products earn a holiday homepage slot, and which waste one?",
}

SCATTER_ANCHORS_MAX = 15  # the verdict payload's coded products: 5 per actionable quadrant
SCATTER_FACTS = (
    "name", "category", "views", "purchases", "conversion_pct", "adjusted_conversion_pct", "interval_pct",
    "vs_avg", "confidence", "lift_per_1k", "forgone_purchases", "views_undercount", "device_lean",
)
SCATTER_SCOPE = ("window", "filters", "avg_conversion_pct", "views_threshold")

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


_DRAFTS = {
    CHART_SCATTER: _scatter_drafts,
}


def rule_based_notes(anchors: AnchorSet) -> NotesResult:
    """Deterministic notes from the same anchors (the default, and the fallback for AI notes).
    They go through ``validate_notes`` like any other notes."""
    build = _DRAFTS.get(anchors.chart)
    drafts = build(anchors) if build is not None else []
    return validate_notes(drafts, anchors, source="rule")

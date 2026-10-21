"""The AI narrator's payload, prompt, rule-based fallback and streaming wrapper.

``build_payload`` turns the scored product table (``types.PRODUCT_COLUMNS``) and the optional
Phase 2 frames into compact, rounded JSON. Only the coded products (W#, G#, X#) are sent, so
the model can cite nothing else. ``verdict_stream`` streams the model's answer through
``llm.stream_chat`` and, on any failure, falls back to ``rule_based_verdict``, a deterministic
template built from the same payload. Every string that reaches ``st.markdown`` goes through
``md_safe`` first, because Streamlit renders ``$...$`` as LaTeX.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
import numpy as np
import pandas as pd

from . import config, llm
from .types import Caveats, Filters, Thresholds

PROMPT_VERSION = "verdict-2026-10-20.1"
DEFAULT_CAVEATS: tuple[str, ...] = tuple(Caveats().items)

FEATURE_COUNT = 3  # products named in the verdict's "Feature" clause
RETIRE_COUNT = 2
MAX_BULLETS = 5
NAME_MAX = 60  # characters of a product name sent to the model
RULE_NAME_MAX = 42  # ... and shown in the rule-based text
SHIFT_ALPHA = 0.05  # holiday-shift lifts reported only below this p-value
MAX_SHIFTS = 6
MAX_ALERTS = 3
MAX_PAIRS = 4
VIEW_DEFINITION = (
    "a product view = the product appeared in a product-page view event; "
    "the GA4 sample lists about 12 products per page view"
)
# Status labels of phase2.cannibalization_tests (spelled out to keep this module light).
CANNIBAL_FLAGS = ("Likely cannibalized", "Possible")
STRONG_LEANS = ("Mobile-first", "Desktop-first")

_SECTIONS = (
    ("Winners", "winners"),
    ("Hidden gems", "hidden_gems"),
    ("Wasted exposure", "wasted_exposure"),
)
_CODE_TOKEN = re.compile(r"\b[A-Z]\d{1,3}\b")
_BRACKETS = re.compile(r"\[([^\[\]\n]{1,60})\]")
_DOLLAR = re.compile(r"(?<!\\)\$")

SYSTEM_PROMPT = """\
You are a merchandising analyst briefing the Head of Merchandising of the Google Merchandise \
Store. They must decide which products get the few holiday homepage feature slots (Black \
Friday, Cyber Monday, the December gifting push), ranked by view-to-purchase efficiency, not \
raw traffic.

Grounding
- Use ONLY the JSON in the user message. Never invent products, numbers, causes, stock levels \
or margins.
- Follow every product name with its code in square brackets, for example "Google Tee [G1]". \
Cite only codes that appear in the JSON; never put any other text in square brackets.

Definitions
- Conversion = purchases / product views. A "product view" means the product appeared in a \
product-page view event: the GA4 sample lists about 12 products per page view, so rates are \
low (under 1% on average). Write percentages with 1-2 decimals.
- adjusted_conversion_pct is the shrunk (empirical-Bayes) rate used for ranking; vs_avg is it \
divided by scope.avg_conversion_pct.

Logic
- Feature = Hidden gems and Winners with the highest adjusted conversion; recommended.feature \
lists them in order. Retire = Wasted exposure with the most forgone purchases; \
recommended.retire lists them. If recommended.retire is empty, say no product clearly wastes \
exposure.
- Mention a device lean only for products whose device_lean is Mobile-first or Desktop-first.
- Mention holiday_shift, outside_signal (Google Trends) and cannibalization only if those \
keys are present.
- If fewer than 3 feature candidates qualify (scope.evidence is "thin"), say the evidence is \
thin and suggest lowering min views.

Format
- First line, in this style: **Verdict:** Feature A [G1], B [W2] and C [G3]. Retire X [X1] \
from the homepage - it drives views but almost no purchases.
- Then 3-5 short "- " bullets with numbers from the JSON explaining why.
- Then one line starting with "Watch-outs:" drawn from caveats.
- At most 170 words. Plain markdown only: no headings, tables, LaTeX or emoji."""

_WATCH_OUTS = (
    "Watch-outs: candidates, not proof that featuring causes sales (no homepage-placement history); "
    "no inventory or margin data, so confirm stock before assigning a slot."
)


# --------------------------------------------------------------------------- small helpers


def _num(value: Any) -> float | None:
    """A finite float, or None for missing / NaN / non-numeric values."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(value: Any) -> int:
    x = _num(value)
    return int(round(x)) if x is not None else 0


def _round(value: Any, digits: int) -> float | None:
    x = _num(value)
    return None if x is None else round(x, digits)


def _pct(rate: Any) -> float | None:
    """A rate as a percentage: 2 decimals, 3 when a non-zero value would round to 0.00."""
    x = _num(rate)
    if x is None:
        return None
    pct = 100.0 * x
    return round(pct, 2) if pct == 0 or abs(pct) >= 0.005 else round(pct, 3)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _display_name(row: Mapping[str, Any], limit: int = NAME_MAX) -> str:
    """Product name without square brackets (they are reserved for codes), clipped."""
    name = _text(row.get("item_name")) or _text(row.get("product_key")) or "Unnamed product"
    name = " ".join(name.replace("[", "(").replace("]", ")").split())
    return name if len(name) <= limit else name[: limit - 3].rstrip() + "..."


def _frame(df: Any, required: Sequence[str]) -> pd.DataFrame | None:
    """``df`` when it is a non-empty frame with the required columns, else None."""
    if not isinstance(df, pd.DataFrame) or df.empty or not set(required) <= set(df.columns):
        return None
    return df


def _code_order(code: str) -> tuple[str, int]:
    digits = code[1:]
    return code[:1], int(digits) if digits.isdigit() else 10**6


def _as_list(value: Any) -> list:
    """A list / tuple / array cell as a list; anything else (NaN, None, a string) as []."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return []


def _bool_mask(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False).astype(bool)


# --------------------------------------------------------------------------- payload


def build_payload(
    products: pd.DataFrame,
    th: Thresholds,
    f: Filters,
    *,
    window_label: str,
    period_shift: pd.DataFrame | None = None,
    device_lean: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
    cannibal: pd.DataFrame | None = None,
    top_n: int = config.TOP_N_CODES,
    device_summary: pd.DataFrame | None = None,
) -> dict:
    """Compact, rounded JSON for the narrator (and the rule-based verdict).

    ``products`` is the scored table (``metrics.compute_products``); only rows with a code are
    sent, ``top_n`` per quadrant. Optional inputs: ``period_shift`` (``metrics.period_comparison``,
    first window = base), ``device_lean`` (``metrics.device_lean``), ``alerts``
    (``phase2.trending_alerts``), ``cannibal`` (``phase2.cannibalization_tests``) and
    ``device_summary`` (``metrics.device_summary``). Absent or empty inputs leave their key out.
    """
    p_bar = _num(th.conversion) or 0.0
    leans = _lean_lookup(device_lean)
    refs = _product_refs(products)

    payload: dict[str, Any] = {"prompt_version": PROMPT_VERSION}
    sections: dict[str, list[dict]] = {}
    feature_pool: list[tuple[int, float, float, str]] = []
    for quadrant, key in _SECTIONS:
        entries = []
        for row in _coded_rows(products, quadrant, top_n):
            entry = _entry(row, quadrant, p_bar, leans)
            entries.append(entry)
            confidence = _text(row.get("confidence")) or "uncertain"
            if quadrant != "Wasted exposure" and confidence != "below":
                tier = 0 if confidence == "above" else 1
                cr_adj = _num(row.get("cr_adj"))
                feature_pool.append(
                    (tier, -(cr_adj if cr_adj is not None else -1.0), -float(_int(row.get("views"))), entry["code"])
                )
        sections[key] = entries
    feature = [code for *_, code in sorted(feature_pool)[:FEATURE_COUNT]]
    retire = [e["code"] for e in sections["wasted_exposure"][:RETIRE_COUNT]]

    views = pd.to_numeric(products["views"], errors="coerce").fillna(0) if "views" in products else pd.Series(dtype=float)
    eligible = int(_bool_mask(products["eligible"]).sum()) if "eligible" in products else int((views > 0).sum())
    payload["scope"] = {
        "window": window_label,
        "start": f.start.isoformat(),
        "end": f.end.isoformat(),
        "filters": f.label(),
        "devices": list(f.devices) or ["all"],
        "categories": list(f.categories) or ["all"],
        "min_views": int(f.min_views),
        "products_in_scope": int((views > 0).sum()),
        "eligible_products": eligible,
        "avg_conversion_pct": _pct(p_bar),
        "views_threshold": _int(th.views),
        "views_rule": str(th.views_rule),
        "view_definition": VIEW_DEFINITION,
        "feature_candidates": len(feature),
        "evidence": "thin" if len(feature) < FEATURE_COUNT else "ok",
    }
    payload["recommended"] = {"feature": feature, "retire": retire}
    payload.update(sections)

    optional = {
        "holiday_shift": _shift_section(period_shift),
        "device": _device_section(device_summary),
        "outside_signal": _alert_section(alerts, refs),
        "cannibalization": _cannibal_section(cannibal, refs),
    }
    payload.update({k: v for k, v in optional.items() if v})
    payload["caveats"] = list(DEFAULT_CAVEATS)
    return payload


def _coded_rows(products: pd.DataFrame, quadrant: str, top_n: int) -> list[dict]:
    if "code" not in products.columns or "quadrant" not in products.columns or top_n <= 0:
        return []
    codes = products["code"].map(_text)
    sel = products[(products["quadrant"].map(_text) == quadrant) & (codes != "")]
    rows = sel.to_dict("records")
    for row in rows:
        row["code"] = _text(row["code"])
    rows.sort(key=lambda r: (_code_order(r["code"]), _text(r.get("product_key"))))
    return rows[:top_n]


def _entry(row: Mapping[str, Any], quadrant: str, p_bar: float, leans: Mapping[str, tuple[str, Any]]) -> dict:
    cr_adj = _num(row.get("cr_adj"))
    entry: dict[str, Any] = {
        "code": row["code"],
        "name": _display_name(row),
        "category": _text(row.get("category_group")) or "Other",
        "views": _int(row.get("views")),
        "purchases": _int(row.get("purchases")),
        "conversion_pct": _pct(row.get("cr_raw")),
        "adjusted_conversion_pct": _pct(cr_adj),
        "interval_pct": [_pct(row.get("ci_low")), _pct(row.get("ci_high"))],
        "vs_avg": round(cr_adj / p_bar, 2) if cr_adj is not None and p_bar > 0 else None,
        "confidence": _text(row.get("confidence")) or "uncertain",
        "revenue_usd": _int(row.get("revenue_usd")),
    }
    if quadrant == "Wasted exposure":
        entry["forgone_purchases"] = _round(row.get("forgone_purchases"), 1)
        entry["views_share_pct"] = _round(100.0 * (_num(row.get("views_share")) or 0.0), 1)
    else:
        lift = _num(row.get("lift_per_1k"))
        entry["lift_per_1k"] = None if lift is None else round(lift, 2 if abs(lift) < 1 else 1)
    lean = leans.get(_text(row.get("product_key")))
    if lean is not None:
        entry["device_lean"] = lean[0]
        rr = _round(lean[1], 2)
        if rr is not None:
            entry["mobile_vs_desktop_x"] = rr
    return entry


def _lean_lookup(device_lean: pd.DataFrame | None) -> dict[str, tuple[str, Any]]:
    """product_key -> (lean, rr) for strong leans only."""
    df = _frame(device_lean, ("product_key", "lean"))
    if df is None:
        return {}
    rr = df["rr"] if "rr" in df.columns else pd.Series([None] * len(df), index=df.index)
    out: dict[str, tuple[str, Any]] = {}
    for key, lean, r in zip(df["product_key"], df["lean"], rr):
        if _text(lean) in STRONG_LEANS:
            out.setdefault(_text(key), (_text(lean), r))
    return out


def _product_refs(products: pd.DataFrame) -> dict[str, dict]:
    """product_key -> {"name", "code"?} for cross-references in the Phase 2 sections."""
    if "product_key" not in products.columns:
        return {}
    cols = [c for c in ("product_key", "item_name", "code") if c in products.columns]
    refs: dict[str, dict] = {}
    for row in products[cols].to_dict("records"):
        key = _text(row.get("product_key"))
        if key and key not in refs:
            ref = {"name": _display_name(row)}
            code = _text(row.get("code"))
            if code:
                ref["code"] = code
            refs[key] = ref
    return refs


def _ref(refs: Mapping[str, dict], key: Any, fallback_name: Any = None) -> dict:
    found = refs.get(_text(key))
    if found is not None:
        return dict(found)
    return {"name": _display_name({"item_name": fallback_name, "product_key": key})}


def _shift_section(period_shift: pd.DataFrame | None) -> list[dict] | None:
    df = _frame(period_shift, ("group", "period_label", "cr", "lift", "p_value"))
    if df is None:
        return None
    if "is_base" in df.columns:
        base_mask = _bool_mask(df["is_base"])
    else:
        base_mask = df["period"] == df["period"].iloc[0] if "period" in df.columns else df["lift"].isna()
    base = df[base_mask]
    base_label = _text(base["period_label"].iloc[0]) if len(base) else ""
    base_cr = dict(zip(base["group"].map(_text), base["cr"]))
    p = pd.to_numeric(df["p_value"], errors="coerce")
    keep = ~base_mask & (p < SHIFT_ALPHA)
    if "low_sample" in df.columns:
        keep &= ~_bool_mask(df["low_sample"])
    rows = []
    for r in df[keep].to_dict("records"):
        lift = _num(r.get("lift"))
        if lift is None:
            continue
        group = _text(r.get("group"))
        rows.append(
            {
                "category": group,
                "base": base_label,
                "focus": _text(r.get("period_label")),
                "base_conversion_pct": _pct(base_cr.get(group)),
                "focus_conversion_pct": _pct(r.get("cr")),
                "lift_pct": round(100.0 * lift, 1),
                "p_value": _round(r.get("p_value"), 4),
            }
        )
    # Site-wide row first, then the biggest moves.
    rows.sort(key=lambda r: (r["category"] != "All products", -abs(r["lift_pct"]), r["category"], r["focus"]))
    return rows[:MAX_SHIFTS] or None


def _device_section(device_summary: pd.DataFrame | None) -> dict | None:
    df = _frame(device_summary, ("device", "cr"))
    if df is None:
        return None
    out = {}
    for r in df.to_dict("records"):
        device = _text(r.get("device"))
        if not device or _num(r.get("views")) == 0:
            continue
        item = {"conversion_pct": _pct(r.get("cr"))}
        if "views_share" in r:
            item["views_share_pct"] = _round(100.0 * (_num(r.get("views_share")) or 0.0), 1)
        out[device] = item
    return out or None


def _alert_section(alerts: pd.DataFrame | None, refs: Mapping[str, dict]) -> list[dict] | None:
    df = _frame(alerts, ("group", "flagged"))
    if df is None:
        return None
    rows = []
    for r in df[_bool_mask(df["flagged"])].to_dict("records"):
        m_ext, m_site = _num(r.get("M_ext")), _num(r.get("M_site"))
        keys, names = _as_list(r.get("promote_ahead")), _as_list(r.get("promote_names"))
        promote = [_ref(refs, k, names[i] if i < len(names) else None) for i, k in enumerate(keys[:3])]
        item: dict[str, Any] = {
            "group": _text(r.get("group")),
            "group_type": _text(r.get("group_type")) or "category",
            "keywords": _text(r.get("keywords")),
            "google_change_pct": None if m_ext is None else round(100.0 * (m_ext - 1.0)),
            "site_views_change_pct": None if m_site is None else round(100.0 * (m_site - 1.0)),
        }
        if promote:
            item["promote_ahead"] = promote
        rows.append(item)
    rows.sort(key=lambda r: (-(r["google_change_pct"] or 0), r["group"]))
    return rows[:MAX_ALERTS] or None


def _cannibal_section(cannibal: pd.DataFrame | None, refs: Mapping[str, dict]) -> list[dict] | None:
    df = _frame(cannibal, ("product_a", "product_b", "status"))
    if df is None:
        return None
    rows = []
    for r in df[df["status"].map(_text).isin(CANNIBAL_FLAGS)].to_dict("records"):
        a = _ref(refs, r.get("product_a"), r.get("item_name_a"))
        b = _ref(refs, r.get("product_b"), r.get("item_name_b"))
        if "code" not in a and "code" not in b:
            continue  # only pairs that touch a pick
        dip = _num(r.get("dip"))
        rows.append(
            {
                "status": _text(r.get("status")),
                "category": _text(r.get("category_group")),
                "spiking_product": a,
                "affected_product": b,
                "dip_pct": None if dip is None else round(100.0 * dip),
                "q_value": _round(r.get("q_value"), 3),
                "spike_days": _int(r.get("n_spike_days")),
            }
        )
    rows.sort(key=lambda r: (CANNIBAL_FLAGS.index(r["status"]), r["q_value"] if r["q_value"] is not None else 1.0))
    return rows[:MAX_PAIRS] or None


def build_messages(payload: Mapping[str, Any]) -> list[dict]:
    """System prompt plus the payload as compact JSON."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Write the homepage verdict for this scope. JSON:\n" + data},
    ]


def cache_key(payload: Mapping[str, Any], model: str) -> str:
    """sha256 of PROMPT_VERSION + model + canonical JSON, so dict ordering never matters."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{canonical}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- citations + escaping


def payload_codes(payload: Any) -> set[str]:
    """Every product code that appears anywhere in the payload."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if k == "code" and isinstance(v, str) and v:
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    walk(payload)
    return found


def unknown_citations(text: str, payload: Mapping[str, Any]) -> list[str]:
    """Bracketed codes (``[X9]``, also ``[G1, W2]``) in ``text`` that the payload does not contain."""
    known = payload_codes(payload)
    out: list[str] = []
    for inner in _BRACKETS.findall(text or ""):
        for code in _CODE_TOKEN.findall(inner):
            if code not in known and code not in out:
                out.append(code)
    return out


def md_safe(text: str) -> str:
    """Escape ``$`` so ``st.markdown`` does not start LaTeX. Idempotent."""
    return _DOLLAR.sub(r"\\$", text or "")


# --------------------------------------------------------------------------- rule-based verdict


def _fmt_pct(value: Any) -> str:
    x = _num(value)
    if x is None:
        return "n/a"
    return f"{x:.2f}%" if x == 0 or abs(x) >= 0.005 else f"{x:.3f}%"


def _fmt_num(value: Any, digits: int = 0) -> str:
    x = _num(value)
    if x is None:
        return "n/a"
    return f"{x:,.{digits}f}"


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _entries_by_code(payload: Mapping[str, Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for _, key in _SECTIONS:
        for e in payload.get(key) or []:
            if isinstance(e, Mapping) and e.get("code"):
                out.setdefault(str(e["code"]), dict(e))
    return out


def _recommended(payload: Mapping[str, Any], entries: Mapping[str, dict]) -> tuple[list[str], list[str]]:
    rec = payload.get("recommended")
    if isinstance(rec, Mapping):
        feature = [c for c in rec.get("feature") or [] if c in entries]
        retire = [c for c in rec.get("retire") or [] if c in entries]
        return feature, retire
    # Hand-built payloads: same rule as build_payload, on the rounded numbers.
    pool = [
        (0 if e.get("confidence") == "above" else 1, -(_num(e.get("adjusted_conversion_pct")) or -1.0),
         -float(e.get("views") or 0), c)
        for c, e in entries.items()
        if c[:1] in ("W", "G") and e.get("confidence") != "below"
    ]
    feature = [c for *_, c in sorted(pool)[:FEATURE_COUNT]]
    retire = sorted((c for c in entries if c[:1] == "X"), key=_code_order)[:RETIRE_COUNT]
    return feature, retire


def _cite(entry: Mapping[str, Any]) -> str:
    name = str(entry.get("name") or "Unnamed product")
    if len(name) > RULE_NAME_MAX:
        name = name[: RULE_NAME_MAX - 3].rstrip() + "..."
    code = entry.get("code")
    return f"{name} [{code}]" if code else name


def _feature_bullet(e: Mapping[str, Any]) -> str:
    vs = _num(e.get("vs_avg"))
    vs_txt = f", {vs:.1f}x the average" if vs is not None else ""
    if str(e.get("code", ""))[:1] == "G":
        lift = _num(e.get("lift_per_1k"))
        text = (
            f"- {_cite(e)} converts at {_fmt_pct(e.get('conversion_pct'))} on only {_fmt_num(e.get('views'))} views "
            f"({_fmt_pct(e.get('adjusted_conversion_pct'))} adjusted{vs_txt}): about "
            f"{_fmt_num(lift, 2 if lift is not None and lift < 1 else 1)} extra purchases per 1,000 extra views "
            "if new viewers convert like current ones."
        )
    else:
        text = (
            f"- {_cite(e)} already earns its traffic: {_fmt_num(e.get('views'))} views at "
            f"{_fmt_pct(e.get('adjusted_conversion_pct'))} adjusted conversion{vs_txt}, "
            f"${_fmt_num(e.get('revenue_usd'))} revenue."
        )
    if e.get("confidence") == "uncertain":
        text += " Its interval still overlaps the average."
    return text


def _device_bullet(picks: Sequence[Mapping[str, Any]]) -> str | None:
    parts = []
    for e in picks:
        lean = e.get("device_lean")
        if lean in STRONG_LEANS:
            word = "mobile" if lean == "Mobile-first" else "desktop"
            verb = "converts best on " if not parts else "on "
            parts.append(f"{_cite(e)} {verb}{word}")
    if not parts:
        return None
    return f"- Device lean: {_join(parts)}; match the slot to that audience."


def _trend_bullet(payload: Mapping[str, Any]) -> str | None:
    alerts = payload.get("outside_signal") or []
    if not alerts:
        return None
    a = alerts[0]
    google = _num(a.get("google_change_pct"))
    site = _num(a.get("site_views_change_pct"))
    text = f"- Outside signal: Google interest in {a.get('group')} is up {_fmt_num(google)}%"
    if site is not None:
        text += f" while on-site views moved {site:+.0f}%"
    promote = [_cite(p) for p in a.get("promote_ahead") or []]
    text += f"; promote {_join(promote)} ahead of demand." if promote else "; demand is building off-site."
    return text


def _cannibal_bullet(payload: Mapping[str, Any]) -> str | None:
    pairs = payload.get("cannibalization") or []
    if not pairs:
        return None
    p = pairs[0]
    a, b = p.get("spiking_product") or {}, p.get("affected_product") or {}
    hedge = "" if p.get("status") == "Likely cannibalized" else " (possible, not yet conclusive)"
    return (
        f"- Cannibalization{hedge}: over the full season, on {_cite(a)}'s spike days, {_cite(b)} sold about "
        f"{_fmt_num(p.get('dip_pct'))}% less than expected; avoid featuring both at once."
    )


def _shift_bullet(payload: Mapping[str, Any]) -> str | None:
    shifts = payload.get("holiday_shift") or []
    if not shifts:
        return None
    s = shifts[0]
    lift = _num(s.get("lift_pct"))
    if lift is None:
        return None
    return (
        f"- Holiday shift: {s.get('category')} conversion moved {lift:+.1f}% in {s.get('focus')} vs "
        f"{s.get('base')} ({_fmt_pct(s.get('base_conversion_pct'))} to {_fmt_pct(s.get('focus_conversion_pct'))})."
    )


def rule_based_verdict(payload: Mapping[str, Any]) -> str:
    """Deterministic verdict in the narrator's format, citing only payload codes. Already ``md_safe``."""
    scope = payload.get("scope") or {}
    entries = _entries_by_code(payload)
    feature, retire = _recommended(payload, entries)
    feats = [entries[c] for c in feature]
    avg = _fmt_pct(scope.get("avg_conversion_pct"))
    min_views = _fmt_num(scope.get("min_views"))

    if len(feats) >= FEATURE_COUNT:
        head = f"**Verdict:** Feature {_join([_cite(e) for e in feats])}."
    elif feats:
        word = "product qualifies" if len(feats) == 1 else "products qualify"
        head = (
            f"**Verdict:** The evidence is thin: only {len(feats)} {word} to feature. Feature "
            f"{_join([_cite(e) for e in feats])} for now, and lower min views (now {min_views}) to widen the pool."
        )
    else:
        head = (
            f"**Verdict:** The evidence is thin: no product qualifies to feature at {min_views}+ views. "
            "Lower min views or widen the date range."
        )
    x = entries[retire[0]] if retire else None
    if x is not None:
        forgone = _num(x.get("forgone_purchases"))
        head += (
            f" Retire {_cite(x)} from the homepage - it drew {_fmt_num(x.get('views'))} views but converted at "
            f"{_fmt_pct(x.get('conversion_pct'))} vs the {avg} average (about "
            f"{_fmt_num(forgone, 1 if forgone is not None and forgone < 10 else 0)} purchases forgone)."
        )
    else:
        head += " No product clearly wastes homepage exposure in this scope."

    feature_bullets = [_feature_bullet(e) for e in feats]
    extras = [
        b
        for b in (_cannibal_bullet(payload), _trend_bullet(payload), _device_bullet(feats), _shift_bullet(payload))
        if b
    ]
    # Keep at least two feature bullets, then fill up to MAX_BULLETS with the signals.
    extras = extras[: MAX_BULLETS - min(len(feature_bullets), 2)]
    bullets = feature_bullets[: MAX_BULLETS - len(extras)] + extras
    if len(bullets) < 3 and x is not None and x.get("views_share_pct") is not None:
        bullets.append(
            f"- {_cite(x)} takes {_fmt_num(x.get('views_share_pct'), 1)}% of in-scope views at "
            f"{_fmt_pct(x.get('adjusted_conversion_pct'))} adjusted conversion."
        )
    if len(bullets) < 3:
        bullets.append(
            f"- Scope: {_fmt_num(scope.get('eligible_products'))} eligible products at {min_views}+ views, "
            f"{avg} average conversion; the high-traffic line is {_fmt_num(scope.get('views_threshold'))} views "
            f"({scope.get('views_rule', 'median')})."
        )
    return md_safe("\n".join([head, "", *bullets, "", _WATCH_OUTS]))


# --------------------------------------------------------------------------- streaming


def _md_escape_inline(text: str) -> str:
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", text)


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, llm.OpenRouterError):
        return exc.reason
    if isinstance(exc, httpx.TimeoutException):
        return "request timed out"
    if isinstance(exc, httpx.HTTPError):
        return f"network error: {type(exc).__name__}"
    return type(exc).__name__


def verdict_stream(
    payload: Mapping[str, Any],
    *,
    api_key: str | None,
    model: str,
    fallbacks: Sequence[str] | None = None,
    sink: dict,
    transport: httpx.BaseTransport | None = None,
) -> Iterator[str]:
    """Stream the AI verdict as ``md_safe`` chunks (for ``st.write_stream``).

    ``sink`` receives ``text`` (raw model text; the rule-based text on failure), ``model`` (the
    model that served it), ``usage``, ``error`` (None on success), ``fallback`` and ``unknown``
    (codes not in the payload). On an ``OpenRouterError`` or httpx error it yields a short notice
    followed by ``rule_based_verdict(payload)``. ``fallbacks=None`` means
    ``llm.default_fallbacks(model)``. ``GA4APP_LLM_FAKE=1`` selects ``llm.fake_transport()``.
    """
    sink.update(text="", model=None, usage=None, error=None, fallback=False, unknown=[], finish_reason=None)
    model = (model or "").strip()
    if transport is None and llm.fake_mode():
        transport = llm.fake_transport()
    parts: list[str] = []
    try:
        if not model:
            raise llm.OpenRouterError(None, "no_model", "no model selected")
        chunks = llm.stream_chat(
            build_messages(payload),
            api_key=api_key or "",
            model=model,
            fallbacks=llm.default_fallbacks(model) if fallbacks is None else tuple(fallbacks),
            transport=transport,
        )
        try:
            for ev in chunks:
                if ev.model:
                    sink["model"] = ev.model
                if ev.usage:
                    sink["usage"] = ev.usage
                if ev.finish_reason:
                    sink["finish_reason"] = ev.finish_reason
                if ev.kind == "text" and ev.text:
                    parts.append(ev.text)
                    yield md_safe(ev.text)
                elif ev.kind == "error":
                    raise llm.OpenRouterError(ev.status, ev.code, ev.text or "the stream failed")
        finally:
            chunks.close()  # closing the connection stops billing on supported providers
        text = "".join(parts)
        if not text.strip():
            raise llm.OpenRouterError(
                None, "empty", f"the model returned no text (finish reason: {sink['finish_reason'] or 'none'})"
            )
    except (llm.OpenRouterError, httpx.HTTPError) as exc:
        reason = _failure_reason(exc)
        fallback = rule_based_verdict(payload)
        sink.update(text=fallback, error=reason, fallback=True, unknown=[])
        lead = "\n\n" if parts else ""
        yield md_safe(f"{lead}_AI narrator unavailable ({_md_escape_inline(reason)}). Showing the rule-based verdict._\n\n")
        yield fallback
        return
    sink["text"] = text
    sink["unknown"] = unknown_citations(text, payload)

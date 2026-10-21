"""Reusable page pieces: headers, KPI tiles, chart + data twins, the verdict card, the inspector."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import streamlit as st

from . import charts, config, data, llm, narrative, state
from .types import Caveats, Filters

PLOTLY_CONFIG = {"displaylogo": False}
QUESTIONS = {
    "Q1": "Which high-traffic products waste homepage exposure?",
    "Q2": "Which low-traffic products convert well enough to promote?",
    "Q3": "How does conversion shift between the baseline and the holiday windows?",
    "Q4": "Which products convert better on mobile or on desktop?",
    "Phase 2a": "Does featuring one product steal sales from its neighbours?",
    "Phase 2b": "Which categories are trending on Google ahead of our own traffic?",
}
COST_NOTE = "About \\$0.02–0.05 per verdict with Claude Opus 5.5; free models are rate-limited."


def esc(text: Any) -> str:
    """Escape ``$`` so st.markdown does not render dynamic text as LaTeX."""
    return narrative.md_safe(str(text))


def pct(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if math.isnan(v) else f"{100 * v:.{digits}f}%"


def header(title: str, subtitle: str, questions: Sequence[str] = ()) -> None:
    """Page title, a plain-language subtitle and the pitch question(s) the page answers."""
    st.title(title)
    st.caption(esc(subtitle))
    if questions:
        lines = [f"**{q}** · {QUESTIONS.get(q, q)}" for q in questions]
        st.markdown("Answers " + " &nbsp; ".join(lines))


def kpi_row(items: Sequence[Mapping[str, Any]]) -> None:
    """One row of st.metric tiles. Each item: label, value, optional delta, delta_color, help."""
    if not items:
        return
    for col, item in zip(st.columns(len(items)), items):
        col.metric(
            item["label"], item["value"], delta=item.get("delta"), delta_color=item.get("delta_color", "normal"),
            help=item.get("help"), border=True,
        )


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def chart_with_table(
    fig, df: pd.DataFrame | None, *, key: str, caption: str | None = None, on_select: Callable | None = None,
    file_name: str | None = None,
):
    """The Plotly chart plus a "Show data" expander with the same rows and a CSV download."""
    kwargs: dict[str, Any] = {}
    if on_select is not None:
        kwargs = {"on_select": on_select, "selection_mode": "points"}
    event = st.plotly_chart(fig, theme=None, width="stretch", key=key, config=PLOTLY_CONFIG, **kwargs)
    if caption:
        st.caption(esc(caption))
    if df is not None:
        with st.expander("Show data"):
            st.dataframe(df, hide_index=True, width="stretch", key=f"{key}_table")
            st.download_button("Download CSV", _csv(df), file_name=file_name or f"{key}.csv", mime="text/csv",
                               on_click="ignore", key=f"{key}_csv")
    return event


def data_table(df: pd.DataFrame, *, key: str, column_config: Mapping[str, Any] | None = None,
               on_select: Callable | None = None, height: int | str = "auto"):
    """A dataframe with column formats; single-row selection when ``on_select`` is given."""
    kwargs: dict[str, Any] = {}
    if on_select is not None:
        kwargs = {"on_select": on_select, "selection_mode": "single-row"}
    return st.dataframe(df, key=key, column_config=dict(column_config or {}), hide_index=True, width="stretch",
                        height=height, **kwargs)


def _selection(widget_key: str) -> Mapping[str, Any]:
    event = st.session_state.get(widget_key)
    try:
        return event["selection"] or {}
    except (KeyError, TypeError):
        return {}


def selected_row_key(table_key: str, frame: pd.DataFrame, column: str = "product_key") -> str | None:
    """The product key of the selected row of a data_table (inside its on_select callback)."""
    rows = list(_selection(table_key).get("rows") or [])
    if not rows or rows[0] >= len(frame):
        return None
    return str(frame.iloc[rows[0]][column])


def selected_point_key(chart_key: str) -> str | None:
    """customdata[0] (the product_key) of the first selected scatter point."""
    for p in _selection(chart_key).get("points") or []:
        cd = p.get("customdata")
        if cd:
            return str(cd[0])
    return None


# --------------------------------------------------------------------------- AI verdict


@st.cache_resource
def _shared_ai_cache() -> dict[str, dict]:
    """AI verdicts shared across sessions (outputs contain no secrets)."""
    return {}


def verdict_block(payload: dict, *, key_prefix: str) -> None:
    """Rule-based verdict at once; "Write AI verdict" streams the model's version in its place."""
    model, api_key, _source = state.ai_settings()
    cache = st.session_state.setdefault(state.AI_CACHE, {})
    ck = narrative.cache_key(payload, model or "-")
    hit = cache.get(ck) or _shared_ai_cache().get(ck)
    rule = narrative.rule_based_verdict(payload)
    box = st.container(border=True)
    body = box.empty()
    if hit:
        body.markdown(narrative.md_safe(hit["text"]))
        box.caption(f"AI verdict · {esc(hit.get('model') or model)} · checked: every product code cites the data.")
    else:
        body.markdown(rule)
        box.caption("Rule-based verdict computed from the numbers on this page.")

    c1, c2 = box.columns([1, 3])
    clicked = c1.button("Write AI verdict", key=f"{key_prefix}ai_write", disabled=not model, type="secondary")
    c2.caption(COST_NOTE + ("" if api_key or llm.fake_mode() else " Add an OpenRouter key under **AI narrator** in the sidebar."))
    last_key = f"{key_prefix}ai_last"
    auto = bool(st.session_state.get(state.AI_AUTO)) and not hit and st.session_state.get(last_key) != ck
    if not (clicked or auto) or not model:
        return
    st.session_state[last_key] = ck
    sink: dict = {}
    with body.container():
        st.write_stream(narrative.verdict_stream(payload, api_key=api_key, model=model, sink=sink))
    unknown = sink.get("unknown") or []
    if sink.get("error"):
        body.markdown(narrative.md_safe(sink.get("text") or rule))
        box.warning(f"AI narrator unavailable ({esc(sink['error'])}). Showing the rule-based verdict.")
    elif unknown:
        body.markdown(rule)
        box.warning(f"The AI cited product codes that are not in the data ({esc(', '.join(unknown))}), so the rule-based verdict is shown instead.")
    else:
        entry = {"text": sink.get("text", ""), "model": sink.get("model") or model, "usage": sink.get("usage")}
        cache[ck] = entry
        _shared_ai_cache()[ck] = entry
        body.markdown(narrative.md_safe(entry["text"]))
        box.caption(f"AI verdict · {esc(entry['model'])}")


# --------------------------------------------------------------------------- product inspector


def _pin(key: str) -> None:
    pinned = list(st.session_state.get(state.PINNED) or [])
    if key not in pinned:
        pinned.append(key)
    st.session_state[state.PINNED] = pinned


def product_inspector(snapshot: data.Snapshot, f: Filters, key: str | None) -> None:
    """Stats, interval, daily timeline and device lean for one product."""
    with st.container(border=True):
        st.markdown("**Product inspector**")
        if not key:
            st.caption("Click a bubble in the chart or a row in a table to inspect a product here.")
            return
        products, _th = data.products(snapshot, f)
        row = products.loc[products["product_key"] == key]
        if row.empty:
            st.info("This product has no activity in the current scope. Widen the filters to inspect it.")
            return
        r = row.iloc[0]
        st.markdown(f"#### {esc(r['item_name'])}")
        quad = r["quadrant"] or "Not ranked (below min views)"
        code = f" · code {r['code']}" if r.get("code") else ""
        st.caption(f"{esc(r['category_group'])} · {esc(r['brand'])} · {esc(quad)}{code}")
        c1, c2 = st.columns(2)
        c1.metric("Product views", f"{int(r['views']):,}")
        c2.metric("Purchases", f"{int(r['purchases']):,}")
        c1.metric("Conversion (raw)", pct(r["cr_raw"]))
        c2.metric("Adjusted conversion", pct(r["cr_adj"]), help="Shrunk toward its category average so small samples don't look extreme.")
        st.caption(
            f"80% likely range: {pct(r['ci_low'])} to {pct(r['ci_high'])}; that is **{esc(r['confidence'])}** the "
            f"category average of {pct(r['prior_mean'])}. Revenue \\${float(r['revenue_usd']):,.0f}."
        )
        if bool(r.get("purchases_exceed_views")):
            st.warning("Data check: this product has more purchases than views (likely a tracking gap).")
        lean = data.device_lean(snapshot, (key,), f)
        if len(lean):
            st.caption(f"Device lean: **{esc(lean.iloc[0]['lean'])}**")
        daily = data.product_daily(snapshot, key, f)
        fig = charts.product_timeline(daily, ["cyber_week", "december"], height=300)
        st.plotly_chart(fig, theme=None, width="stretch", key=f"inspect_timeline_{key}", config=PLOTLY_CONFIG)
        st.caption("Full season, shaded: Cyber Week and the December gifting push.")
        pinned = key in (st.session_state.get(state.PINNED) or [])
        st.button("Added to slot plan shortlist" if pinned else "Add to slot plan shortlist", key="inspect_pin",
                  on_click=_pin, args=(key,), disabled=pinned, width="stretch")


def caveat_note() -> None:
    """The plain-language limits of this analysis."""
    with st.expander("What this analysis can't tell you"):
        st.markdown("\n".join(f"- {esc(c)}" for c in Caveats().items))


def synthetic_banner() -> None:
    st.warning("**SYNTHETIC DATA.** These numbers are generated for testing and do not describe the real store.")

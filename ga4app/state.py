"""Sidebar filters and session state (widget keys are shared by every page).

The filters also live in the URL (``?preset=cyber_week&dev=mobile&cat=Bags&minv=500``, or
``start`` / ``end`` ISO dates for a custom range), so a link reproduces the view: the URL is read
once per session, before the widgets are drawn, and rewritten after every change.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from . import config, llm
from .types import Filters

# Global widget keys (DESIGN section 10).
DATES, PRESET, DEV, CAT, MINV = "dates", "preset", "dev", "cat", "minv"
AI_MODEL, AI_CUSTOM, AI_KEY, AI_AUTO, AI_CACHE = "ai_model", "ai_custom_model", "ai_key", "ai_auto", "ai_cache"
INSPECT = "inspect_key"
PINNED = "sp_pinned"  # products pinned to the slot plan (inspector button or the Slot Plan page)
EXCLUDED = "sp_excluded"  # products kept out of the slot plan (e.g. out of stock)
CUSTOM_MODEL = "Custom…"
DEFAULT_PRESET = "full"
_LAST_DATES = "_dates_valid"  # last complete (start, end) pick
_QP_LOADED = "_query_params_loaded"
QP_LISTS = ("dev", "cat")  # URL keys that repeat (one value per device / category)
# Filters that do not apply on a page are shown disabled.
DATES_OFF = {"holiday_shift", "outside_signal"}
DEVICES_OFF = {"device_split"}


def _preset_dates(key: str) -> tuple[date, date]:
    p = config.PERIODS[key]
    return p.start, p.end


def _preset_label(key: str) -> str:
    return config.PERIODS[key].label


def init_state() -> None:
    """Seed missing keys with their defaults (idempotent)."""
    ss = st.session_state
    ss.setdefault(PRESET, DEFAULT_PRESET)
    ss.setdefault(DATES, _preset_dates(DEFAULT_PRESET))
    ss.setdefault(_LAST_DATES, _preset_dates(DEFAULT_PRESET))
    ss.setdefault(DEV, [])
    ss.setdefault(CAT, [])
    ss.setdefault(MINV, config.DEFAULT_MIN_VIEWS)
    ss.setdefault(AI_MODEL, config.DEFAULT_MODEL)
    ss.setdefault(AI_AUTO, False)
    ss.setdefault(AI_CACHE, {})
    ss.setdefault(INSPECT, None)
    ss.setdefault(PINNED, [])
    ss.setdefault(EXCLUDED, [])


def _iso(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def load_query_params() -> None:
    """Seed the filters from the URL once per session. Unknown or invalid values are ignored."""
    ss = st.session_state
    if ss.get(_QP_LOADED):
        return
    ss[_QP_LOADED] = True
    qp = st.query_params
    preset = qp.get("preset")
    start, end = _iso(qp.get("start")), _iso(qp.get("end"))
    if preset in config.PRESETS:
        ss[PRESET] = preset
        ss[DATES] = ss[_LAST_DATES] = _preset_dates(preset)
    elif start and end:
        start, end = sorted((start, end))
        start, end = max(start, config.DATA_START), min(end, config.ITEM_DATA_END)
        if start <= end:
            ss[PRESET] = None
            ss[DATES] = ss[_LAST_DATES] = (start, end)
    devices = [d for d in config.DEVICES if d in qp.get_all("dev")]
    if devices:
        ss[DEV] = devices
    cats = [c for c in config.CATEGORY_GROUPS if c in qp.get_all("cat")]
    if cats:
        ss[CAT] = cats
    minv = qp.get("minv", "")
    if minv.isdigit() and int(minv) in config.MIN_VIEWS_OPTIONS:
        ss[MINV] = int(minv)


def query_params_for(f: Filters, preset: str | None) -> dict[str, str | list[str]]:
    """The URL parameters that reproduce ``f`` (defaults are left out, so a plain link = defaults)."""
    out: dict[str, str | list[str]] = {}
    if preset:
        if preset != DEFAULT_PRESET:
            out["preset"] = preset
    else:
        out["start"], out["end"] = f.start.isoformat(), f.end.isoformat()
    if f.devices:
        out["dev"] = list(f.devices)
    if f.categories:
        out["cat"] = list(f.categories)
    if f.min_views != config.DEFAULT_MIN_VIEWS:
        out["minv"] = str(f.min_views)
    return out


def sync_query_params() -> None:
    """Rewrite the URL when it no longer matches the sidebar (no rerun is triggered)."""
    want = query_params_for(current_filters(), st.session_state.get(PRESET))
    qp = st.query_params
    have = {k: (qp.get_all(k) if k in QP_LISTS else qp.get(k)) for k in qp.keys()}
    if have != want:
        qp.from_dict(want)


def _on_preset() -> None:
    key = st.session_state.get(PRESET)
    if key:
        st.session_state[DATES] = _preset_dates(key)
        st.session_state[_LAST_DATES] = _preset_dates(key)


def _on_dates() -> None:
    st.session_state[PRESET] = None
    picked = st.session_state.get(DATES) or ()
    if len(picked) == 2:
        st.session_state[_LAST_DATES] = (picked[0], picked[1])


def reset_filters() -> None:
    """Back to the full season, all devices and categories, the default min views."""
    ss = st.session_state
    ss[PRESET] = DEFAULT_PRESET
    ss[DATES] = _preset_dates(DEFAULT_PRESET)
    ss[_LAST_DATES] = _preset_dates(DEFAULT_PRESET)
    ss[DEV] = []
    ss[CAT] = []
    ss[MINV] = config.DEFAULT_MIN_VIEWS
    ss[INSPECT] = None


def current_filters() -> Filters:
    """The sidebar scope from session state (the last complete date range while picking)."""
    ss = st.session_state
    picked = ss.get(DATES) or ()
    start, end = (picked[0], picked[1]) if len(picked) == 2 else ss.get(_LAST_DATES, _preset_dates(DEFAULT_PRESET))
    if start > end:
        start, end = end, start
    devices = tuple(d for d in config.DEVICES if d in (ss.get(DEV) or []))
    cats = tuple(c for c in config.CATEGORY_GROUPS if c in (ss.get(CAT) or []))
    return Filters(start=start, end=end, devices=devices, categories=cats, min_views=int(ss.get(MINV, config.DEFAULT_MIN_VIEWS)))


def ai_settings() -> tuple[str, str | None, str]:
    """(model id, API key or None, key source). The key is never displayed."""
    ss = st.session_state
    model = ss.get(AI_MODEL) or config.DEFAULT_MODEL
    if model == CUSTOM_MODEL:
        model = (ss.get(AI_CUSTOM) or "").strip()
    key, source = llm.resolve_api_key(ss.get(AI_KEY) or None)
    return model, key, source


def render_sidebar(snapshot, page_id: str) -> Filters:
    """Draw the global filters and the AI narrator settings; return the current scope."""
    load_query_params()
    init_state()
    item_dates = snapshot.item_daily["event_date"]
    lo = min(config.DATA_START, pd.Timestamp(item_dates.min()).date())
    hi = config.ITEM_DATA_END  # item-level purchases stop after this day
    dates_off, dev_off = page_id in DATES_OFF, page_id in DEVICES_OFF
    groups = [g for g in config.CATEGORY_GROUPS if g in set(snapshot.product_dim["category_group"].astype(str))]

    with st.sidebar:
        st.subheader("Scope")
        st.segmented_control(
            "Season window", options=list(config.PRESETS), format_func=_preset_label, key=PRESET,
            on_change=_on_preset, disabled=dates_off, help="Pick a window; editing the dates below switches to a custom range.",
        )
        st.date_input("Dates", min_value=lo, max_value=hi, key=DATES, on_change=_on_dates, disabled=dates_off, format="MM/DD/YYYY")
        if dates_off:
            st.caption("This page compares fixed season windows, so the date filter is off here.")
        elif len(st.session_state.get(DATES) or ()) < 2:
            st.caption("Pick an end date; the last complete range is used meanwhile.")
        st.pills("Devices", options=list(config.DEVICES), selection_mode="multi", key=DEV, disabled=dev_off,
                 format_func=str.capitalize)
        if dev_off:
            st.caption("This page compares devices side by side, so the device filter is off here.")
        elif not st.session_state.get(DEV):
            st.caption("None selected = all devices.")
        st.multiselect("Category groups", options=groups, key=CAT, placeholder="All categories")
        st.select_slider("Minimum product views", options=list(config.MIN_VIEWS_OPTIONS), key=MINV,
                         help="Products with fewer views in scope are shown greyed out and are not ranked.")
        st.button("Reset filters", on_click=reset_filters, key="reset_filters", width="stretch")
        dates = snapshot.manifest.get("generated_at") or ""
        st.caption(
            f"Data: {snapshot.manifest.get('source', 'unknown')} snapshot, built {str(dates)[:10]}. "
            f"Dates stop at {config.ITEM_DATA_END:%b %d, %Y}: after that the sample's purchases carry no "
            "products (see Data & Method). The page URL keeps these filters, so it can be shared."
        )

        with st.expander("AI narrator"):
            options = [*config.LLM_MODELS, CUSTOM_MODEL]
            if st.session_state.get(AI_MODEL) not in options:
                st.session_state[AI_MODEL] = config.DEFAULT_MODEL
            st.selectbox("Model", options=options, key=AI_MODEL)
            if st.session_state.get(AI_MODEL) == CUSTOM_MODEL:
                st.text_input("Custom OpenRouter model ID", key=AI_CUSTOM, placeholder="vendor/model-name")
            st.text_input("OpenRouter API key (this session only)", type="password", key=AI_KEY)
            _, key, source = ai_settings()
            labels = {"session": "key entered for this session", "secrets": "key from .streamlit/secrets.toml",
                      "env": "key from the OPENROUTER_API_KEY environment variable", "none": "no key: the rule-based verdict is shown"}
            st.caption(f"Key status: {labels.get(source, source)}.")
            st.toggle("Auto-write when filters change", key=AI_AUTO)
    sync_query_params()
    return current_filters()

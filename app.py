"""Streamlit entry point: snapshot guard, top navigation, sidebar filters."""

from __future__ import annotations

import streamlit as st

from ga4app import config, data, state, ui

st.set_page_config(page_title=config.APP_TITLE, layout="wide")

PAGES = (
    ("views/feature_picks.py", "Feature Picks", "feature_picks"),
    ("views/holiday_shift.py", "Holiday Shift", "holiday_shift"),
    ("views/device_split.py", "Device Split", "device_split"),
    ("views/outside_signal.py", "Outside Signal", "outside_signal"),
    ("views/cannibalization.py", "Cannibalization Check", "cannibalization"),
    ("views/slot_plan.py", "Slot Plan", "slot_plan"),
    ("views/data_method.py", "Data & Method", "data_method"),
)

data_dir = data.resolve_data_dir()
missing = data.missing_required(data_dir)
if missing:
    st.error(
        f"Snapshot files are missing from `{data_dir}`: {', '.join(missing)}.\n\n"
        "Build them with the pipeline (run from the project root):\n\n"
        "```\npython scripts/bq_auth.py\npython scripts/export_bigquery.py\npython scripts/build_site_aggregates.py\n"
        "python scripts/build_snapshots.py\npython scripts/fetch_trends.py   # optional: Google Trends\n"
        "python scripts/verify_snapshots.py\n```\n\n"
        "For a quick test run instead: `python scripts/make_synthetic_data.py --out data_synthetic` and set "
        f"`{config.DATA_DIR_ENV}=data_synthetic`."
    )
    st.stop()

snapshot = data.get_snapshot()
if snapshot.is_synthetic:
    ui.synthetic_banner()

pages = [st.Page(path, title=title, default=(i == 0)) for i, (path, title, _) in enumerate(PAGES)]
pg = st.navigation(pages, position="top")
page_id = next((pid for _, title, pid in PAGES if title == pg.title), "feature_picks")
state.render_sidebar(snapshot, page_id)
pg.run()

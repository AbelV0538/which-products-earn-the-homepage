"""Streamlit entry point: snapshot guard, top navigation, sidebar filters."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ga4app import config, data, state, ui

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title=config.APP_TITLE, page_icon=":material/storefront:", layout="wide")

PAGES = (  # (script, nav title, page id, nav icon)
    ("views/feature_picks.py", "Feature Picks", "feature_picks", ":material/star:"),
    ("views/holiday_shift.py", "Holiday Shift", "holiday_shift", ":material/celebration:"),
    ("views/device_split.py", "Device Split", "device_split", ":material/devices:"),
    ("views/outside_signal.py", "Outside Signal", "outside_signal", ":material/trending_up:"),
    ("views/cannibalization.py", "Cannibalization Check", "cannibalization", ":material/compare_arrows:"),
    ("views/slot_plan.py", "Slot Plan", "slot_plan", ":material/view_module:"),
    ("views/data_method.py", "Data & Method", "data_method", ":material/database:"),
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

# After the guard: the wordmark's light text is drawn for the navy sidebar, which the error page above lacks.
# With the sidebar collapsed, the header shows the square icon instead.
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "logo_icon.svg"))

snapshot = data.get_snapshot()
if snapshot.is_synthetic:
    ui.synthetic_banner()

pages = [st.Page(path, title=title, icon=icon, default=(i == 0)) for i, (path, title, _, icon) in enumerate(PAGES)]
pg = st.navigation(pages, position="top")
page_id = next((pid for _, title, pid, _ in PAGES if title == pg.title), "feature_picks")
state.render_sidebar(snapshot, page_id)
pg.run()

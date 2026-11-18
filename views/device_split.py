"""Device Split (Q4): do the best homepage candidates convert differently on mobile vs desktop?"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from ga4app import annotate, charts, config, data, metrics, state, ui
from ga4app.types import Filters

P = "ds_"
MODES = ("Winners + Hidden gems", "Top adjusted rate", "Pick products")
DEVICE_LABELS = {"desktop": "Desktop", "mobile": "Mobile", "tablet": "Tablet"}


@st.cache_data(show_spinner="Splitting by device…", max_entries=32)
def _matrix(version: str, keys: tuple[str, ...], f: Filters, _snap: data.Snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = metrics.device_matrix(_snap.item_daily, _snap.product_dim, list(keys), f, data.shrinkage_m(_snap))
    return matrix, metrics.device_lean(matrix)


@st.cache_data(show_spinner=False, max_entries=16)
def _funnel(version: str, f: Filters, _snap: data.Snapshot) -> pd.DataFrame | None:
    sf = _snap.site_funnel
    if sf is None or sf.empty:
        return None
    d = pd.to_datetime(sf["event_date"]).dt.date
    sub = sf[(d >= f.start) & (d <= f.end)]
    cols = ["sessions", "view_sessions", "cart_sessions", "checkout_sessions", "purchase_sessions"]
    out = sub.groupby("device_category")[cols].sum().reindex(list(config.DEVICES)).dropna(how="all").fillna(0)
    return out.astype("int64").rename_axis("device").reset_index()


snapshot = data.get_snapshot()
f = replace(state.current_filters(), devices=())

ui.header(
    "Device Split",
    "Do the strongest homepage candidates sell better on phones or on computers? "
    f"The device filter does not apply here: every device is shown side by side. · {f.label()}",
    ("Q4",),
)

summary = data.device_summary(snapshot, f)
if summary.empty or int(summary["views"].sum()) == 0:
    st.info("No product views in this scope. Widen the dates or clear the category filter.")
    st.stop()

tiles = []
for r in summary.itertuples():
    tiles.append({
        "label": DEVICE_LABELS.get(str(r.device), str(r.device).title()), "value": f"{ui.pct(r.cr)} conversion",
        "delta": f"{100 * r.views_share:.0f}% of views · {100 * r.purchases_share:.0f}% of purchases",
        "delta_color": "off", "delta_arrow": "off",
        "help": f"{int(r.views):,} product views, {int(r.purchases):,} purchases.",
    })
ui.kpi_row(tiles)

products, th = data.products(snapshot, f)
eligible = products[products["eligible"].astype(bool)].sort_values(["cr_adj", "views"], ascending=False, kind="mergesort")
if eligible.empty:
    st.info(f"No product has at least {f.min_views:,} views in this scope. Lower the minimum views in the sidebar.")
    st.stop()

c1, c2 = st.columns([1.4, 1])
mode = c1.segmented_control("Candidates", list(MODES), default=MODES[0], key=f"{P}mode") or MODES[0]
n = c2.slider("How many", min_value=10, max_value=40, value=20, step=5, key=f"{P}n", disabled=mode == MODES[2])

if mode == MODES[0]:
    keys = eligible[eligible["quadrant"].isin(["Winners", "Hidden gems"])]["product_key"].astype(str).head(n).tolist()
elif mode == MODES[1]:
    keys = eligible["product_key"].astype(str).head(n).tolist()
else:
    names = dict(zip(eligible["product_key"].astype(str), eligible["item_name"].astype(str)))
    keys = st.multiselect("Products", list(names), default=list(names)[:10], format_func=names.get,
                          key=f"{P}pick", max_selections=40)

if not keys:
    st.info("No candidates to compare. Pick products, or switch to another candidate list.")
    st.stop()

matrix, lean = _matrix(snapshot.version, tuple(keys), f, snapshot)
st.subheader("Candidate conversion by device")
heat_table = matrix[["product_key", "item_name", "device", "views", "purchases", "cr_raw", "cr_adj_d", "index"]].copy()
for c in ("cr_raw", "cr_adj_d"):
    heat_table[c] = 100 * heat_table[c].astype(float)
# Notes name a product by its Feature Picks code (median cutoff), or by its row when it has none.
codes = {k: c for k, c in zip(products["product_key"].astype(str), products["code"].astype(str)) if c}
preset = st.session_state.get(state.PRESET)
note_anchors = annotate.heatmap_anchors(
    charts.heatmap_cells(matrix, codes=codes), lean,
    scope={"window": config.PERIODS[preset].label if preset else "Custom range", "filters": f.label(), "candidates": mode},
)
notes = ui.chart_notes_controls(note_anchors, key_prefix=P)
ui.chart_with_table(
    charts.device_heatmap(matrix, lean, notes=notes, codes=codes), heat_table.round(3), key=f"{P}heat",
    file_name="device_split_heatmap.csv",
    caption="Cell text is the product's conversion on that device; colour compares it with the product's own usual "
            "rate across devices: navy = converts better on this device, coral = worse, gray = as usual. "
            f"Tablet is faded (low volume) and shows \"–\" under {charts.LOW_VOLUME_MIN_VIEWS} views. A lean column "
            "appears on the right only when some product converts clearly better on one device.",
    alt_extra=ui.notes_alt(notes),
)
ui.chart_notes_list(notes, note_anchors, key_prefix=P)

names = matrix.drop_duplicates("product_key").set_index("product_key")["item_name"].astype(str)
mob = [names.get(k, k) for k in lean.loc[lean["lean"] == "Mobile-first", "product_key"]]
desk = [names.get(k, k) for k in lean.loc[lean["lean"] == "Desktop-first", "product_key"]]
st.markdown(ui.esc(
    f"**Mobile-first heroes:** {', '.join(mob) if mob else 'none'}  \n"
    f"**Desktop-first heroes:** {', '.join(desk) if desk else 'none'}"
))
st.caption("Mobile-first = converts at least 1.25x better on mobile than desktop (relative to each device's average) "
           "with p < 0.10; Desktop-first is the reverse. Put mobile-first heroes in the mobile hero slot.")

lean_table = lean.assign(item_name=lean["product_key"].map(names))[
    ["product_key", "item_name", "views_mobile", "purchases_mobile", "views_desktop", "purchases_desktop", "rr", "p_value", "lean"]
]
ui.data_table(lean_table, key=f"{P}lean", column_config={
    "product_key": None,
    "item_name": st.column_config.TextColumn("Product", width="large"),
    "views_mobile": st.column_config.NumberColumn("Mobile views", format="%d"),
    "purchases_mobile": st.column_config.NumberColumn("Mobile purchases", format="%d"),
    "views_desktop": st.column_config.NumberColumn("Desktop views", format="%d"),
    "purchases_desktop": st.column_config.NumberColumn("Desktop purchases", format="%d"),
    "rr": st.column_config.NumberColumn("Mobile vs desktop (x)", format="%.2f",
                                        help="Above 1 = relatively better on mobile; below 1 = better on desktop."),
    "p_value": st.column_config.NumberColumn("p-value", format="%.3f"),
    "lean": "Lean",
})

st.subheader("Site session funnel by device")
funnel = _funnel(snapshot.version, f, snapshot)
if funnel is None or funnel.empty:
    st.info("The site session funnel (site_funnel) is not in this snapshot, so the device funnel can't be shown.")
else:
    s = funnel["sessions"].replace(0, pd.NA).astype("Float64")
    funnel["view_rate"] = 100 * funnel["view_sessions"] / s
    funnel["purchase_rate"] = 100 * funnel["purchase_sessions"] / s
    funnel["view_to_purchase"] = 100 * funnel["purchase_sessions"] / funnel["view_sessions"].replace(0, pd.NA).astype("Float64")
    funnel["device"] = funnel["device"].map(DEVICE_LABELS).fillna(funnel["device"])
    ui.data_table(funnel[["device", "sessions", "view_sessions", "purchase_sessions", "view_rate", "purchase_rate",
                          "view_to_purchase"]], key=f"{P}funnel", column_config={
        "device": "Device",
        "sessions": st.column_config.NumberColumn("Sessions", format="%d"),
        "view_sessions": st.column_config.NumberColumn("Viewed a product", format="%d"),
        "purchase_sessions": st.column_config.NumberColumn("Purchased", format="%d"),
        "view_rate": st.column_config.NumberColumn("% sessions viewing", format="%.1f%%"),
        "purchase_rate": st.column_config.NumberColumn("% sessions buying", format="%.2f%%"),
        "view_to_purchase": st.column_config.NumberColumn("Viewers who buy", format="%.2f%%"),
    })
    st.caption("Context for the whole site in the selected dates: how many sessions on each device get to a product "
               "view and then a purchase.")

_tablet = summary.loc[summary["device"] == "tablet", "views_share"]
_tablet_share = float(_tablet.iloc[0]) if len(_tablet) else 0.0
with st.expander("Caveats for this page"):
    st.markdown(ui.esc("\n".join([
        f"- Tablet has very little traffic ({_tablet_share:.0%} of product views here), so tablet cells are faded and left out of the lean test.",
        "- A product's lean compares it with each device's own average, so a store-wide desktop advantage does not "
        "make every product look desktop-first.",
        "- Products with fewer than 10 mobile + desktop purchases get 'Not enough data'.",
    ])))
ui.caveat_note()

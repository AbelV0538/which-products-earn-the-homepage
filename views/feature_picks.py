"""Feature Picks (Q1 + Q2): which products earn a homepage slot, and which waste one."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ga4app import charts, config, data, narrative, state, ui

P = "fp_"
SCATTER_KEY = f"{P}scatter"
Y_OPTIONS = {"Raw": "cr_raw", "Adjusted": "cr_adj"}
SIZE_OPTIONS = {"Revenue": "revenue_usd", "Orders": "purchases"}
SCALE_OPTIONS = ("Log", "Linear")
RULE_LABELS = {"median": "Median", "p75": "Top 25%", "p90": "Top 10%"}  # by views: the 75th / 90th percentile

snapshot = data.get_snapshot()
f = state.current_filters()
preset = st.session_state.get(state.PRESET)
window_label = config.PERIODS[preset].label if preset else "Custom range"

ui.header(
    "Feature Picks",
    f"Products ranked by how often a view turns into a purchase, not by traffic · {window_label} · {f.label()}",
    ("Q1", "Q2"),
)

# One row of page controls.
c1, c6, c2, c3, c5 = st.columns([1.2, 1.0, 1.9, 1.7, 1.3])
y_label = c1.segmented_control("Conversion axis", list(Y_OPTIONS), default="Raw", key=f"{P}y",
                               help="Adjusted = shrunk toward the category average so low-traffic products don't look extreme.")
y_scale = c6.segmented_control("Scale", list(SCALE_OPTIONS), default="Log", key=f"{P}scale",
                               help="Log shows every product, including the top converters a linear axis has to clip.")
rule = c2.segmented_control("High-traffic cutoff", list(config.VIEWS_RULES), default="median", key=f"{P}rule",
                            format_func=RULE_LABELS.get)
color_mode = c3.segmented_control("Colour by", ["Category", "Highlight one"], default="Category", key=f"{P}color")
size_label = c5.segmented_control("Bubble size", list(SIZE_OPTIONS), default="Revenue", key=f"{P}size",
                                  help="Orders = purchases.")
y_col = Y_OPTIONS.get(y_label or "Raw", "cr_raw")
y_scale = (y_scale or "Log").lower()
rule = rule or "median"

products, th = data.products(snapshot, f, views_rule=rule, y=y_col)
highlight = None
if color_mode == "Highlight one":
    present = [g for g in config.CATEGORY_GROUPS if g in set(products["category_group"].astype(str))]
    c4, _ = st.columns([1.4, 4.5])  # second row, only when highlighting
    highlight = c4.selectbox("Category to highlight", present, key=f"{P}highlight") if present else None

eligible = products[products["eligible"].astype(bool)]
codes = tuple(products.loc[products["code"].astype(str) != "", "product_key"].astype(str))
lean = data.device_lean(snapshot, codes, f)
payload = narrative.build_payload(
    products, th, f, window_label=window_label, device_lean=lean,
    device_summary=data.device_summary(snapshot, f), cannibal=data.season_cannibal(snapshot),
    **data.verdict_context(snapshot, f),
)
ui.verdict_block(payload, key_prefix=P)

# KPI tiles.
season_products, season_th = data.products(snapshot, data.season_filters(f), views_rule=rule, y=y_col)
delta = th.conversion - season_th.conversion
wasted = products[products["quadrant"] == "Wasted exposure"]
ui.kpi_row([
    {"label": "Eligible products", "value": f"{len(eligible):,}",
     "help": f"Products with at least {f.min_views:,} views in scope; only these are ranked."},
    {"label": "Average conversion", "value": ui.pct(th.conversion),
     "delta": None if (f.start, f.end) == (data.FULL_PERIOD.start, data.FULL_PERIOD.end) else f"{100 * delta:+.2f} pts vs full season",
     "help": "Purchases ÷ product views over every product in scope."},
    {"label": "Product views", "value": f"{int(products['views'].sum()):,}"},
    {"label": "Purchases", "value": f"{int(products['purchases'].sum()):,}"},
    {"label": "Forgone purchases", "value": f"{wasted['forgone_purchases'].sum():,.0f}",
     "help": "Extra purchases the Wasted-exposure products would have made converting at the average rate."},
])


def _inspect_point() -> None:
    key = ui.selected_point_key(SCATTER_KEY)
    if key:
        st.session_state[state.INSPECT] = key


left, right = st.columns([2.3, 1])
with left:
    fig = charts.quadrant_scatter(products, th, y=y_col, size=SIZE_OPTIONS.get(size_label or "Revenue", "revenue_usd"),
                                  color_map=snapshot.category_colors, highlight=highlight, y_scale=y_scale)
    cols = ["code", "product_key", "item_name", "category_group", "views", "purchases", "cr_raw", "cr_adj", "revenue_usd", "quadrant"]
    ui.chart_with_table(
        fig, eligible[cols], key=SCATTER_KEY, on_select=_inspect_point, file_name="feature_picks_quadrants.csv",
        caption=(f"Each bubble is a product. Right of the dashed line = high traffic ({RULE_LABELS[rule].lower()}, "
                 f"{th.views:,.0f} views); above the other dashed line = converts better than the {ui.pct(th.conversion)} average. "
                 "Faded bubbles have too few views to be sure. "
                 + ("Both axes are log scale, so each gridline step is a multiple, not a fixed amount. " if y_scale == "log" else "")
                 + "Click a bubble to inspect it."),
    )
with right:
    ui.product_inspector(snapshot, f, st.session_state.get(state.INSPECT))

# Ranked tables.
conflict_keys = {k for pair in data.conflicts(snapshot) for k in pair}
all_lean = data.device_lean(snapshot, tuple(eligible["product_key"].astype(str)), f)
lean_map = dict(zip(all_lean["product_key"].astype(str), all_lean["lean"].astype(str)))


def _table(frame: pd.DataFrame, extra: str | None) -> pd.DataFrame:
    flags = []
    for r in frame.itertuples():
        fl = []
        if r.confidence == "uncertain":
            fl.append("uncertain")
        if bool(r.purchases_exceed_views):
            fl.append("purchases > views")
        if bool(r.views_undercount):
            fl.append("views under-counted")
        if r.product_key in conflict_keys:
            fl.append("cannibal pair")
        flags.append(", ".join(fl))
    out = pd.DataFrame({
        "product_key": frame["product_key"].astype(str),
        "code": frame["code"].astype(str),
        "item_name": frame["item_name"].astype(str),
        "category_group": frame["category_group"].astype(str),
        "views": frame["views"].astype("int64"),
        "purchases": frame["purchases"].astype("int64"),
        "cr_raw": 100 * frame["cr_raw"].astype(float),
        "cr_adj": 100 * frame["cr_adj"].astype(float),
        "interval": [f"{100 * a:.2f}–{100 * b:.2f}%" for a, b in zip(frame["ci_low"], frame["ci_high"])],
        "rpv": frame["rpv"].astype(float),
        "revenue_usd": frame["revenue_usd"].astype(float),
    })
    if extra:
        out[extra] = frame[extra].astype(float).to_numpy()
    out["lean"] = out["product_key"].map(lean_map).fillna("–")
    out["flags"] = flags
    return out.reset_index(drop=True)


COLUMNS = {
    "product_key": None,
    "code": st.column_config.TextColumn("Code", width="small"),
    "item_name": st.column_config.TextColumn("Product", width="large"),
    "category_group": "Category",
    "views": st.column_config.NumberColumn("Views", format="%d"),
    "purchases": st.column_config.NumberColumn("Purchases", format="%d"),
    "cr_raw": st.column_config.NumberColumn("Conversion", format="%.2f%%"),
    "cr_adj": st.column_config.NumberColumn("Adjusted", format="%.2f%%", help="Shrunk toward the category average."),
    "interval": st.column_config.TextColumn("80% likely range"),
    "rpv": st.column_config.NumberColumn("Revenue / view", format="$%.2f"),
    "revenue_usd": st.column_config.NumberColumn("Revenue", format="$%.0f"),
    "lift_per_1k": st.column_config.NumberColumn("Extra purchases / 1k views", format="%.2f"),
    "forgone_purchases": st.column_config.NumberColumn("Forgone purchases", format="%.1f"),
    "lean": "Device lean",
    "flags": st.column_config.TextColumn(
        "Flags", help="uncertain: interval overlaps the average · views under-counted: checkout events reach half its "
                      "product views, so its rate is likely overstated · cannibal pair: see Cannibalization Check"),
}

TABS = (
    ("Hidden gems: promote (answers Q2)", "Hidden gems", "lift_per_1k",
     "Low traffic, high conversion: the best candidates for a homepage slot, by extra purchases per 1,000 views."),
    ("Wasted exposure: retire (answers Q1)", "Wasted exposure", "forgone_purchases",
     "High traffic, low conversion: they use attention without selling, by purchases forgone."),
    ("Winners: keep", "Winners", None, "High traffic and high conversion: keep them visible."),
    ("All products", None, None, "Every eligible product, ranked by adjusted conversion (the efficiency score)."),
)


def _make_select(table_key: str, frame: pd.DataFrame):
    def _cb() -> None:
        key = ui.selected_row_key(table_key, frame)
        if key:
            st.session_state[state.INSPECT] = key
    return _cb


for tab, (label, quad, extra, note) in zip(st.tabs([t[0] for t in TABS]), TABS):
    with tab:
        st.caption(note + " Select a row to inspect it.")
        sel = eligible if quad is None else eligible[eligible["quadrant"] == quad]
        sort = extra or "cr_adj"
        sel = sel.sort_values([sort, "views"], ascending=[False, False], kind="mergesort")
        frame = _table(sel, extra)
        tkey = f"{P}tab_{(quad or 'all').replace(' ', '_').lower()}"
        if frame.empty:
            st.info("No products in this group for the current scope. Try lowering the minimum views.")
            continue
        ui.data_table(frame, key=tkey, column_config=COLUMNS, on_select=_make_select(tkey, frame), height=360)
        st.download_button("Download CSV", frame.drop(columns="product_key").to_csv(index=False).encode("utf-8"),
                           file_name=f"{tkey}.csv", mime="text/csv", on_click="ignore", key=f"{tkey}_csv")

ui.caveat_note()

"""Holiday Shift (Q3): does the conversion pattern change during Black Friday / Cyber Monday?"""

from __future__ import annotations


import pandas as pd
import streamlit as st

from ga4app import charts, config, data, metrics, state, ui
from ga4app.types import Filters

P = "hs_"
COMPARE_KEY = f"{P}compare"


def _fixed(f: Filters) -> Filters:
    """``f`` with the date range dropped: this page compares fixed windows."""
    return data.season_filters(f)


LOW_SAMPLE_OPACITY = 0.35


def _fade_low_sample(fig, df: pd.DataFrame):
    """Draw low-sample bars faint. ``charts.period_bars`` adds one bar trace per window, in the
    order windows first appear in ``df``, with one bar per group in first-appearance order."""
    if df is None or df.empty or not {"period", "group", "low_sample"} <= set(df.columns):
        return fig
    periods = list(dict.fromkeys(df["period"].astype(str)))
    groups = list(dict.fromkeys(df["group"].astype(str)))
    low = (df.assign(_p=df["period"].astype(str), _g=df["group"].astype(str))
           .drop_duplicates(["_p", "_g"]).set_index(["_p", "_g"])["low_sample"])
    bars = [t for t in fig.data if t.type == "bar"]
    for p, trace in zip(periods, bars):
        flags = [bool(low.get((p, g), False)) for g in groups]
        trace.marker.opacity = [LOW_SAMPLE_OPACITY if x else 1.0 for x in flags]
    return fig


@st.cache_data(show_spinner="Comparing windows…", max_entries=32)
def _periods(version: str, f: Filters, base: str, focus: str, by: str | None, _snap: data.Snapshot) -> pd.DataFrame:
    pers = [config.PERIODS[base], config.PERIODS[focus]]
    return metrics.period_comparison(_snap.item_daily, _snap.product_dim, pers, f, by=by)


@st.cache_data(show_spinner="Finding quadrant movers…", max_entries=32)
def _movers(version: str, f: Filters, base: str, focus: str, _snap: data.Snapshot) -> pd.DataFrame:
    return metrics.quadrant_movers(_snap.item_daily, _snap.product_dim, base, focus, f, data.shrinkage_m(_snap))


@st.cache_data(show_spinner=False, max_entries=16)
def _daily(version: str, f: Filters, _snap: data.Snapshot) -> pd.DataFrame:
    return metrics.daily_totals(_snap.item_daily, _snap.product_dim, f, ignore_dates=True)


def _label(pair: tuple[str, str]) -> str:
    b, fo = config.PERIODS[pair[0]], config.PERIODS[pair[1]]
    return f"{b.label} ({b.short()}) vs {fo.label} ({fo.short()})"


snapshot = data.get_snapshot()
f = _fixed(state.current_filters())

ui.header(
    "Holiday Shift",
    "Does conversion change in the holiday windows, and which categories and products move? "
    f"The sidebar dates do not apply here: the page compares fixed windows. · {f.label().split(' · ', 1)[1]}",
    ("Q3",),
)

pair = st.selectbox("Compare", list(config.COMPARISONS), format_func=_label, key=COMPARE_KEY,
                    help="The first window is the base. Only about 3 pre-holiday weeks exist (data starts Nov 1), "
                         "so a 4-week January baseline is offered as an alternative.")
base_key, focus_key = pair or config.COMPARISONS[0]
base, focus = config.PERIODS[base_key], config.PERIODS[focus_key]

site = _periods(snapshot.version, f, base_key, focus_key, None, snapshot)
if site.empty or int(site["views"].sum()) == 0:
    st.info("No product views in these windows for the current device and category filters. "
            "Clear the sidebar filters to compare windows.")
    st.stop()

b_row = site[site["period"] == base_key].iloc[0]
f_row = site[site["period"] == focus_key].iloc[0]


def _rel(new: float, old: float) -> str | None:
    return f"{100 * (new / old - 1):+.0f}% vs {base.label.lower()}" if old > 0 else None


p_site = f_row["p_value"]
sig = "significant (p < 0.05)" if pd.notna(p_site) and p_site < 0.05 else "not significant"
ui.kpi_row([
    {"label": f"Conversion · {base.label}", "value": ui.pct(b_row["cr"]), "help": f"{base.short()}, {base.days} days."},
    {"label": f"Conversion · {focus.label}", "value": ui.pct(f_row["cr"]),
     "delta": f"{100 * (f_row['cr'] - b_row['cr']):+.2f} pts", "help": f"Change vs the base window is {sig}."},
    {"label": f"Product views / day · {focus.label}", "value": f"{f_row['views_per_day']:,.0f}",
     "delta": _rel(f_row["views_per_day"], b_row["views_per_day"]),
     "help": f"{base.label}: {b_row['views_per_day']:,.0f} per day."},
    {"label": f"Purchases / day · {focus.label}", "value": f"{f_row['purchases_per_day']:,.1f}",
     "delta": _rel(f_row["purchases_per_day"], b_row["purchases_per_day"]),
     "help": f"{base.label}: {b_row['purchases_per_day']:,.1f} per day."},
])

# Daily line with both windows shaded.
daily = _daily(snapshot.version, f, snapshot)
st.subheader("Daily conversion across the season")
ui.chart_with_table(
    charts.daily_conversion(daily, [base, focus]),
    daily.assign(cr_pct=100 * daily["cr"]).drop(columns="cr"), key=f"{P}daily", file_name="holiday_shift_daily.csv",
    caption="Purchases per product view each day. Shaded bands are the two windows being compared. "
            "Purchases surge from Nov 23 and collapse after the Dec 18 shipping cutoff.",
)

# The pitch chart: conversion by category, focus vs base.
groups = _periods(snapshot.version, f, base_key, focus_key, "category_group", snapshot)
st.subheader(f"Conversion by category: {focus.label} vs {base.label}")
if groups.empty:
    st.info("No category has views in both windows for the current filters.")
else:
    cols = ["group", "period_label", "views", "purchases", "cr", "ci_low", "ci_high", "lift", "p_value",
            "lift_vs_rest", "p_value_vs_rest", "low_sample"]
    table = groups[cols].copy()
    for c in ("cr", "ci_low", "ci_high", "lift", "lift_vs_rest"):
        table[c] = 100 * table[c].astype(float)
    ui.chart_with_table(
        _fade_low_sample(charts.period_bars(groups), groups), table.round(3), key=f"{P}bars", file_name="holiday_shift_categories.csv",
        caption="Bars are conversion in each window; whiskers are 95% intervals. A % label marks a change that is "
                "statistically significant (p < 0.05). Faint bars have fewer than 10 purchases.",
    )
    focus_rows = groups[~groups["is_base"].astype(bool)]
    up, down, flat = [], [], []
    for r in focus_rows.itertuples():
        if pd.isna(r.p_value) or r.low_sample:
            continue
        txt = f"{r.group} ({100 * r.lift:+.0f}%)"
        if r.p_value < 0.05:
            (up if r.lift > 0 else down).append(txt)
        else:
            flat.append(str(r.group))
    beats = [str(r.group) for r in focus_rows.itertuples()
             if pd.notna(r.p_value_vs_rest) and r.p_value_vs_rest < 0.05 and r.lift_vs_rest > 0]
    lines = []
    if up:
        lines.append(f"- **Converts better in {focus.label}:** " + ", ".join(up))
    if down:
        lines.append(f"- **Converts worse in {focus.label}:** " + ", ".join(down))
    if flat:
        lines.append("- **No clear change:** " + ", ".join(flat))
    if beats:
        lines.append("- **Shifts more than the rest of the store** (its own holiday effect, beyond the site-wide surge): "
                     + ", ".join(beats))
    if not lines:
        lines.append("- Too few purchases in these windows to call any category shift.")
    st.markdown(ui.esc("\n".join(lines)))

# Quadrant movers.
st.subheader("Which products change quadrant?")
movers = _movers(snapshot.version, f, base_key, focus_key, snapshot)
both = movers[movers["status"] != "Insufficient data"] if not movers.empty else movers
if both.empty:
    st.info(f"No product has at least {f.min_views:,} views in both windows. Lower the minimum views in the sidebar.")
else:
    counts = metrics.transition_counts(movers)
    left, right = st.columns([1.1, 1])
    with left:
        ui.chart_with_table(
            charts.transition_matrix(counts, base_label=base.label, focus_label=focus.label),
            counts.reset_index(), key=f"{P}matrix", file_name="holiday_shift_transitions.csv",
            caption=f"Each cell counts products by their quadrant in {base.label} (rows) and {focus.label} (columns). "
                    "Off-diagonal cells are products that moved. Each window uses its own thresholds.",
        )
    with right:
        moved = movers[movers["moved"].astype(bool)]
        new_gems = moved[moved["quadrant_focus"] == "Hidden gems"]
        new_wasted = moved[moved["quadrant_focus"] == "Wasted exposure"]
        ui.kpi_row([
            {"label": "Products in both windows", "value": f"{len(both):,}"},
            {"label": "Changed quadrant", "value": f"{len(moved):,}"},
        ])
        ui.kpi_row([
            {"label": "New Hidden gems", "value": f"{len(new_gems):,}"},
            {"label": "New Wasted exposure", "value": f"{len(new_wasted):,}"},
        ])

    def _mv_table(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "product_key": frame["product_key"].astype(str),
            "item_name": frame["item_name"].astype(str),
            "category_group": frame["category_group"].astype(str),
            "quadrant_base": frame["quadrant_base"].astype(str),
            "quadrant_focus": frame["quadrant_focus"].astype(str),
            "cr_adj_base": 100 * frame["cr_adj_base"].astype(float),
            "cr_adj_focus": 100 * frame["cr_adj_focus"].astype(float),
            "rel_change": 100 * frame["rel_change"].astype(float),
            "views_focus": frame["views_focus"].astype("int64"),
            "status": frame["status"].astype(str),
        }).reset_index(drop=True)

    mv_cols = {
        "product_key": None,
        "item_name": st.column_config.TextColumn("Product", width="large"),
        "category_group": "Category",
        "quadrant_base": base.label,
        "quadrant_focus": focus.label,
        "cr_adj_base": st.column_config.NumberColumn(f"Adjusted · {base.label}", format="%.2f%%"),
        "cr_adj_focus": st.column_config.NumberColumn(f"Adjusted · {focus.label}", format="%.2f%%"),
        "rel_change": st.column_config.NumberColumn("Change", format="%+.0f%%"),
        "views_focus": st.column_config.NumberColumn(f"Views · {focus.label}", format="%d"),
        "status": st.column_config.TextColumn("Status", help="Rising / Fading = a significant change of at least 25%."),
    }
    tabs = st.tabs([f"Became Hidden gems in {focus.label}", "Top risers", "Top fallers", "Lost Winner status"])
    risers = both[both["rel_change"] > 0].sort_values("z", ascending=False, kind="mergesort").head(15)
    fallers = both[both["rel_change"] < 0].sort_values("z", ascending=True, kind="mergesort").head(15)
    lost = moved[moved["quadrant_base"] == "Winners"].sort_values("z", kind="mergesort")
    sets = (
        (new_gems.sort_values("z", ascending=False, kind="mergesort"), "gems",
         f"Products that were not Hidden gems in {base.label} but are in {focus.label}: candidates for a holiday slot."),
        (risers, "risers", "Biggest gains in adjusted conversion (ranked by strength of evidence)."),
        (fallers, "fallers", "Biggest drops in adjusted conversion (ranked by strength of evidence)."),
        (lost, "lost", f"Winners in {base.label} that left the Winners quadrant in {focus.label}."),
    )
    for tab, (frame, name, note) in zip(tabs, sets):
        with tab:
            st.caption(note)
            if frame.empty:
                st.info("No products in this list for the selected windows.")
                continue
            ui.data_table(_mv_table(frame), key=f"{P}mv_{name}", column_config=mv_cols)

with st.expander("Caveats for this page"):
    st.markdown(ui.esc("\n".join([
        f"- Add-to-cart and checkout were not tracked reliably before {config.CART_TRACKING_START:%b %d}, "
        "so this page compares purchases per product view only, never cart rates.",
        "- The windows come from the data: purchases surged on Nov 23 (before Thanksgiving) and collapsed after "
        "the Dec 18 shipping cutoff.",
        "- Only about 3 pre-holiday weeks exist (data starts Nov 1). The January 4-week baseline is the alternative, "
        "and January demand is not the same as November demand, so treat both baselines as approximate.",
        "- Lifts are tested with a two-proportion z-test; 'shifts more than the rest' compares the category's change "
        "with the rest of the store pooled. Categories with fewer than 10 purchases in a window are not tested.",
    ])))
ui.caveat_note()

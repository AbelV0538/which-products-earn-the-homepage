"""Slot Plan: the decision. Which products fill each holiday campaign's homepage slots."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ga4app import config, data, metrics, narrative, state, ui

P = "sp_"
# Trend alerts per campaign: Google momentum in the campaign's window vs the pre-holiday baseline.
TREND_FOCUS = {"Black Friday": "cyber_week", "Cyber Monday": "cyber_week", "December gifting": "december"}

snapshot = data.get_snapshot()
f = state.current_filters()
season_f = data.season_filters(f)

ui.header(
    "Slot Plan",
    "The decision: the best products for each campaign's limited homepage slots, scored on each window's own "
    f"evidence borrowed against the full season · {f.label()} (dates are set by each campaign)",
    ("Q1", "Q2"),
)

c1, c2, c3 = st.columns([1, 1.4, 1.4])
n_slots = int(c1.number_input("Slots per campaign", min_value=1, max_value=12, value=config.DEFAULT_SLOTS, step=1, key=f"{P}slots"))
avoid = c2.toggle("Avoid cannibalizing pairs", value=True, key=f"{P}avoid",
                  help="Never put two products that took sales from each other in the same campaign.")
repeats = c3.toggle("Allow repeats across campaigns", value=False, key=f"{P}repeats")

scores = {c: data.campaign_scores(snapshot, c, f) for c in config.CAMPAIGNS}
cannibal = data.season_cannibal(snapshot)
pairs = data.conflicts(snapshot)
plan = metrics.slot_plan(scores, n_slots, conflicts=pairs if avoid else frozenset(), allow_repeats=repeats)

lean = data.device_lean(snapshot, tuple(plan["product_key"].astype(str)), f)
lean_map = dict(zip(lean["product_key"].astype(str), lean["lean"].astype(str)))
partners: dict[str, set[str]] = {}
for pair in pairs:
    a, b = sorted(pair)
    partners.setdefault(a, set()).add(b)
    partners.setdefault(b, set()).add(a)
trend_groups: dict[str, set[str]] = {}
if snapshot.has_trends:
    for campaign, focus in TREND_FOCUS.items():
        alerts = data.trending_alerts(snapshot, f, "baseline", focus)
        if alerts is not None and len(alerts):
            flagged = alerts[alerts["flagged"].astype(bool) & (alerts["group_type"] == "category")]
            trend_groups[campaign] = set(flagged["group"].astype(str))

LEAN_BADGE = {"Mobile-first": ("Mobile-first", "blue"), "Desktop-first": ("Desktop-first", "violet"),
              "Both": ("Converts on both devices", "gray")}


def _times(vs: float) -> str:
    """vs_avg (score / average - 1) as a multiple: one decimal below 10x, otherwise whole."""
    m = 1.0 + float(vs)
    return f"{m:.1f}x" if m < 10 else f"{m:.0f}x"


def _multiple(vs, avg) -> str:
    if pd.isna(vs) or pd.isna(avg):
        return ""
    return f" · {_times(vs)} the {1000 * float(avg):.1f} average"


def _card(row: pd.Series, campaign: str, picked: set[str]) -> None:
    key = str(row["product_key"])
    with st.container(border=True):
        st.markdown(f"**{int(row['slot'])}. {ui.esc(row['item_name'])}**")
        st.caption(ui.esc(row["category_group"]))
        vs_txt = _multiple(row["vs_avg"], row["window_avg"])
        st.markdown(f"{row['score_per_1k']:.1f} purchases per 1,000 views{vs_txt}")
        st.caption(f"In this window: {int(row['views_w']):,} views, {int(row['purchases_w']):,} purchases.")
        badge = LEAN_BADGE.get(lean_map.get(key, ""))
        if badge:
            st.badge(badge[0], color=badge[1])
        if str(row["category_group"]) in trend_groups.get(campaign, set()):
            st.badge("Trending on Google", color="green")
        clash = partners.get(key, set())
        if clash:
            same = clash & picked
            names = ", ".join(data.name_of(snapshot, k) for k in sorted(clash))
            msg = f"Cannibal risk: sales dipped when featured next to {names}."
            st.warning(("Also in this campaign. " if same else "") + msg, icon=":material/warning:")


cols = st.columns(len(config.CAMPAIGNS))
for col, campaign in zip(cols, config.CAMPAIGNS):
    period = config.PERIODS[config.CAMPAIGNS[campaign]]
    with col:
        st.subheader(campaign)
        st.caption(f"{period.short()} · scored on that window")
        rows = plan[plan["campaign"] == campaign]
        if rows.empty:
            st.info("No eligible products. Lower the minimum views or widen the categories.")
        picked = set(rows["product_key"].astype(str))
        for _, row in rows.iterrows():
            _card(row, campaign, picked)

st.caption("The Black Friday and Cyber Monday windows are only 4 and 2 days, so each score leans on the product's "
           "full-season rate, adjusted by how its category moved in that window. A product can therefore rank highly "
           "with few window views.")

# Retire list: the worst Wasted-exposure products over the full season.
season_products, season_th = data.products(snapshot, season_f)
retire = (season_products[season_products["quadrant"] == "Wasted exposure"]
          .sort_values("forgone_purchases", ascending=False).head(max(n_slots, 5)))
st.subheader("Retire from the homepage")
st.caption("High-traffic products that convert below average over the full season, by purchases forgone. "
           "Their homepage exposure is better spent on the picks above.")
retire_view = pd.DataFrame({
    "Product": retire["item_name"].astype(str),
    "Category": retire["category_group"].astype(str),
    "Views": retire["views"].astype("int64"),
    "Conversion": 100 * retire["cr_raw"].astype(float),
    "Forgone purchases": retire["forgone_purchases"].astype(float),
})
if retire_view.empty:
    st.info("No Wasted-exposure products in this scope.")
else:
    st.dataframe(retire_view, hide_index=True, width="stretch", key=f"{P}retire", column_config={
        "Conversion": st.column_config.NumberColumn(format="%.2f%%"),
        "Forgone purchases": st.column_config.NumberColumn(format="%.1f")})

pinned = list(st.session_state.get(state.PINNED) or [])
if pinned:
    with st.expander(f"Your shortlist from Feature Picks ({len(pinned)})"):
        rows = []
        for k in pinned:
            row = {"Product": data.name_of(snapshot, k)}
            for campaign, frame in scores.items():
                hit = frame.loc[frame["product_key"] == k, "score_per_1k"]
                row[f"{campaign} / 1k views"] = float(hit.iloc[0]) if len(hit) else float("nan")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key=f"{P}pinned_table")
        st.button("Clear shortlist", key=f"{P}clear_pins", on_click=lambda: st.session_state.update({state.PINNED: []}))

# Downloads.
export = plan.assign(
    device_lean=plan["product_key"].astype(str).map(lean_map).fillna(""),
    cannibal_partners=plan["product_key"].astype(str).map(lambda k: "; ".join(sorted(partners.get(k, set())))),
)
md = ["# Holiday homepage slot plan", "", f"Scope: {f.label()}", ""]
for campaign in config.CAMPAIGNS:
    md.append(f"## {campaign}")
    for _, r in plan[plan["campaign"] == campaign].iterrows():
        vs = "" if pd.isna(r["vs_avg"]) else f" ({_times(r['vs_avg'])} the average)"
        md.append(f"{int(r['slot'])}. {r['item_name']}: {r['score_per_1k']:.1f} purchases per 1,000 views{vs}")
    md.append("")
md.append("## Retire from the homepage")
md += [f"- {r.item_name}: {r.views:,} views, {100 * r.cr_raw:.2f}% conversion" for r in retire.itertuples()]
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("Download plan (CSV)", export.to_csv(index=False).encode("utf-8"), file_name="slot_plan.csv",
                   mime="text/csv", on_click="ignore", key=f"{P}csv", width="stretch")
d2.download_button("Download plan (Markdown)", "\n".join(md).encode("utf-8"), file_name="slot_plan.md",
                   mime="text/markdown", on_click="ignore", key=f"{P}md", width="stretch")

# Optional AI verdict on the plan.
st.subheader("Verdict on the plan")
codes = season_products.set_index("product_key")["code"].astype(str).to_dict()
payload = narrative.build_payload(season_products, season_th, season_f, window_label="Full season (slot plan)", cannibal=cannibal)
payload["slot_plan"] = [
    {"campaign": r["campaign"], "slot": int(r["slot"]), "name": str(r["item_name"]),
     **({"code": codes[r["product_key"]]} if codes.get(r["product_key"]) else {}),
     "per_1k_views": round(float(r["score_per_1k"]), 2)}
    for _, r in plan.iterrows()
]
ui.verdict_block(payload, key_prefix=P)
ui.caveat_note()

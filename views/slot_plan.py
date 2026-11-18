"""Slot Plan: the decision. Which products fill each holiday campaign's homepage slots."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ga4app import charts, config, data, metrics, narrative, state, ui

P = "sp_"
INCLUDE_W, EXCLUDE_W = f"{P}include", f"{P}exclude"  # widget keys; state.PINNED / EXCLUDED hold the lists

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

season_products, season_th = data.products(snapshot, season_f)
names = dict(zip(season_products["product_key"].astype(str), season_products["item_name"].astype(str)))
by_name = sorted(names, key=lambda k: names[k].lower())


def _sync(widget: str, target: str, visible: frozenset[str]) -> None:
    """Copy a multiselect into its list, keeping entries this scope cannot show (e.g. another category)."""
    hidden = [k for k in st.session_state.get(target) or [] if k not in visible]
    st.session_state[target] = hidden + list(st.session_state.get(widget) or [])


pinned_now = [k for k in st.session_state.get(state.PINNED) or [] if k in names]
excluded_now = [k for k in st.session_state.get(state.EXCLUDED) or [] if k in names]
include_opts = [k for k in by_name if k not in excluded_now]
exclude_opts = [k for k in by_name if k not in pinned_now]
st.session_state[INCLUDE_W] = [k for k in pinned_now if k in include_opts]
st.session_state[EXCLUDE_W] = [k for k in excluded_now if k in exclude_opts]
i1, i2 = st.columns(2)
i1.multiselect(
    "Always include", include_opts, key=INCLUDE_W, format_func=names.get, placeholder="Pin products to the plan",
    on_change=_sync, args=(INCLUDE_W, state.PINNED, frozenset(include_opts)),
    help="Each pinned product takes a slot in the campaign where it scores best. \"Add to slot plan\" in the "
         "Feature Picks inspector pins products here too.",
)
i2.multiselect(
    "Exclude (e.g. out of stock)", exclude_opts, key=EXCLUDE_W, format_func=names.get, placeholder="Keep products out",
    on_change=_sync, args=(EXCLUDE_W, state.EXCLUDED, frozenset(exclude_opts)),
    help="The plan never uses these; the next-best product takes the slot.",
)
pinned = list(st.session_state.get(state.PINNED) or [])
excluded = list(st.session_state.get(state.EXCLUDED) or [])

scores = {c: data.campaign_scores(snapshot, c, f) for c in config.CAMPAIGNS}
cannibal = data.season_cannibal(snapshot)
pairs = data.conflicts(snapshot)
plan = metrics.slot_plan(scores, n_slots, conflicts=pairs if avoid else frozenset(), allow_repeats=repeats,
                         exclude=excluded, must_include=pinned)
unplaced = list(plan.attrs.get("unplaced") or [])
if unplaced:
    st.warning(
        "Could not place " + ", ".join(ui.esc(data.name_of(snapshot, k)) for k in unplaced) + ": it has no views in "
        "this scope, or every campaign is full or already holds a product it cannibalizes. Add slots per campaign "
        "or unpin something.", icon=":material/info:",
    )

lean = data.device_lean(snapshot, tuple(plan["product_key"].astype(str)), f)
lean_map = dict(zip(lean["product_key"].astype(str), lean["lean"].astype(str)))
undercounted = set(season_products.loc[season_products["views_undercount"].astype(bool), "product_key"].astype(str))
partners: dict[str, set[str]] = {}
for pair in pairs:
    a, b = sorted(pair)
    partners.setdefault(a, set()).add(b)
    partners.setdefault(b, set()).add(a)
trend_groups: dict[str, set[str]] = {}
if snapshot.has_trends:
    for campaign, focus in config.CAMPAIGN_TREND_FOCUS.items():  # Google momentum vs the pre-holiday baseline
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


def _range(row) -> str:
    """The 80% range per 1,000 views, or "" when the interval is missing."""
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{1000 * float(lo):.0f}–{1000 * float(hi):.0f}"


def _card(row: pd.Series, campaign: str, picked: set[str]) -> None:
    key = str(row["product_key"])
    with st.container(border=True):
        st.markdown(f"**{int(row['slot'])}. {ui.esc(row['item_name'])}**")
        st.caption(ui.esc(row["category_group"]))
        vs_txt = _multiple(row["vs_avg"], row["window_avg"])
        st.markdown(f"{row['score_per_1k']:.1f} purchases per 1,000 views{vs_txt}")
        rng = _range(row)
        evidence = f"Window: {int(row['views_w']):,} views, {int(row['purchases_w']):,} purchases"
        if rng:
            evidence += f" · 80% range {rng}"
        if bool(row.get("thin_window")):
            evidence += f" · {float(row['window_weight']):.0%} of the score from the window"
        st.caption(evidence + ".")
        badges = []  # (text, colour), drawn inline on one line
        if bool(row.get("pinned")):
            badges.append(("Pinned", "primary"))
        if bool(row.get("thin_window")):
            badges.append(("Mostly season evidence", "gray"))
        lean_badge = LEAN_BADGE.get(lean_map.get(key, ""))
        if lean_badge:
            badges.append(lean_badge)
        if str(row["category_group"]) in trend_groups.get(campaign, set()):
            badges.append(("Trending on Google", "green"))
        if key in undercounted:
            badges.append(("Views under-counted", "orange"))
        if badges:
            st.markdown(" ".join(f":{colour}-badge[{text}]" for text, colour in badges))
        clash = partners.get(key, set())
        if clash:
            same = clash & picked
            clash_names = ", ".join(data.name_of(snapshot, k) for k in sorted(clash))
            msg = f"Cannibal risk: sales dipped when featured next to {clash_names}."
            st.warning(("Also in this campaign. " if same else "") + msg, icon=":material/warning:")


cols = st.columns(len(config.CAMPAIGNS))
for col, campaign in zip(cols, config.CAMPAIGNS):
    period = config.PERIODS[config.CAMPAIGNS[campaign]]
    with col:
        st.subheader(campaign)
        st.caption(f"{period.short()} · {period.days} days · scored on that window")
        rows = plan[plan["campaign"] == campaign]
        if rows.empty:
            st.info("No eligible products. Lower the minimum views or widen the categories.")
        picked = set(rows["product_key"].astype(str))
        for _, row in rows.iterrows():
            _card(row, campaign, picked)

st.caption(
    "The Black Friday and Cyber Monday windows are only 4 and 2 days, so a product sees a few dozen views at most "
    "there. Its score is then mostly its full-season rate, adjusted by how its category moved in the window "
    "(\"Mostly season evidence\"). \"Views under-counted\": checkout events reach half the product's views, so its "
    "rate is likely overstated; confirm it before giving it a lead slot."
)

# How far ahead are the picks? One campaign's candidates with their ranges.
st.subheader("How far ahead are the picks?")
campaigns = list(config.CAMPAIGNS)
which = st.segmented_control("Campaign", campaigns, default=campaigns[0], key=f"{P}chart_campaign") or campaigns[0]
cand = scores[which]
picked_keys = plan.loc[plan["campaign"] == which, "product_key"].astype(str)
cand_table = pd.DataFrame()
if cand is not None and len(cand):
    top = cand[cand["eligible"].astype(bool) | cand["product_key"].astype(str).isin(set(picked_keys))]
    top = top.sort_values("score_per_1k", ascending=False, kind="mergesort").head(25)
    cand_table = pd.DataFrame({
        "product": top["item_name"].astype(str),
        "in_plan": top["product_key"].astype(str).isin(set(picked_keys)),
        "per_1k_views": top["score_per_1k"].astype(float).round(1),
        "range_low": (1000 * top["ci_low"].astype(float)).round(1),
        "range_high": (1000 * top["ci_high"].astype(float)).round(1),
        "window_views": top["views_w"].astype("int64"),
        "window_purchases": top["purchases_w"].astype("int64"),
        "window_share_of_score": top["window_weight"].astype(float).round(3),
    })
ui.chart_with_table(
    charts.slot_candidates(cand, picked_keys), cand_table, key=f"{P}candidates",
    file_name=f"slot_candidates_{config.CAMPAIGNS[which]}.csv",
    caption=(f"The best-scoring candidates for {which}: expected purchases per 1,000 views with an 80% range; filled "
             "dots are in the plan. Where ranges overlap, the order between those products is not settled by the data. "
             "The right-hand column counts each product's views inside the window."),
)

# Retire list: the worst Wasted-exposure products over the full season.
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

# Downloads.
export = plan.assign(
    device_lean=plan["product_key"].astype(str).map(lean_map).fillna(""),
    cannibal_partners=plan["product_key"].astype(str).map(lambda k: "; ".join(sorted(partners.get(k, set())))),
    views_undercount=plan["product_key"].astype(str).isin(undercounted),
)
md = ["# Holiday homepage slot plan", "", f"Scope: {f.label()}", ""]
for campaign in config.CAMPAIGNS:
    md.append(f"## {campaign}")
    for _, r in plan[plan["campaign"] == campaign].iterrows():
        vs = "" if pd.isna(r["vs_avg"]) else f" ({_times(r['vs_avg'])} the average)"
        rng = _range(r)
        notes = [f"80% range {rng}"] if rng else []
        notes.append(f"{int(r['views_w']):,} window views")
        if bool(r.get("thin_window")):
            notes.append("mostly season evidence")
        if bool(r.get("pinned")):
            notes.append("pinned")
        md.append(f"{int(r['slot'])}. {r['item_name']}: {r['score_per_1k']:.1f} purchases per 1,000 views{vs}; "
                  + ", ".join(notes))
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
payload = narrative.build_payload(
    season_products, season_th, season_f, window_label="Full season (slot plan)", cannibal=cannibal,
    slot_plan=plan, **data.verdict_context(snapshot, f),
)
ui.verdict_block(payload, key_prefix=P)
ui.caveat_note()

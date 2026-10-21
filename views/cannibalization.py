"""Cannibalization Check (Phase 2a): when one product is pushed, do its same-category peers sell less?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ga4app import charts, config, data, metrics, phase2, state, ui
from ga4app.types import Filters

P = "cn_"
FLAGGED = (phase2.LIKELY, phase2.POSSIBLE)


@st.cache_data(show_spinner="Finding exposure spikes…", max_entries=16)
def _spikes(version: str, f: Filters, ratio: float, _snap: data.Snapshot) -> pd.DataFrame:
    return phase2.exposure_spikes(_snap.item_daily, _snap.product_dim, f, ratio=ratio)


@st.cache_data(show_spinner="Testing product pairs…", max_entries=16)
def _tests(version: str, f: Filters, ratio: float, _snap: data.Snapshot) -> pd.DataFrame:
    spikes = _spikes(version, f, ratio, _snap)
    return phase2.cannibalization_tests(_snap.item_daily, _snap.product_dim, spikes, f)


def _fmt_p(x: float) -> str:
    return "–" if x is None or pd.isna(x) or not np.isfinite(x) else (f"{x:.1e}" if x < 0.001 else f"{x:.3f}")


def _fmt_num(x, fmt: str) -> str:
    """Format a number, or an en dash when it is missing (never the string "None")."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(v) else format(v, fmt)


def _default_a(tests: pd.DataFrame, candidates: list[str], fallback: str) -> str:
    """Featured product of the strongest flagged pair (Likely first, lowest q), else the most testable peers."""
    flagged = tests[tests["status"].isin(FLAGGED) & tests["product_a"].astype(str).isin(candidates)]
    if len(flagged):
        order = flagged.assign(_rank=(flagged["status"] != phase2.LIKELY).astype(int),
                               _q=flagged["q_value"].astype(float).fillna(np.inf))
        return str(order.sort_values(["_rank", "_q"], kind="mergesort").iloc[0]["product_a"])
    n_tested = tests[tests["tested"].astype(bool)].groupby(tests["product_a"].astype(str)).size()
    n_tested = n_tested[n_tested.index.isin(candidates)]
    if len(n_tested) and int(n_tested.max()) > 0:
        return str(n_tested.idxmax())
    return fallback


snapshot = data.get_snapshot()
f = state.current_filters()

ui.header(
    "Cannibalization Check",
    "Before giving a product a homepage slot: when it got a burst of attention, did similar products in its "
    "category sell less? · " + f.label(),
    ("Phase 2a", "Q1", "Q2"),
)
st.info(
    "**How \"featured\" is measured.** The store has no history of which products sat on the homepage, and its "
    "promotion events carry banner names, not product ids. So a product counts as featured on the days its share "
    "of its category's views spiked (at least the chosen multiple of its usual share). On those days we test whether "
    "each same-category peer bought less than its other peers would predict, compared with its own pace on nearby "
    "non-spike days (within a week of a spike)."
)

# Candidates for product A: products that are among their category's most-viewed (the tested set).
f_tests = Filters(start=f.start, end=f.end, devices=f.devices, categories=f.categories, min_views=0)
c1, c2 = st.columns([2.4, 1])
ratio = c2.segmented_control(
    "Spike = share at least", list(config.CANNIBAL["spike_ratio_options"]), default=config.CANNIBAL["spike_ratio"],
    key=f"{P}ratio", format_func=lambda r: f"{r:g}x usual",
    help="How far above its usual share of category views a product must jump for the day to count as featured.",
) or config.CANNIBAL["spike_ratio"]
tests = _tests(snapshot.version, f_tests, float(ratio), snapshot)
spikes = _spikes(snapshot.version, f_tests, float(ratio), snapshot)

if tests.empty:
    st.info("No products have enough daily views in this scope to test. Widen the dates or clear the filters.")
    ui.caveat_note()
else:
    candidates = list(dict.fromkeys(tests["product_a"].astype(str)))
    products, _th = data.products(snapshot, f)
    picks = metrics.feature_candidates(products, n=len(products)) if len(products) else products
    fallback = next((k for k in picks.get("product_key", pd.Series(dtype=str)).astype(str) if k in candidates), candidates[0])
    default = _default_a(tests, candidates, fallback)
    key_a = c1.selectbox("Product A (the one being featured)", candidates, index=candidates.index(default),
                         key=f"{P}a", format_func=lambda k: data.name_of(snapshot, k),
                         help="Defaults to the featured product of the strongest flagged pair.")
    pair = tests[tests["product_a"].astype(str) == key_a]
    name_a = data.name_of(snapshot, key_a)
    n_spikes = int(spikes.loc[spikes["product_key"].astype(str) == key_a, "is_spike"].sum())
    tested = pair[pair["tested"].astype(bool)]
    hits = pair[pair["status"].isin(FLAGGED)]

    if hits.empty:
        summary = (f"On {n_spikes} spike day(s) for **{ui.esc(name_a)}**, none of its {len(tested)} testable "
                   "same-category peers sold noticeably less: no sign it takes sales from them.")
    else:
        worst = hits.sort_values("rr").iloc[0]
        summary = (f"On {n_spikes} spike day(s) for **{ui.esc(name_a)}**, {len(hits)} peer(s) sold less than expected; "
                   f"the largest dip is **{ui.esc(worst['item_name_b'])}** at {100 * worst['dip']:.0f}% below "
                   f"expected ({ui.esc(worst['status'])}).")
    st.markdown(summary)
    ui.kpi_row([
        {"label": "Spike days for A", "value": f"{n_spikes:,}"},
        {"label": "Peers tested", "value": f"{len(tested):,}", "help": "Peers with enough purchases on both kinds of day."},
        {"label": "Peers flagged", "value": f"{len(hits):,}"},
    ])

    st.subheader("Product A's share of its category's views")
    fig = charts.spike_share(spikes, key_a, ["cyber_week", "december"])
    one = spikes[spikes["product_key"].astype(str) == key_a]
    ui.chart_with_table(fig, one[["event_date", "views", "category_views", "share", "baseline_share", "is_spike"]],
                        key=f"{P}share", file_name="spike_share.csv",
                        caption="Marked days are spike days: A's share jumped to the chosen multiple of its usual share. "
                                "These stand in for days A was featured.")

    st.subheader("Did peers sell less on those days?")
    if tested.empty:
        min_days = int(config.CANNIBAL["min_spike_days"])
        if n_spikes < min_days:
            why = (f"{ui.esc(name_a)} had only {n_spikes} spike day(s) at this ratio; at least {min_days} are "
                   "needed before any peer can be tested.")
        else:
            why = (f"{ui.esc(name_a)} had {n_spikes} spike days, but none of its same-category peers had enough "
                   "views or purchases on those days to test.")
        st.info(f"**No peers could be tested for this product.** {why} Try a lower spike ratio (above right) "
                "or pick another product A.")
    else:
        fig = charts.rr_dotplot(pair)
        cols = ["item_name_b", "rr", "rr_low", "rr_high", "dip", "p_value", "q_value", "fisher_p", "status", "reason"]
        ui.chart_with_table(fig, pair[cols], key=f"{P}rr", file_name="cannibalization_pairs.csv",
                            caption="Each dot is a peer's relative risk: its purchases on A's spike days vs expected, "
                                    "relative to other days. 1 = no change; left of 1 = sold less. Lines are 95% intervals.")

    table = pd.DataFrame({
        "Peer (B)": pair["item_name_b"].astype(str),
        "Relative risk": pair["rr"].map(lambda x: _fmt_num(x, ".2f")),
        "Dip %": pair["dip"].map(lambda x: _fmt_num(100 * x if x is not None else None, ".0f")),
        "p": pair["p_value"].astype(float).map(_fmt_p),
        "q (BH)": pair["q_value"].astype(float).map(_fmt_p),
        "Fisher p": pair["fisher_p"].astype(float).map(_fmt_p),
        "Status": pair["status"].astype(str),
        "Spike-day views": pair["spike_views_b"].map(lambda x: _fmt_num(x, ",.0f")),
        "Why": pair["reason"].fillna("").astype(str),
    })
    ui.data_table(table, key=f"{P}pairs", height=320)

    st.subheader("Flagged pairs across the top candidates")
    flagged = tests[tests["status"].isin(FLAGGED)]
    if flagged.empty:
        st.success("No pair among each category's most-viewed products shows a likely or possible dip.")
    else:
        rollup = (flagged.groupby("category_group", observed=True)
                  .agg(pairs=("status", "size"), likely=("status", lambda s: int((s == phase2.LIKELY).sum())))
                  .reset_index().rename(columns={"category_group": "Category", "pairs": "Flagged pairs",
                                                 "likely": "Likely cannibalized"}))
        ui.data_table(rollup, key=f"{P}rollup")
        detail = flagged[["category_group", "item_name_a", "item_name_b", "rr", "q_value", "status"]].rename(columns={
            "category_group": "Category", "item_name_a": "Featured (A)", "item_name_b": "Peer (B)",
            "rr": "Relative risk", "q_value": "q", "status": "Status"})
        ui.data_table(detail, key=f"{P}flagged")

    st.warning(
        "**Correlation, not causation.** A spike day is not a confirmed homepage placement, and other things change "
        "on the same days (campaigns, emails, stock-outs). A flagged pair is a reason to avoid featuring both "
        "products together, not proof that one steals sales from the other."
    )
    st.caption(
        "Purchases cluster by day (one busy day moves many orders together), but the tests treat each purchase as "
        "independent, so p and q are optimistic. Treat a flag as a prompt to check the pair, not a finding."
    )
    with st.expander("Method"):
        st.markdown(
            f"- Pairs: each category's {config.CANNIBAL['top_k']} most-viewed products, tested in both directions.\n"
            "- Expected purchases for B on each day come from its same-category peers (excluding A and B), so "
            "site-wide holiday surges cancel out.\n"
            "- Baseline: B's usual pace is measured on A's non-spike days within 7 days of one of A's spike days, "
            "so B's own seasonal swings (say, a December peak against January spikes) are not read as a dip.\n"
            "- Test: one-sided exact binomial test for fewer purchases on spike days; q-values are Benjamini-Hochberg "
            "adjusted across all tested pairs; Fisher's exact test checks B's share of peer purchases as a second view.\n"
            f"- Likely cannibalized: relative risk at most {config.CANNIBAL['rr_flag']} and q at most "
            f"{config.CANNIBAL['q_flag']}. Possible: same relative risk with p at most {config.CANNIBAL['p_possible']}.\n"
            f"- Guards: at least {config.CANNIBAL['min_spike_days']} spike days, {config.CANNIBAL['min_other_days']} "
            f"nearby other days and {config.CANNIBAL['min_purchases']} purchases for B; otherwise \"Insufficient data\"."
        )
    ui.caveat_note()

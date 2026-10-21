"""Outside Signal (Phase 2b): categories trending on Google faster than on our own site."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ga4app import charts, config, data, phase2, state, ui
from ga4app.types import Filters

P = "os_"
# (base, focus) comparisons offered here; the first is the default.
COMPARISONS = (("baseline", "december"), ("baseline", "cyber_week"), ("cyber_week", "december"))
MANUAL_URL = ("https://trends.google.com/trends/explore?date=2020-10-01%202021-01-31&geo=US&q=<keyword>&hl=en-US")


@st.cache_data(show_spinner="Indexing Google Trends…", max_entries=16)
def _index(version: str, f: Filters, base: str, _snap: data.Snapshot) -> pd.DataFrame:
    return phase2.trend_index(_snap.trends_daily, _snap.item_daily, _snap.product_dim, _snap.keyword_map,
                              config.PERIODS[base], f)


def _no_data(meta: dict) -> None:
    reason = "Google Trends data has not been fetched yet"
    if meta.get("status") == "unavailable":
        reason = "the last Google Trends fetch returned no usable data (status: unavailable)"
    st.info(
        f"**No outside signal to show:** {reason}. Nothing is estimated or filled in, so this page stays empty.\n\n"
        "To fetch it, run from the project root:\n\n"
        "```\npython scripts/fetch_trends.py\npython scripts/build_snapshots.py\n```\n\n"
        "If Google rate-limits the fetch, download each keyword's *Interest over time* CSV by hand from "
        f"`{MANUAL_URL}` into `data/trends_manual/`, then run `python scripts/fetch_trends.py --methods manual_csv`. "
        "The keywords are listed in `config/trends_keywords.csv`."
    )


def _gem_table(alert: pd.Series, scored: pd.DataFrame) -> pd.DataFrame:
    raw = alert["promote_ahead"]
    keys = [str(k) for k in raw] if isinstance(raw, (list, tuple, pd.Series)) or hasattr(raw, "tolist") else []
    if not keys:
        return pd.DataFrame()
    rows = scored.set_index("product_key").reindex(keys)
    return pd.DataFrame({
        "Product": [data.name_of(snapshot, k) for k in keys],
        "Views": rows["views"].fillna(0).astype("int64").to_numpy(),
        "Purchases": rows["purchases"].fillna(0).astype("int64").to_numpy(),
        "Conversion": (100 * rows["cr_raw"].astype(float)).round(2).to_numpy(),
        "Adjusted": (100 * rows["cr_adj"].astype(float)).round(2).to_numpy(),
    })


def _rel(x) -> str:
    """m_ext_rel as text; an en dash when there was no other category to benchmark against."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if v != v else f"{v:.2f}x"


def _best_lag(ll: pd.DataFrame) -> pd.DataFrame:
    """Per group: the lag (weeks Google leads) with the strongest correlation; descriptive only."""
    ok = ll.dropna(subset=["corr"])
    if ok.empty:
        return pd.DataFrame(columns=["group_type", "group", "lead_weeks", "lead_corr"])
    ok = ok.assign(_abs=ok["corr"].abs()).sort_values(["group_type", "group", "_abs"], ascending=[True, True, False])
    best = ok.groupby(["group_type", "group"], as_index=False).head(1)
    return best.rename(columns={"lag": "lead_weeks", "corr": "lead_corr"})[["group_type", "group", "lead_weeks", "lead_corr"]]


snapshot = data.get_snapshot()
f = state.current_filters()
meta = snapshot.trends_meta or {}

ui.header(
    "Outside Signal",
    "Trending outside our walls: categories whose Google search interest is rising faster than their views on our "
    "store, so their best converters can be promoted ahead of demand. The sidebar dates do not apply here: the page "
    "compares fixed windows. · " + f.label().split(" · ", 1)[1],
    ("Phase 2b", "Q2"),
)
st.caption("Each keyword is compared only with itself: Google rescales every request to 0–100, so a keyword's "
           "level says nothing about its volume next to another keyword; only its change against its own baseline counts.")
st.caption("Generic gift keywords rise every December, so each category's Google momentum is also benchmarked "
           "against the other categories (\"vs other categories\"): a category is flagged only when outside interest "
           f"grows at least {config.TRENDING['gap_min']:.2f}x faster than the general holiday tide.")

if not snapshot.has_trends or meta.get("status") == "unavailable":
    _no_data(meta)
else:
    labels = {f"{config.PERIODS[b].label} → {config.PERIODS[fo].label}": (b, fo) for b, fo in COMPARISONS}
    choice = st.segmented_control("Compare", list(labels), default=list(labels)[0], key=f"{P}cmp")
    base, focus = labels.get(choice or list(labels)[0])
    alerts = data.trending_alerts(snapshot, f, base=base, focus=focus)
    index_df = _index(snapshot.version, f, base, snapshot)
    scored, _th = data.products(snapshot, data.season_filters(f))

    if alerts is None or alerts.empty:
        st.info("No category had enough baseline Google interest and on-site views to compare in this scope.")
        alerts = pd.DataFrame(columns=phase2.ALERT_COLUMNS)
    flagged = alerts[alerts["flagged"].astype(bool)]
    ui.kpi_row([
        {"label": "Groups compared", "value": f"{len(alerts):,}"},
        {"label": "Trending outside", "value": f"{len(flagged):,}",
         "help": f"Google momentum at least {config.TRENDING['ext_min']:.2f}x and at least "
                 f"{config.TRENDING['gap_min']:.2f}x faster than both on-site views and the other categories' "
                 "Google momentum."},
        {"label": "Keywords", "value": f"{index_df['keyword'].nunique() if len(index_df) else 0:,}"},
    ])

    st.subheader("Alerts")
    if flagged.empty:
        st.success("No category is trending on Google faster than on our site for this comparison. "
                   "Nothing needs promoting ahead of demand.")
    for i, (_, a) in enumerate(flagged.iterrows()):
        with st.container(border=True):
            st.markdown(
                f"**{ui.esc(a['group'])}** · Google interest {a['M_ext']:.2f}x its baseline vs {a['M_site']:.2f}x for "
                f"on-site views (gap {a['gap']:.2f}x) · {_rel(a['m_ext_rel'])} vs other categories · "
                f"keywords: {ui.esc(a['keywords'])}"
            )
            st.caption(ui.esc(a["reason"]))
            gems = _gem_table(a, scored)
            if gems.empty:
                st.caption("No hidden gems or winners in this group to promote ahead.")
            else:
                st.markdown("Promote ahead (best converters in this group):")
                ui.data_table(gems, key=f"{P}gems_{i}", column_config={
                    "Conversion": st.column_config.NumberColumn(format="%.2f%%"),
                    "Adjusted": st.column_config.NumberColumn(format="%.2f%%"),
                })

    st.subheader("Google interest vs on-site views")
    fig = charts.trends_small_multiples(index_df, [base, focus])
    show = index_df[["group_type", "group", "keyword", "week", "n_days", "interest", "site_views",
                     "google_index", "site_views_index"]] if len(index_df) else index_df
    ui.chart_with_table(fig, show, key=f"{P}multiples", file_name="trends_index.csv",
                        caption="Both lines are weekly means indexed so the baseline window = 100, on one shared axis. "
                                "A Google line climbing above the on-site line means outside demand is growing faster.")

    st.subheader("Momentum table")
    lags = _best_lag(phase2.lead_lag(index_df))
    table = alerts[["group_type", "group", "keywords", "M_ext", "m_ext_rel", "M_site", "gap", "reliable", "flagged"]].merge(
        lags, on=["group_type", "group"], how="left")
    ui.data_table(table, key=f"{P}table", column_config={
        "group_type": "Type", "group": "Group", "keywords": "Keywords",
        "M_ext": st.column_config.NumberColumn("Google momentum", format="%.2fx"),
        "m_ext_rel": st.column_config.NumberColumn(
            "Google momentum vs other categories", format="%.2fx",
            help="Google momentum ÷ the median Google momentum of the other categories: above 1 beats the holiday tide."),
        "M_site": st.column_config.NumberColumn("On-site momentum", format="%.2fx"),
        "gap": st.column_config.NumberColumn("Gap", format="%.2fx", help="Google momentum ÷ on-site momentum."),
        "reliable": st.column_config.CheckboxColumn("Readable baseline"),
        "flagged": st.column_config.CheckboxColumn("Flagged"),
        "lead_weeks": st.column_config.NumberColumn("Google leads by (weeks)", format="%d"),
        "lead_corr": st.column_config.NumberColumn("Lead correlation", format="%.2f"),
    })
    st.caption("Lead/lag is descriptive only: it correlates weekly changes over about 13 weeks, too few to test.")

    with st.expander("Provenance"):
        methods = meta.get("method_used", {}) or {}
        km = snapshot.keyword_map
        prov = pd.DataFrame({"keyword": list(methods), "method": list(methods.values())}) if methods else pd.DataFrame()
        if km is not None and len(prov):
            prov = prov.merge(km[["keyword", "group_type", "group_value", "notes"]], on="keyword", how="left")
        st.markdown(
            f"Source: Google Trends, geo **{ui.esc(meta.get('geo', '–'))}**, timeframe "
            f"**{ui.esc(meta.get('timeframe', '–'))}**, fetched **{ui.esc(meta.get('fetched_at', '–'))}**, "
            f"status **{ui.esc(meta.get('status', '–'))}**."
        )
        if meta.get("warnings"):
            st.warning("Keywords with many zero days: " + ui.esc(", ".join(meta["warnings"])))
        if meta.get("errors"):
            st.warning("Keywords that failed: " + ui.esc(", ".join(meta["errors"])))
        if len(prov):
            ui.data_table(prov, key=f"{P}prov")

ui.caveat_note()

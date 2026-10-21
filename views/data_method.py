"""Data & Method: where the numbers come from, how they were cleaned, and what they can't tell us."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ga4app import charts, config, data, ui

P = "dm_"
LIVE_CAP_BYTES = 1024**3  # 1 GiB cap for the live proof re-run
PROOF_KEYS = (("event_count", "Events"), ("user_count", "Users"), ("day_count", "Days"))
GRAIN = {
    "item_daily": "one row per product × day × device: views, purchases, revenue",
    "product_dim": "one row per product: display name, category, brand, price",
    "promo_daily": "one row per promotion banner × day × device",
    "site_daily": "one row per day × device × event name (site-wide counts)",
    "site_funnel": "one row per day × device: funnel step counts",
    "trends_daily": "one row per keyword × day: Google interest (0–100, own scale)",
}


def _n(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "–"


def _credential_source() -> str | None:
    """Where local BigQuery credentials would come from, or None when there are none."""
    adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc and Path(adc).is_file():
        return "GOOGLE_APPLICATION_CREDENTIALS"
    try:
        from ga4app import bq

        root = os.getenv("APPDATA") if os.name == "nt" else None
        root = root or str(Path.home() / ".config")
        if (Path(root) / bq.CRED_CACHE_DIR / bq.CRED_CACHE_FILE).is_file():
            return "cached Google sign-in"
    except Exception:
        return None
    return None


def _run_live_proof(project: str) -> dict:
    from ga4app import bq

    sql = bq.read_sql("proof_counts")
    client = bq.get_client(project, bq.get_credentials(use_local_webserver=False))
    dr = bq.dry_run(client, sql, "proof_counts")
    if dr.bytes_processed > LIVE_CAP_BYTES:
        raise RuntimeError(f"dry run estimates {dr.bytes_processed:,} bytes, above the 1 GiB cap")
    table, meta = bq.run_to_arrow(client, sql, cap_bytes=bq.effective_cap(dr, LIVE_CAP_BYTES),
                                  labels={"step": "proof_live"})
    row = table.to_pylist()[0] if table.num_rows else {}
    return {**{k: row.get(k) for k, _ in PROOF_KEYS}, **meta}


snapshot = data.get_snapshot()
overview, proof, profile, manifest = snapshot.overview, snapshot.bq_proof, snapshot.profile, snapshot.manifest

ui.header(
    "Data & Method",
    "Where every number comes from: the Google Merchandise Store GA4 sample (Nov 1, 2020 – Jan 31, 2021), "
    "how it was cleaned, how the scores are computed, and what the data cannot tell us.",
    ("Q1", "Q2", "Q3", "Q4"),
)

# --------------------------------------------------------------------------- source facts
st.subheader("Source: the data loads, and both routes agree")
st.markdown(f"BigQuery table `{config.BQ_TABLE}` (shards {config.BQ_SUFFIX_START}–{config.BQ_SUFFIX_END}), "
            "and the team's CSV export of the same events.")
facts = pd.DataFrame([
    {"Measure": label, "CSV export (overview.json)": overview.get(k), "BigQuery proof (bq_proof.json)": proof.get(k),
     "Pitch": (proof.get("expected") or {}).get(k)}
    for k, label in PROOF_KEYS
])
facts["Match"] = [a is not None and a == b for a, b in zip(facts.iloc[:, 1], facts.iloc[:, 2])]
ui.kpi_row([{"label": label, "value": _n(proof.get(k, overview.get(k)))} for k, label in PROOF_KEYS]
           + [{"label": "Sessions (CSV)", "value": _n(overview.get("session_count"))}])
if not proof:
    st.info("No BigQuery proof file (bq_proof.json) in this data folder, so only the CSV counts are shown.")
elif bool(facts["Match"].all()):
    st.success("The CSV export and the live BigQuery proof query return the same events, users and days.")
else:
    st.warning("The CSV export and the BigQuery proof differ; see the table.")
ui.data_table(facts, key=f"{P}facts", column_config={"Match": st.column_config.CheckboxColumn("Match")})

with st.expander("Proof query and job metadata"):
    st.code(proof.get("sql") or "(no SQL recorded)", language="sql")
    job = {k: proof.get(k) for k in ("project", "run_at", "job_id", "location", "bytes_processed", "bytes_billed",
                                      "cap_bytes", "cache_hit", "rows", "sql_sha256") if k in proof}
    if job:
        ui.data_table(pd.DataFrame({"Field": list(job), "Value": [str(v) for v in job.values()]}), key=f"{P}job")

cred = _credential_source()
project = proof.get("project") or os.environ.get(config.GCP_PROJECT_ENV)
can_run = bool(cred and project)
if st.button("Re-run proof query live", key=f"{P}live", disabled=not can_run,
             help=None if can_run else "Needs local BigQuery credentials (python scripts/bq_auth.py) and a project id."):
    try:
        with st.spinner("Running the proof query on BigQuery (1 GiB cap)…"):
            st.session_state[f"{P}live_result"] = _run_live_proof(str(project))
    except Exception as exc:  # any auth, network or quota failure is shown, never raised
        st.session_state[f"{P}live_result"] = {"error": f"{type(exc).__name__}: {exc}"}
if not can_run:
    st.caption("Live re-run is off here: no local BigQuery credentials were found. Run `python scripts/bq_auth.py` "
               "on a machine with Google sign-in to enable it. The saved proof above came from a real run.")
live = st.session_state.get(f"{P}live_result")
if live:
    if "error" in live:
        st.error("Live proof failed: " + ui.esc(live["error"]))
    else:
        ok = all(live.get(k) == proof.get(k) for k, _ in PROOF_KEYS)
        (st.success if ok else st.warning)(
            f"Live result: {_n(live.get('event_count'))} events, {_n(live.get('user_count'))} users, "
            f"{_n(live.get('day_count'))} days · {_n(live.get('bytes_billed'))} bytes billed · "
            + ("matches the saved proof." if ok else "differs from the saved proof."))

# --------------------------------------------------------------------------- grain
st.subheader("Snapshots: grain and row counts")
files = manifest.get("files", {}) or {}
grain = pd.DataFrame([
    {"Snapshot": name, "Grain": GRAIN.get(name, ""), "Rows": int(meta.get("rows", 0)),
     "Built from": ", ".join(map(str, meta.get("built_from", [])))[:120], "sha256": str(meta.get("sha256", ""))[:12]}
    for name, meta in files.items()
])
if grain.empty:
    st.info("The manifest lists no snapshot files.")
else:
    ui.data_table(grain, key=f"{P}grain", column_config={"Rows": st.column_config.NumberColumn(format="%d")})
st.caption(f"Manifest source: {ui.esc(manifest.get('source', '–'))} · generated {ui.esc(manifest.get('generated_at', '–'))} · "
           f"product key: {ui.esc(manifest.get('product_key', '–'))} · view metric: {ui.esc(manifest.get('view_metric', '–'))}")

# --------------------------------------------------------------------------- cleaning
st.subheader("Cleaning steps")
fields = profile.get("item_fields", []) or []
if fields:
    ph = pd.DataFrame([{
        "Event": r.get("event_name"), "Item rows": int(r.get("item_rows", 0)),
        "Missing id %": 100 * r.get("bad_item_id", 0) / max(r.get("item_rows", 1), 1),
        "Missing name %": 100 * r.get("bad_item_name", 0) / max(r.get("item_rows", 1), 1),
        "Missing category %": 100 * r.get("bad_item_category", 0) / max(r.get("item_rows", 1), 1),
    } for r in fields])
    st.markdown("**1. Placeholder values** (" + ui.esc(", ".join(f"'{p}'" for p in manifest.get("placeholders", []))) +
                ") are treated as missing. Share of item rows affected:")
    ui.data_table(ph, key=f"{P}placeholders", column_config={
        c: st.column_config.NumberColumn(format="%.2f%%") for c in ("Missing id %", "Missing name %", "Missing category %")})

stats = (profile.get("decisions", {}) or {}).get("stats", {}) or {}
keys = profile.get("keys", {}) or {}
purch = profile.get("purchases", {}) or {}
comp = profile.get("view_item_composition", {}) or {}
aliases = manifest.get("name_aliases", {}) or {}
id_cov, name_cov = stats.get("purchase_id_coverage"), stats.get("purchase_name_coverage")
st.markdown(
    f"**2. Product key = normalised product name.** Item ids are re-issued across event types "
    f"({_n(keys.get('distinct_ids'))} ids for {_n(keys.get('distinct_names'))} names): purchase rows match a viewed "
    f"product by id only {ui.pct(id_cov, 1) if id_cov is not None else '1.1%'} of the time, by name "
    f"{ui.pct(name_cov, 1) if name_cov is not None else '93.2%'}. So the pitch's \"join view_item to purchase on "
    "item_id\" became a join on the normalised name (HTML entities decoded, spaces collapsed, lower case). "
    f"{len(aliases)} legacy purchase-catalogue names are folded into their viewed product "
    "(for example, purchases recorded as \"Womens Google Striped LS\" belong to \"Google Women's Striped L/S\")."
)
st.markdown(
    f"**3. A \"view\" = the product appeared in a product-page view.** The obfuscated sample lists about "
    f"{stats.get('avg_items_per_view_item', 11.4):.1f} products per view_item event "
    f"({_n(comp.get('multi_item_events'))} of {_n(comp.get('view_events'))} events carry several), and list position "
    "carries no main-product signal: 88% of purchased products appeared in a view_item list earlier in the session, "
    "but only 29% of those at position 0. Views are therefore appearances, and conversion is purchases per appearance."
)
st.markdown(
    f"**4. Purchases are de-duplicated by transaction id:** {_n(purch.get('purchase_events'))} purchase events, "
    f"{_n(purch.get('distinct_txn'))} distinct transactions, {_n(purch.get('duplicate_events'))} duplicate events "
    f"removed, {_n(purch.get('missing_txn'))} events without an id kept as their own transaction. Revenue "
    f"\\${purch.get('purchase_revenue_usd_raw', 0):,.0f} raw → \\${purch.get('purchase_revenue_usd_dedup', 0):,.0f} de-duplicated."
)
promo = snapshot.promo_daily
banners = ""
if promo is not None and not promo.empty and "promotion_name" in promo.columns:
    top = promo.groupby("promotion_name")["promo_views"].sum().sort_values(ascending=False).head(4).index
    banners = " (" + ", ".join(f"\"{ui.esc(n)}\"" for n in top) + ")"
st.markdown(
    f"**5. Promotions carry banner names, not products**{banners}, so they cannot say which product was featured."
)
st.markdown(
    "**6. Add-to-cart is not a cart count.** add_to_cart events also list about 11 products each, and cart tracking "
    f"only becomes reliable from {config.CART_TRACKING_START:%b %d}. Carts are left out of the scores."
    if hasattr(config, "CART_TRACKING_START") else
    "**6. Add-to-cart is not a cart count.** add_to_cart events also list about 11 products each, and cart tracking "
    "only becomes reliable from late November. Carts are left out of the scores."
)

gaps = overview.get("tracking_gaps", {}) or {}
site = snapshot.site_daily
if site is not None and len(site) and {"event_name", "events"} <= set(site.columns):
    shown = [e for e in ("view_item", "add_to_cart", "select_item", "purchase") if e in set(site["event_name"].astype(str))]
    daily = (site[site["event_name"].isin(shown)].groupby(["event_date", "event_name"], as_index=False)["events"].sum())
    fig = go.Figure()
    palette = list(config.SCATTER_CATEGORICAL) + [config.PALETTE["navy"]]
    for name, colour in zip(shown, palette):
        d = daily[daily["event_name"] == name]
        fig.add_trace(go.Scatter(x=d["event_date"], y=d["events"], name=name, mode="lines", line={"width": 1.6, "color": colour}))
    charts.apply_template(fig, height=340, yaxis_title="Events per day (site)", yaxis_type="log",
                          legend={"orientation": "h", "y": 1.12})
    ui.chart_with_table(fig, daily, key=f"{P}gaps", file_name="tracking_gaps.csv",
                        caption="Tracking gaps: add_to_cart and select_item are almost untracked in early November "
                                "(log scale). This is why carts never enter the scores.")
if gaps:
    ui.data_table(pd.DataFrame([{"Event": k, "Median per day": v.get("median_daily"), "Gap days": v.get("gap_days"),
                                 "First reliable date": v.get("first_reliable_date")} for k, v in gaps.items()]),
                  key=f"{P}gap_table")

# --------------------------------------------------------------------------- categories
st.subheader("Category mapping")
dim = snapshot.product_dim
cats = (dim.groupby(["category_raw", "category_group"], as_index=False, dropna=False)
        .agg(products=("product_key", "size"), views=("total_views", "sum"))
        .sort_values("views", ascending=False))
st.caption("Each product's most common view-row category, mapped to one of seven groups. Collections (Sale, New, "
           "Shop by Brand, Campus Collection…) are not product types, so those products are grouped by name keywords.")
ui.data_table(cats.rename(columns={"category_raw": "Raw category", "category_group": "Group", "products": "Products",
                                   "views": "Views"}), key=f"{P}cats", height=320)

# --------------------------------------------------------------------------- metrics
st.subheader("Metric definitions")
m = data.shrinkage_m(snapshot)
m_raw = data.shrinkage_m_raw(snapshot)
lo_m, hi_m = config.SHRINK_M_BOUNDS
if m_raw != m_raw:  # NaN: too few products to estimate
    m_how = f"set to the default {m:,.0f} because there were too few products to estimate it"
elif m_raw < lo_m:
    m_how = f"estimated at about {m_raw:,.0f} by the method of moments, floored at {lo_m:,.0f} for stability"
elif m_raw > hi_m:
    m_how = (f"capped at {hi_m:,.0f} (the method-of-moments estimate found little real spread between products)"
             if m_raw == float("inf") else
             f"estimated at about {m_raw:,.0f} by the method of moments, capped at {hi_m:,.0f}")
else:
    m_how = "estimated from the data by the method of moments"
st.markdown("**Conversion** of product *i*: purchases per product view, and the site average over all products in scope.")
st.latex(r"cr_i = \frac{P_i}{V_i}, \qquad \bar p = \frac{\sum_i P_i}{\sum_i V_i}")
st.markdown(f"**Adjusted conversion (empirical-Bayes shrinkage).** Each product is pulled toward its category's average "
            f"μ_c by m = **{m:,.0f}** pseudo-views ({m_how}), so a product with "
            "a few lucky sales does not look extreme. The 80% likely range is the Beta posterior's 10th–90th percentile.")
st.latex(r"cr^{adj}_i = \frac{P_i + m\,\mu_c}{V_i + m}, \qquad \mathrm{Beta}\big(P_i + m\mu_c,\; V_i - P_i + m(1-\mu_c)\big)")
st.markdown("**Forgone purchases** (Wasted exposure) and **extra purchases per 1,000 views** (Hidden gems):")
st.latex(r"\text{forgone}_i = V_i\,\max(\bar p - cr^{adj}_i,\,0), \qquad \text{lift}_{1k} = 1000\,\max(cr^{adj}_i - \bar p,\,0)")

st.subheader("Season windows")
win = pd.DataFrame([{"Window": p.label, "Start": p.start, "End": p.end, "Days": (p.end - p.start).days + 1}
                    for p in config.PERIODS.values()])
ui.data_table(win, key=f"{P}windows")
st.markdown(
    "- **Pre-holiday baseline (Nov 1–22):** the only normal weeks before the surge; the data starts Nov 1, so a full "
    "4-week baseline is not possible. A January 4-week baseline is offered as a second reference.\n"
    "- **Cyber Week (Nov 23–30)**, with **Black Friday weekend** and **Cyber Monday** inside it: the first two campaigns.\n"
    "- **December gifting push (Dec 2–18):** ends at the usual standard-shipping cutoff for Christmas delivery."
)

st.subheader("What this data can't tell us")
st.markdown(
    "- **No placement history:** we never see which products were on the homepage, so \"featured\" is proxied by "
    "spikes in a product's share of views.\n"
    "- **No inventory:** a low-converting product may simply have been out of stock in some sizes.\n"
    "- **No margin:** revenue is shown, but profit per sale is unknown, so efficiency is in purchases, not dollars.\n"
    "- **Obfuscation:** the public GA4 sample is obfuscated (re-issued ids, product lists in view events, placeholder "
    "values), so absolute numbers describe the sample, not the real store.\n"
    "- **Tracking gaps:** cart and list-click events are missing for parts of November.\n"
    "- **Google Trends normalisation:** every request is rescaled to 0–100, so each keyword is only compared with itself."
)

st.subheader("How we go further")
st.markdown(
    "| Existing analysis | What it does | What this app adds |\n|---|---|---|\n"
    "| [rupalchopra16 (GitHub)](https://github.com/rupalchopra16/Google-Analytics-Web-Analysis-using-Python) | "
    "Session-level funnel, cohorts, device segmentation | Item/SKU-level ranking for a placement decision |\n"
    "| [guiarpi (GitHub)](https://github.com/guiarpi/GA4-funnel-analysis) | Static funnel charts | An interactive "
    "tool tied to one decision: the holiday homepage slots |\n"
    "| [inna.koliandra (Tableau)](https://public.tableau.com/app/profile/inna.koliandra/viz/EcommerceFunnelAnalysisGA4/Dashboard1) | "
    "Session-level funnel with filters | Shrunk per-product conversion, cannibalization tests and an outside Google Trends signal |"
)

st.subheader("SQL")
for path in sorted(config.SQL_DIR.glob("*.sql")):
    with st.expander(path.name):
        try:
            st.code(path.read_text(encoding="utf-8"), language="sql")
        except OSError as exc:
            st.caption(ui.esc(f"Could not read {path.name}: {exc}"))

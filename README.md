# Which Products Earn the Homepage?

**A Streamlit decision tool for the Google Merchandise Store** that ranks products by **view-to-purchase efficiency** (not raw traffic). It decides which SKUs should get the site's limited holiday homepage slots: Black Friday, Cyber Monday and the December gifting push.

*ITCS 5122 Visual Analytics · Team 9 · Account D · Pitch 1. Client: Head of Merchandising.*

> **Status: work in progress.**
> - **Built and verified:** the data pipeline (live BigQuery pull, profiling, exports, site-level aggregates).
> - **Being built now:** the Streamlit app, its analysis modules and the pages.
>
> See [What is still missing](#what-is-still-missing) for the exact gaps.

---

## The questions the app answers

The app answers four questions from the pitch, plus the two Phase 2 ideas:

| # | Question | Planned view |
|---|---|---|
| Q1 | Which products get lots of views but rarely get bought? | Quadrant scatter, "Wasted exposure" quadrant + retire list |
| Q2 | Which products convert well but stay hidden? | Quadrant scatter, "Hidden gems" quadrant + promote list |
| Q3 | Does the pattern shift during Black Friday / Cyber Monday? | Grouped bars: conversion by category, holiday window vs baseline; quadrant movers |
| Q4 | Does it differ by device (mobile vs desktop)? | Heatmap of top candidates × device, with a mobile-first / desktop-first call |
| 2a | Cannibalization check | When product A's exposure spikes, does a same-category product B's conversion dip? |
| 2b | "Trending outside our walls" alert | Google Trends interest by category vs on-site exposure → promote ahead of demand |

An **AI layer** (an LLM via [OpenRouter](https://openrouter.ai), default `anthropic/claude-opus-5.5`) narrates the quadrant chart into a meeting-ready verdict, for example *"Feature X, Y, Z. Retire A from the homepage — it drives views but almost no purchases."* A deterministic rule-based verdict is always available as the fallback.

## Data

| Source | What it gives us |
|---|---|
| `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` (BigQuery public dataset) | Event-level GA4 data with the nested `items` array, Nov 1 2020 – Jan 31 2021 (92 days) |
| `ga4_events_export.csv` (team export, 450 MB, **not in this repo**) | The same 4,295,584 events, but only 7 event-level columns and **no item fields**. Used for site-level context. |
| Google Trends | External search interest per merchandise category (Phase 2b) |

**Proof it loads.** The pitch's query was re-run live (`sql/proof_counts.sql`, 0.12 GiB billed) and reproduces the slide exactly:

| event_count | user_count | day_count |
|---|---|---|
| 4,295,584 | 270,154 | 92 |

The CSV reproduces the same numbers, plus 360,129 sessions. See `data/bq_proof.json` and `data/overview.json`.

### What profiling the real data changed

Profiling (`scripts/profile_bigquery.py`; results in `data/profile.json`) overturned three assumptions in the pitch:

1. **Products cannot be matched on `item_id`.** The obfuscated sample re-issues item ids across event types: 1,374 ids for only 429 product names.
   - Purchased items match a viewed item **by id only 1.1%** of the time, but **by name 93.2%**.
   - The pitch's "join view_item ↔ purchase on item_id" therefore becomes a join on a **normalised product name**: HTML-unescaped, whitespace-collapsed, lowercased.
2. **A `view_item` event lists ~12 different products, not just the one viewed.**
   - 219,633 of the 241,876 item-bearing `view_item` events carry exactly 12 items.
   - List position carries no signal. Of 13,648 products bought in a session, 88% had appeared in a `view_item` list earlier in that session, but only 29% of those ever appeared in the first position.
   - So a **"product view" = the product appeared in a product-page view event**: about 2.66M product views across 420 products.
   - This is the same scale as the pitch mock-up: about 10³–10⁵ views per product and about 0.55% pooled conversion.
3. **Purchases need de-duplication.**
   - 5,692 purchase events, but only 4,451 distinct transactions: 335 duplicate events, 906 without an id.
   - Purchases are de-duplicated by transaction id. Revenue is $362k raw and $339k after de-duplication.

Other facts the app relies on:

- **Add-to-cart was not tracked consistently before Nov 25/26, 2020** (≈0 events on Nov 1–15 and Nov 21–24). Only purchase ÷ view is compared across the season.
- **Season windows come from the data.** Purchases surge on Mon Nov 23 (before Thanksgiving), peak Dec 9–16 and collapse after the Dec 18 shipping cutoff.
  - Baseline: Nov 1–22
  - Cyber Week: Nov 23–30
  - Black Friday weekend: Nov 26–29
  - Cyber Monday: Nov 30–Dec 1
  - December gifting: Dec 2–18
  - January 4-week alternative baseline: Jan 4–31
- **Site-wide purchase ÷ view_item:** 1.18% (baseline), 1.85% (Cyber Week), 1.70% (December), 1.47% (full season). By device: desktop 1.43%, mobile 1.54%, tablet 1.33%.
- **Promotions can't be tied to products.** Promotion events carry banner names ("Reach New Heights", "Act Responsible", …) but no products. The cannibalization check therefore uses spikes in a product's share of category views as its "featured" proxy.
- **Category labels use two vocabularies.** View rows use path-like values (`Home/Apparel/Men's / Unisex/`, `Lifestyle/Drinkware/`, `Home/Sale/`, …). Purchase rows use flat ones (`Apparel`, `Campus Collection`, …). Collections such as Sale, New and Shop by Brand are mapped to product types by name keywords.

### What this data can't tell us

- **No homepage-placement history.** The app recommends candidates; it cannot prove that featuring causes sales.
- **No inventory data.** A high-converting item may have been out of stock.
- **No margin or cost data.** The app optimises conversion, not profit.
- **The sample is obfuscated by Google.** Read relative patterns, not exact business totals.

## Data pipeline (implemented)

Every BigQuery query is dry-run first and runs under a `maximum_bytes_billed` cap. Jobs are logged to `data/raw/export_log.json`.

| Step | Script | Output | BigQuery cost |
|---|---|---|---|
| 1. Sign in + proof query | `scripts/bq_auth.py --project <id>` | `data/bq_proof.json` | 0.12 GiB |
| 2. Profile the `items` array, record decisions | `scripts/profile_bigquery.py --with-event-params` | `data/profile.json`, `data/raw/profile_raw.json` | ~5.4 GiB |
| 3. Export item × day × device funnel + attributes + promotions | `scripts/export_bigquery.py` | `data/raw/bq_item_daily.parquet` (96,653 rows), `bq_item_attrs.parquet` (8,321), `bq_promo_daily.parquet` (2,330) | ~1.4 GiB |
| 4. Site-level aggregates from the CSV (DuckDB, ~5 s) | `scripts/build_site_aggregates.py` | `data/site_daily.parquet`, `data/site_funnel.parquet`, `data/overview.json` | — |
| 5. Google Trends | `scripts/fetch_trends.py` | `data/trends_daily.parquet` | *in progress* |
| 6. App snapshots | `scripts/build_snapshots.py` + `scripts/verify_snapshots.py` | `data/item_daily.parquet`, `data/product_dim.parquet`, `data/manifest.json` | *in progress* |

All of this fits comfortably in the free 1 TiB/month BigQuery tier: under 10 GiB in total, including exploratory queries. Any Google Cloud project with the BigQuery API enabled works, including a free sandbox project.

```bash
conda activate ITCS5122            # or any Python ≥ 3.11 env
pip install -r requirements.txt    # requirements files are being added; see below
python scripts/bq_auth.py --project <your-gcp-project-id>   # opens a browser once
python scripts/profile_bigquery.py --with-event-params
python scripts/export_bigquery.py
python scripts/build_site_aggregates.py --csv ga4_events_export.csv
```

The resulting `data/` snapshots are committed, so the app will run without BigQuery or the 450 MB CSV.

## Metrics (design)

These are defined in `docs/DESIGN.md` §5 and implemented in `ga4app/metrics.py`, which is in progress.

- **Conversion:** purchases ÷ product views (the pitch's y-axis).
- **Efficiency score:** an empirical-Bayes shrunk conversion rate, `(P + m·μ_category) / (V + m)`.
  - Low-traffic products are not ranked on noise.
  - `m` is estimated once by method of moments.
  - An 80% posterior interval gives each product a confidence call: above average, below average, or uncertain.
- **Quadrants:**
  - The x split is a high-traffic cutoff (median views of eligible products; p75/p90 selectable).
  - The y split is pooled conversion.
- **Retire priority:** forgone purchases, `views × (average − adjusted rate)`.
- **Promote priority:** extra purchases per 1,000 extra views.
- **Holiday shift:** category conversion per window with Wilson intervals and two-proportion tests. Products that change quadrant are listed.
- **Device lean:** each product's conversion indexed to its device's own baseline, with a binomial test.
- **Cannibalization:**
  - "Featured" days are spikes in a product's share of its category's views.
  - Each peer's purchases are compared with what its same-day category peers predict, which controls for seasonality.
  - Pairs are flagged with a conditional binomial test plus Benjamini–Hochberg FDR.
- **Trending alert:** Google search momentum vs on-site view growth per category.

## Project structure

```
ga4app/            analysis package
  config.py        paths, season windows, thresholds, validated colour palette, OpenRouter models   ✅
  contracts.py     parquet/json schemas, validation, typed atomic writers                           ✅
  types.py         Filters, Thresholds, product-frame column contract                               ✅
  bq.py            BigQuery auth (browser sign-in, cached), dry-run, byte-capped runs              ✅
  metrics.py, phase2.py, charts.py, llm.py, narrative.py, taxonomy.py, trends.py,
  data.py, state.py, ui.py, synthetic.py                                                           🚧 in progress
sql/               12 BigQuery queries (proof, 8 profiling, 3 export); paste-able into the console  ✅
scripts/           bq_auth, profile_bigquery, export_bigquery, build_site_aggregates                 ✅
                   build_snapshots, verify_snapshots, fetch_trends, make_synthetic_data             🚧
data/              committed snapshots (small) + raw BigQuery exports in data/raw/                  ✅ partial
docs/DESIGN.md     full design: data contracts, SQL, formulas, page specs, test plan               ✅
app.py, views/     Streamlit entry point + 7 pages                                                  🚧
tests/             pytest unit tests + Streamlit AppTest smoke tests                                🚧
```

## Planned app pages

1. **Feature Picks (Q1, Q2 + AI verdict):**
   - the quadrant bubble scatter (log views × conversion; Hidden gems / Winners / Low priority / Wasted exposure);
   - click-to-inspect;
   - ranked promote / retire / keep tables;
   - a streamed AI verdict.
2. **Holiday Shift (Q3):** the category × window grouped bars with confidence intervals; daily conversion with shaded windows; quadrant movers.
3. **Device Split (Q4):** the top candidates × device heatmap with a mobile-first / desktop-first call.
4. **Outside Signal (2b):** Google interest vs on-site views indexed to baseline; "trending outside our walls" alerts.
5. **Cannibalization Check (2a):** exposure-spike days, pair tests with intervals, flagged pairs.
6. **Slot Plan (the decision):** Black Friday / Cyber Monday / December slot cards with device lean and trend/cannibal flags; retire list; CSV/Markdown export.
7. **Data & Method:** proof counts (CSV vs live BigQuery), cleaning steps with real counts, category mapping, formulas, limits, SQL.

**Shared sidebar filters:** season presets, date range, device, category, and a min-views threshold. The charts update live.

**Charts:** Plotly. The colours are validated for colour-blind separation, so the scatter colours only the 3 largest category groups and shows the rest as "Other", with a highlight-one-category mode.

## What is still missing

| Area | Status |
|---|---|
| Streamlit app (`app.py`, 7 pages, sidebar, AI verdict UI) | 🚧 Being built now |
| Analysis modules: metrics, Phase 2 (cannibalization + trending), charts, OpenRouter narrator, taxonomy | 🚧 Being built now |
| App snapshots (`data/item_daily.parquet`, `product_dim.parquet`, `manifest.json`) | 🚧 Needs `build_snapshots.py` (in progress); raw exports are already here |
| Google Trends data | ❌ Not fetched yet. `pytrends` was archived in 2025 and is usually rate-limited (HTTP 429); the plan is `trendspy` first, with manually downloaded Trends CSVs as a fallback. Phase 2b depends on it. |
| `requirements*.txt`, `pyproject.toml`, `.streamlit/config.toml`, `secrets.toml.example` | 🚧 Being added |
| Tests (unit + Streamlit AppTest) | 🚧 Being written alongside each module |
| Live AI verdict check | ⏳ Needs an `OPENROUTER_API_KEY` in `.streamlit/secrets.toml` (never commit it) |
| Visual QA (screenshots of every page) and an adversarial review pass | ⏳ After the build |
| Deployment (e.g. Streamlit Community Cloud) | ⏳ Not started; the app is designed to run from the committed snapshots alone |
| `TEAM9.pdf` (original pitch deck) | Intentionally not committed |

## Known limitations

- Conclusions are correlational. There is no placement history, inventory or margin data (see above).
- "Views" are list appearances in product-page events, because the obfuscated sample doesn't record which product was actually viewed.
- The pre-holiday baseline has only ~3 weeks of data (Nov 1–22); a January 4-week window is offered as the alternative.
- The tablet segment is small (~2% of views), so device calls exclude it.
- Google Trends values are normalised 0–100 per request. Each keyword is compared only with itself (momentum vs its own baseline).

## Existing analyses of this dataset, and how this goes further

| Existing analysis | What it does | How this project goes further |
|---|---|---|
| [rupalchopra16/Google-Analytics-Web-Analysis-using-Python](https://github.com/rupalchopra16/Google-Analytics-Web-Analysis-using-Python) | Session-level funnel, cohorts and device segmentation | Works at item/SKU level for a placement decision |
| [guiarpi/GA4-funnel-analysis](https://github.com/guiarpi/GA4-funnel-analysis) | Static matplotlib funnel charts | An interactive tool tied to one specific decision |
| [Inna Koliandra's Tableau funnel dashboard](https://public.tableau.com/app/profile/inna.koliandra/viz/EcommerceFunnelAnalysisGA4/Dashboard1) | Session-level funnel with filters | Adds item-level ranking and an external Google Trends signal |

---

*The data is Google's public obfuscated GA4 sample. This is a student project, not affiliated with Google.*

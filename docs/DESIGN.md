# DESIGN — "Which Products Earn the Homepage?" (Team 9 · Account D · Pitch 1)

This is the detailed design behind the approved plan. **The adjustments in this first section override anything below.**

## Approved adjustments (override the sections below)

1. **January 4-week baseline.** `config.PERIODS` gains `january_4wk` (2021-01-04 to 2021-01-31). Holiday Shift comparisons are:
   - Baseline vs Cyber Week (default)
   - Baseline vs December
   - Cyber Week vs December
   - January 4-wk vs Cyber Week
   - January 4-wk vs December

   The pitch asks for a 4-week baseline, but only about 3 pre-holiday weeks exist (data starts Nov 1; the surge starts Nov 23).
2. **Shareable URLs are optional.** Use `bind="query-params"` only if it works cleanly with the season presets and the Reset button; otherwise omit it.
3. **Requirements** are pinned to the versions actually installed in the ITCS5122 conda env. pandas stays at 3.0.2.
4. **Plotly rendering.** Every chart is rendered with `st.plotly_chart(fig, theme=None, width="stretch", ...)`, using explicit colours and the custom light template from `ga4app/charts.py`. Never use `use_container_width`.
5. **Data pipeline ownership.** The lead owns these files, and workflow agents must not modify them:
   - `ga4app/bq.py`
   - `sql/*`
   - `scripts/bq_auth.py`, `scripts/profile_bigquery.py`, `scripts/export_bigquery.py`, `scripts/build_site_aggregates.py`
   - `ga4app/config.py`, `ga4app/contracts.py`

   To change one of these, report the need instead of editing.
6. **OpenRouter model list.** `anthropic/claude-opus-5.5` is the default; the others are `anthropic/claude-sonnet-5.5`, `anthropic/claude-haiku-5.5`, `anthropic/claude-fable-5.1`, `openrouter/free`, `google/gemma-4-31b-it:free`, and a custom ID. Paid models fall back to `anthropic/claude-sonnet-5.5`. Free models have no fallbacks.
7. **Scatter colours.** The 3 largest category groups take `#2a78d6, #eb6834, #1baf7a`; all other groups are "Other" `#8A8F98`. Colours are frozen in the manifest, so colour follows the entity. A "highlight one category" mode is also provided. Never use more than 3 categorical hues on the scatter.
8. **No synthetic data in the final app.** Synthetic snapshots are only for tests and development (tmp dirs, or `data_synthetic/`). The app shows a SYNTHETIC banner whenever `manifest.source == "synthetic"`.
10. **Post-build changes from the review, 2026-10-21.** These override §5 where they differ.
    - **Cannibalization baseline.** B's "usual pace" (O_n/E_n) is measured only on A's non-spike days within ±7 days of one of A's spike days (`phase2.cannibalization_tests(local_days=7)`), not on the whole season. The Fisher check uses the same days.
      - The whole-season baseline flagged seasonal artifacts: Google Sunglasses sells more slowly in January.
      - With the local baseline, no real pair is flagged.
    - **Trend alert benchmark.** `trending_alerts` adds `m_ext_rel` = M_ext ÷ the median M_ext of the other reliable category groups. A group is flagged only if the old rules hold and `m_ext_rel` ≥ 1.15. This discounts the generic December lift in gift searches.
    - **Holiday Shift comparisons.** `config.COMPARISONS` adds baseline vs Black Friday weekend and baseline vs Cyber Monday.
    - **Name aliases (R1).** 8 legacy purchase-catalogue names are folded into their viewed product (`taxonomy.NAME_ALIASES`, recorded in `manifest.name_aliases`).
    - **Google Trends.** pytrends served all 13 keywords; trendspy was rate-limited (HTTP 429).
    - **Shrinkage strength m.** The raw method-of-moments estimate is about 70; it is floored at 100 (`metrics.prior_strength_raw` exposes the raw value).
9. **Shrinkage strength `m` is computed at runtime.** `metrics.estimate_prior_strength` runs over full-season, all-device data and is cached per data version (`data.get_snapshot`). `manifest.shrinkage_m` is optional.
11. **Improvements, 2026-11-17.** These override the sections below where they differ.
    - **Item-data cut.** From Jan 26, 2021 the sample's purchase events carry no items (site purchase events stay at 19–69 a day; item-level purchases fall to 53, 12, 10, 16, 11, 0). `config.ITEM_DATA_END = 2021-01-25`.
      - `data.load_snapshot` drops item rows after it and keeps per-day totals in `Snapshot.item_daily_totals` for Data & Method.
      - `PERIODS["full"]` and `PERIODS["january"]` end on it. `january_4wk` keeps its key but is now "January baseline", Jan 4–25 (22 days, like the pre-holiday baseline).
      - `PERIODS["export"]` (Nov 1 – Jan 31) is the raw extent, used only for site-level checks such as the pitch's 1.47%.
      - `Filters().end` defaults to the cut. The sidebar date picker stops there. Item-level charts that reach the cut show a grey band to Jan 31.
      - `verify_snapshots.py` hard-checks that every day up to the cut has ≥ 1 item purchase row per site purchase event and every later day has fewer.
      - Effect on the real data: average conversion 0.547% → 0.570%; raw `m` ≈ 70 → 66; 14 quadrant changes; SF Campus Tote drops below 200 views; two cannibalization pairs become "Possible" (q = 0.29).
    - **Campaign score evidence.** `metrics.campaign_scores` adds `ci_low` / `ci_high` (80% interval of the same Beta posterior), `window_weight = V_w/(V_w + m)` and `thin_window = window_weight < 0.5`. Slot cards show the range and a "Mostly season evidence" badge. `charts.slot_candidates` plots each campaign's top candidates with their ranges.
    - **Under-counted views.** `metrics.shrink` adds `checkouts` and `views_undercount` (checkouts ≥ `config.UNDERCOUNT_RATIO` = 0.5 × views; both are now in `types.PRODUCT_COLUMNS`). It is a flag only: Feature Picks tables, the inspector, slot cards and the verdict's Watch-outs show it.
    - **Pinned and excluded products.** `metrics.slot_plan(..., must_include=...)` places each pinned product in the campaign where it scores best (respecting room, conflicts and repeats), then fills greedily. Each campaign's slots are numbered by score, a `pinned` column marks pins, and `out.attrs["unplaced"]` lists pins that could not be placed. `state.PINNED` / `state.EXCLUDED` hold the lists.
    - **Shareable URLs** (replaces adjustment 2). `state.load_query_params` seeds the filters from `?preset=`, `start=`/`end=`, `dev=`, `cat=` and `minv=` once per session. `state.sync_query_params` rewrites the URL after each run, leaving defaults out. `bind="query-params"` stays unused.
    - **Charts.** The Feature Picks scatter has a log conversion axis (the default; linear remains), so the top converters are no longer clipped. `charts.device_bars` adds the inspector's per-device bars. The heatmap shows "–" for tablet cells under 30 views and drops the lean column when no product leans. `ui.chart_with_table` passes its caption as the chart's `alt` text.
    - **Narrative.** `PROMPT_VERSION = "verdict-2026-11-17.1"`. Slots carry `window_views`, `season_based` and `views_undercount`; entries carry `views_undercount`. The rule-based plan bullet says when a lead slot rests on few window views, and Watch-outs name under-counted picks.

## Real-data findings (BigQuery profiling, 2026-10-20) — these override the assumptions below

These come from `data/profile.json`, `data/raw/profile_raw.json` and `data/overview.json`.

### Proof query
- BigQuery returned 4,295,584 events, 270,154 users and 92 days, matching the pitch. The CSV export reproduces the same numbers, plus 360,129 sessions.
- The `events_*` wildcard bills as **one** referenced table, so the feared ~0.9 GiB minimum billing floor does not apply. The whole profile plus export used about 6 GiB.

### Product key = normalised item name (R1)
- `item_id` values are re-issued across event types:
  - 1,374 ids but only 429 names;
  - purchase rows match a viewed product by id **1.1%** of the time, by name **93.2%**;
  - names with several ids cover 99.9% of rows.
- So `product_key = name_key(item_name)`:
  1. `html.unescape`
  2. trim
  3. collapse internal whitespace to one space
  4. lowercase

  `Android Iconic 4&quot; Decal` and `Android Iconic 4" Decal` must collapse to one key.
- The pitch's "join view_item ↔ purchase on item_id" therefore becomes "join on the normalised product name". Say so on Data & Method.

### A "view" = the product appeared in a view_item event (R2)
- Event shape:
  - 241,876 of 386,068 view_item events carry items, and those carry about 11.4 *different* products each;
  - 219,633 events carry exactly 12 items;
  - only 314 of 233,150 multi-item events repeat one product.
- **List position carries no "main product" signal.** Of 13,648 products bought in a session, 87.8% had appeared in a view_item list earlier in that session, but only 28.8% of those ever appeared at offset 0. First-seen offsets are spread evenly over 0–11. The page_title test matched 0%.
- So `view_metric = "view_events"`:
  - `views` counts the view_item events whose item list contains the product, about 2.75M product-views over the season.
  - This is the same scale as the pitch mock-up: about 10³–10⁵ views per product and about 0.4–0.6% pooled conversion.
  - On the UI, call them "product views (appearances in product-page views)". On Data & Method, explain that the obfuscated sample lists about 12 products per view_item event.
- add_to_cart events also carry about 11.4 products each, so per-product `add_to_carts` is **not** a cart count. Never present it as one; leave it out of the UI or show it only in Data & Method as a caveat.
- begin_checkout carries about 2.4 items per event.

### Purchases (R5)
- 5,692 purchase events, 4,451 distinct transactions, 906 with no transaction id, and 335 duplicate events.
- 16,003 purchase item rows.
- Revenue: $362,165 raw, $339,357 after de-duplication. Deduped purchases are the conversion numerator.

### Promotions (R4)
- view_promotion and select_promotion items carry promotion names but no product:
  - "Reach New Heights" (127,841 item rows);
  - "Act Responsible";
  - "Complete Your Collection";
  - "Google Mural Collection".
- So `featured_signal = "views_share"`. Data & Method may mention that banner-level promotion data exists but cannot be tied to products.

### Categories (R3): two vocabularies
- **View rows** use path-like values. In order of view rows:
  - `Home/Apparel/Men's / Unisex/`, `Home/Sale/`, `Home/Apparel/`, `Lifestyle/Drinkware/`, `Home/Apparel/Women's/`, `Home/Apparel/Hats/`, `Lifestyle/Bags/`, `Home/New/`;
  - `Home/Shop by Brand/Google/`, `Lifestyle/Small Goods/`, `Home/Shop by Brand/YouTube/`, `Home/Apparel/Kids/`, `Lifestyle/`, `Home/Eco-Friendly/`;
  - `Home/Campus Collection/`, `Home/Stationery/`, `Home/Stationery/Notebooks/`, `Men's T-Shirts/`, `Home/Stationery/Stickers/`, `Sale`, `Home/Stationery/Writing/`;
  - `Home/Shop by Brand/`, `Home/Shop by Brand/Android/`, `Men's / Unisex`, `Home/Apparel/Socks/`, `''`, `Women's`, `Google`, `Drinkware`, `Apparel`;
  - `Small Goods`, `Lifestyle`, `Bags`, `Backpacks/`, `Eco-Friendly`, `Mugs & Tumblers/`, `Campus Collection`, `Water Bottles/`, `Kids`, `Stationery`, …
- **Purchase rows** use flat values: `Apparel`, `Campus Collection`, `New`, `Accessories`, `Office`, `Shop by Brand`, `Drinkware`, `Bags`, `Clearance`, …
- A product's `category_raw` is its most frequent non-placeholder **view-row** category, falling back to a purchase-row category.
- Collections are not product types: Sale, New, Shop by Brand (and its sub-brands), Campus Collection, Eco-Friendly, Clearance, Google, and the empty value. These map to a group by product-name keywords. Examples:
  - tee / hoodie / sweatshirt / jacket / cap / hat / beanie / sock / onesie → Apparel
  - bottle / mug / tumbler / cup → Drinkware
  - bag / tote / backpack / pack / pouch → Bags
  - journal / notebook / pen / sticker / decal / pad → Office & Stationery
- "Other" should stay at or under 15% of views.

### Devices (view_item events / purchase events)
- desktop 225,037 / 3,226
- mobile 152,676 / 2,355
- tablet 8,355 / 111

### CSV tracking gaps
- add_to_cart's first reliable date is 2020-11-25 (`overview.json`). `config.CART_TRACKING_START` stays at Nov 26 as the conservative choice.

---

# Implementation plan: "Which Products Earn the Homepage?" (Team 9 · Account D)

## 0. Research results that change the design

### A. Streamlit 1.66.0 (installed package)

**Navigation and pages**
- `st.navigation(pages, *, position="sidebar"|"top"|"hidden", expanded=False)` returns one of the `st.Page` objects you passed in (same object, so an identity check works).
- The entrypoint script runs on every rerun. Widgets created there keep their state across pages; this is the documented pattern for shared filters.
- `st.Page(page: str|Path|Callable, *, title, icon, url_path, default, visibility)`.

**AppTest**
- `AppTest.switch_page()` only accepts page files (`testing/v1/app_test.py` L683–693 raises `ValueError` if the path isn't a file). Callable pages cannot be switched to in tests, so pages must be thin files.
- `from_file(path, default_timeout=3)`. The 3-second default timeout is too short for this app; set it to 60.
- `.run(timeout=)`, `.secrets` (a dict), `.session_state`, `.query_params` are all available.
- Plotly charts come back as `UnknownElement`. `at.get("plotly_chart")[i].proto.spec` is the figure JSON, so tests can assert on it. Selections cannot be simulated.

**`st.plotly_chart`**
- Signature: `(fig, *, width="stretch", height="content", theme="streamlit", key, on_select="ignore"|"rerun"|callable, selection_mode=("points","box","lasso"), config, alt)`.
- It returns `PlotlyState`; `.selection.points[i]` carries `customdata`, `point_index`, `curve_number`.
- The element ID hashes the whole figure spec even when a key is set (`key_as_main_identity=False`, L790–803). Any change to the figure resets the selection: a filter change, or highlighting the clicked point.
  - Persist the clicked product via the `on_select` callback into `st.session_state`.
  - Never restyle the scatter based on its own selection.
- Selection state is read-only (`writes_allowed=False`).

**Deprecated or removed (code must avoid)**
- `use_container_width` is deprecated on plotly_chart, dataframe, altair, image, pyplot, data_editor, map and graphviz (deprecation date "2025-12-31"). Use `width="stretch"|"content"|int`.
- Removed: `st.cache` and all `st.experimental_*` (`experimental_get_query_params`, `experimental_rerun`, `experimental_dialog`, `experimental_fragment`, `experimental_user`, `experimental_connection`, `experimental_data_editor`).

**`st.dataframe`**
- Key params: `on_select`, `selection_mode` (adds `"single-row-required"`), `selection_default`, `column_config`, `hide_index`, `row_height`, `placeholder`, `alt`.
- `lazy=None` switches to lazy loading automatically above 150k rows, and lazy mode disables selections, search and CSV download.
- `NumberColumn(format=)` accepts `"percent"`, `"dollar"`, `"compact"`, `"localized"` or a printf string.

**Text and streaming**
- `st.write_stream` types out string chunks and parses OpenAI chunks natively.
- `st.markdown` renders `$…$` as LaTeX (`markdown.py` L131). Escape `$` in all narrative text.

**Caching**
- `st.cache_data` takes `scope="global"|"session"` and `refresh_mode`.
- DataFrames of 50,000+ rows are hashed from a 10,000-row sample (`runtime/caching/hashing.py` L53–54, L525). Never pass the fact frame into a cached function; pass small hashable arguments instead.

**Widgets**
- `st.date_input` in range mode returns a 0-, 1- or 2-tuple while the user is still picking, so guard with `len(v) == 2`.
  - Its default value is "today" (2026-10-20), which is outside the data. Always pass a value or pre-seed session_state.
- New on most widgets:
  - `bind="query-params"` binds the value to the URL. It requires a key, and the key becomes the URL parameter name. Bound values can only be changed programmatically through session_state.
  - `persist_state="page"|"session"`.
  - `on_change="ignore"`.
- `st.tabs(..., key, default, on_change="rerun")` enables lazy tab execution through `.open`.
- `segmented_control` and `pills` take `selection_mode`, `default`, `required`.
- `st.metric(..., border, format="percent"|"dollar"|"compact"|printf, delta_description, chart_data)`.
- `st.download_button(..., on_click="ignore")` avoids a rerun on download.
- `st.badge`, `st.space`, `st.dialog` and `st.fragment` exist.

**Secrets and config**
- `st.secrets["K"]` raises `StreamlitSecretNotFoundError` (a FileNotFoundError subclass) when there is no secrets.toml. `.get()` is not safe; catch `(KeyError, FileNotFoundError)`.
- Config keys: `[global] disableWidgetStateDuplicationWarning`, plus `[theme]` (`primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, `font`, `chartCategoricalColors`) and `[theme.sidebar]`.

**pandas 3.0.5, pyarrow 26, plotly 7.1** (run in memory)
- Parquet `date32` comes back as object `datetime.date`; the loader must convert. Strings come back as `str` dtype.
- `px.scatter(color=...)` produces one trace per category, so selection keys must travel in `custom_data`.
- Plotly annotations on log axes need log10 coordinates, while shapes take data values. Put the quadrant labels in paper coordinates.

### B. Item data in ga4_obfuscated_sample_ecommerce (public analyses)

**Categories**
- `item_category` is path-like and inconsistent. Seen values: `Home/Apparel/Men's / Unisex/`, `Home/Apparel/Women's/`, `Apparel`, `Women's`, `Lifestyle/`, `Lifestyle/Drinkware/`, `Lifestyle/Bags/`, `Home/Stationery/Stickers/`, `Home/Shop by Brand/Android/`, `Home/Sale/`, `Home/Eco-Friendly/`.
- It mixes collections (Sale, Eco-Friendly, Shop by Brand) with product types.
- Segment names contain `" / "`, so a naive split on "/" breaks "Men's / Unisex".

**Revenue and quantity** are populated on purchase rows. Examples: Google Zip Hoodie F/C $10,488 from 187 orders; Canteen Bottle Black 248 units from 128 orders.

**item_id to item_name is not 1:1**
- "Google Land & Sea Unisex Tee" has 2 ids.
- "Candy Cane Android Cardboard Sculpture" was purchased under a different id than the one viewed.
- Some names contain double spaces.

**Purchase events**
- 5,692 raw events become about 4.45k orders.
- 335 duplicate events across 326 orders.
- 906 events (16%) have no transaction_id, including all 450 zero-revenue ones.

**Sanity anchors for later checks**
- YouTube Women's Favorite Tee White: 6,885 viewers, 5 orders.
- 9 of the top 10 revenue items are hoodies, sweatshirts, fleeces or jackets.

**Not documented anywhere public**
- How many items a `view_item` event carries.
- Whether `view_promotion` items carry real item_ids. The HF flattening shows `promotion_name` as an event param (`ep_promotion_name`).
- Both go to profiling.

### Local CSV facts (computed from ga4_events_export.csv)

**Tracking gaps** (only views and purchases are consistent all season)
- add_to_cart: essentially untracked Nov 1–15 and Nov 21–24 (0–4 events/day), partial Nov 16–20 and Nov 25, consistent only from Nov 26.
- select_item: 0 until Nov 23.
- begin_checkout: double-fires relative to add_shipping_info on many days from about Nov 12.
- view_item, purchase and view_promotion: consistent all season.

**Holiday shape**
- Purchases surge on Mon Nov 23 (141) and Tue Nov 24 (147), before Thanksgiving.
- Thanksgiving 57, Black Friday 97, Cyber Monday 141.
- December peaks Dec 9–11 and Dec 15–16, then collapses after Dec 18 (52 on Dec 19, 20 on Dec 20).

**Site conversion (purchase events / view_item events)**

| Window | Purchases / views | Conversion |
|---|---|---|
| Baseline, Nov 1–22 | 1,234 / 104,250 | 1.18% |
| Cyber Week, Nov 23–30 | 820 / 44,389 | 1.85% |
| December, Dec 2–18 | 1,876 / 110,633 | 1.70% |
| Full season | 5,692 / 386,068 | 1.47% |

**By device:** desktop 1.43%, mobile 1.54%, tablet 1.33%.

**Other facts**
- US accounts for 44.5% of events (India 9.1%, Canada 7.5%).
- 360,129 distinct user|session keys; no empty session IDs.
- 109 countries; "(not set)" country appears in 32,208 rows.
- `user_pseudo_id` looks numeric (e.g. `1003046.9452926974`) and must be read as text.

### C. BigQuery auth and client

**pydata-google-auth 1.9.1**
- `get_user_credentials(scopes, client_id=None, client_secret=None, credentials_cache=cache.READ_WRITE, use_local_webserver=True, auth_local_webserver=None, redirect_uri=None)`. The code default is `True` even though the docstring says `False`.
- It opens a local server on localhost ports 8080–8089.
- Default cache: `%APPDATA%\.config\pydata\pydata_google_credentials.json`. Other cache options: `cache.ReadWriteCredentialsCache(dirname, filename)`, `cache.REAUTH`, `cache.NOOP`.
- Also available: `save_user_credentials(scopes, path, ...)`, `load_user_credentials(path)`, and `default(scopes, ...)`, which tries Application Default Credentials first.

**google-cloud-bigquery 3.46.2**
- `Client(project=, credentials=, location="US")`.
- `QueryJobConfig(dry_run, use_query_cache, maximum_bytes_billed, labels, job_timeout_ms)`.
- `RowIterator.to_arrow(create_bqstorage_client=True)` by default; pass `False` so the BigQuery Storage Read API isn't needed (helpful in the sandbox).
- `to_dataframe` maps DATE through db-dtypes by default. Using `to_arrow` avoids that dependency entirely.

**Billing floor (important)**
- BigQuery bills a minimum of 10 MB per referenced table plus 10 MB per query.
- `events_*` matches 92 shards, so every query bills at least about 0.9 GiB. A `maximum_bytes_billed` cap below about 1 GiB will fail.

**Fallbacks**
1. `winget install --id Google.CloudSDK --exact`, then `gcloud auth application-default login`. This writes ADC to `%APPDATA%\gcloud\application_default_credentials.json`, which `bigquery.Client()` picks up.
2. A service-account JSON via `GOOGLE_APPLICATION_CREDENTIALS`.
3. Last resort: paste the SQL files into the BigQuery console and save results to Drive as CSV.

### D. Google Trends (late 2026)

**trendspy 0.1.6** (released 2024-12-25; declares only `numpy>=1.19`, `pandas>=1.2`, `python-dateutil`, `requests`, so no pandas downgrade)
- Usage: `from trendspy import Trends; Trends(proxy=None).interest_over_time(keywords, timeframe="2020-10-04 2021-01-31", geo="US", cat=...)`.
- Returns a DataFrame with a time index and an `isPartial` column.
- A multi-keyword request normalizes all keywords against each other.

**Alternatives**
- pytrends 4.9.2 was archived on 2025-04-17 and mostly returns 429s now.
- The official Google Trends API has been an application-gated alpha since 2025-07-24.

**Strategy**
- Try trendspy, then pytrends, then manual CSVs.
- One keyword per request, with jittered 20–40 s sleeps and exponential backoff on 429.
- Cache each keyword's raw response so a failed run can resume. Never synthesize data.
- A roughly 4-month window should come back daily. Store daily values and aggregate to weeks in the app.

### E. OpenRouter through openai-python

**Version:** openai 3.35.1 is current (2026-10-18). Version 3.0.0 (2026-08-12) made HTTPX2 the default HTTP client, and httpx is no longer installed automatically.

**Usage**
- Client: `OpenAI(base_url="https://openrouter.ai/api/v1", api_key=...)`.
- Request: `client.chat.completions.create(model=..., messages=..., stream=True, max_tokens=..., extra_body={"models": [...], "reasoning": {...}}, extra_headers={"HTTP-Referer": ..., "X-OpenRouter-Title": ...})`.
- Iterate chunks via `chunk.choices[0].delta.content`. `chunk.model` reports the model that actually ran; `chunk.usage` is on the final chunk.

**Error behaviour (confirmed in `_streaming.py`)**
- SSE comment lines starting with ":" are dropped.
- A data object with a truthy top-level `"error"` raises `openai.APIError(message=error.message, body=error)`, so OpenRouter's HTTP-200 mid-stream error does surface as an exception.
- Errors before the stream starts raise `AuthenticationError` (401), `BadRequestError` (400), `RateLimitError` (429), or `APIStatusError` (402 and 5xx).

**OpenRouter streaming rules**
- The final usage chunk repeats `finish_reason` with an empty delta; treat it as accounting only, not a second terminal event.
- A 200 stream that contains only an error chunk is a failure.
- Closing the connection cancels billing on supported providers.
- `X-OpenRouter-Title` is the current header name; `X-Title` is an accepted alias.

### Dataviz palette validation (the dataviz skill's validator, run with node)

**Category colours on the scatter**
- Scatter and bubble charts need all-pairs colour-blind separation, and only 3 categorical hues pass.
- `#2a78d6, #eb6834, #1baf7a` passes: worst colour-blind ΔE 9.2, normal-vision ΔE 24.0. Aqua is 2.82:1 against white, which requires a table-view relief.
- Blue/aqua/violet fails (ΔE 4.5).
- So colour the 3 largest category groups (fixed by full-season views) and put everything else in "Other" `#8A8F98`. Add a "highlight one category" mode for the rest.

**Brand colours**
- Navy `#2F3B7D` is too dark for a marker colour (lightness 0.38, below the 0.43 band). Use it for text, badges, the primary colour and ramp endpoints.
- The quadrant meaning goes on background tints and badges with ink-coloured text, not on point colours.
- Navy/coral/gold/gray as four marker colours fails: navy is too dark, gray has no chroma, and gold vs gray is only ΔE 14.7.

**Other charts**
- Period bars: ordinal navy ramp `#9AA6D9 → #6270BD → #2F3B7D`, which passes the ordinal check.
- Heatmap: diverging coral `#F7646A` → gray `#F0EFEC` → navy `#2F3B7D`.
- Text contrast: navy on white 10.3:1, navy on lavender 8.9:1, coral on white 3.0:1 (so never use coral for text).

## 1. Corrections to your draft

1. **Use file pages, not function pages.** Put them in `views/*.py`, because AppTest can only switch to files. Shared filters stay in the entrypoint.
2. **The fact table must carry both cleaned `item_id` and `item_name`,** plus three view variants (`view_rows`, `view_events`, `primary_views`). Profiling then decides the product key and the view metric without re-querying. "Join view_item to purchase on item_id" becomes: aggregate all events on the same product key.
3. **De-duplicate purchases by `ecommerce.transaction_id`** (first event wins). Keep `purchase_events_raw` and `purchases_missing_txn` for audit. Revenue and units come only from counted purchases.
4. **Never use add_to_cart, select_item or begin_checkout in any comparison that crosses Nov 26.** They aren't tracked consistently before then. The pitch's purchase/view metric is the only funnel ratio that's clean all season; say so on the Data & Method page.
5. **Season windows come from the data.**
   - Baseline is Nov 1–22, not Nov 1–25, because the surge starts Nov 23.
   - Cyber Week is Nov 23–30.
   - December gifting is Dec 2–18 (shipping cutoff), not Dec 1–20.
   - Black Friday (Nov 26–29) and Cyber Monday (Nov 30–Dec 1) slots get their own short evidence windows, plus a seasonal prior (section 5).
6. **Quadrant consistency.**
   - The y-line is the pooled rate over all products in scope, not just the eligible ones.
   - The x-line defaults to the median views of eligible products, with top-25%, top-10% and custom options. The mockup's 10⁴ corresponds roughly to "top 25%".
   - Quadrant membership uses whatever is plotted (raw by default, as pitched, with an adjusted toggle).
   - Statistical confidence (80% posterior interval crossing the average) is shown as marker opacity.
7. **Shrinkage strength `m` is estimated once,** from full-season, all-device data by method of moments, then held fixed across filters so scores stay comparable. Only the prior mean (the category rate in scope) changes with filters.
8. **Device lean is indexed to each device's own baseline, with a formal test.** Raw device rates would just repeat site-level device differences. Tablet (2% of views) is excluded from the lean call.
9. **Cannibalization method.**
   - "Featured" proxy: a product's share of category views spikes. That is seasonality-neutral; promo impressions replace it if promotions carry item ids.
   - Outcome: the other product's conversion relative to its same-day category peers.
   - Test: exact conditional binomial, then Benjamini–Hochberg FDR, with minimum-sample guards. Details in section 5.
10. **Trends.** One keyword per request is right, but expect daily data. Use geo = US (44.5% of events). Compare each keyword only to itself, because each request is normalized to its own peak.
11. **LLM client: use a thin httpx client, not the openai SDK.**
    - httpx 0.28.1 is already installed in both Pythons; openai v3 adds the brand-new HTTPX2 stack.
    - We need OpenRouter-specific fields: the model that actually ran, error chunks, and usage.
    - `httpx.MockTransport` makes deterministic SSE tests easy.
    - The SDK path is equivalent (facts in section 0.E) if the team prefers it, behind the same `stream_chat()` interface.
12. **Caching.**
    - Base frames go through `st.cache_resource`, keyed by data directory plus manifest hash.
    - Derived results go through `st.cache_data`, keyed by data version plus filter tuples.
    - Never pass the 100k-row fact frame into a cached function (sampled hashing).
13. **Narrative safety.**
    - The model must cite product codes like `[W1]`, `[G1]`, `[X1]`; codes are validated after streaming.
    - `$` is escaped.
    - Generation is button-triggered (an auto toggle exists but defaults off), and results are cached by a hash of the payload.
    - The deterministic verdict is always visible first.
14. **Sidebar.**
    - Season presets drive the date range.
    - `bind="query-params"` on dates, devices, categories and min-views makes the URL shareable for meetings.
    - Filters that don't apply on a page are disabled, not hidden, so their state survives.
15. **Scatter mechanics.** Keys travel in `custom_data`. The selection is persisted by callback. Quadrant labels are in paper coordinates. The chart is never restyled from its own selection.
16. **Add a Slot Plan page.** It is the decision artifact: Black Friday, Cyber Monday and December slots, with device lean, cannibalization and trend flags, exportable as CSV or Markdown.
17. **Add a synthetic snapshot generator with planted patterns.** It unblocks parallel UI work before BigQuery access and gives tests known answers. The app shows a "SYNTHETIC DATA" banner whenever `manifest.source == "synthetic"`.
18. **BigQuery client settings.**
    - `to_arrow(create_bqstorage_client=False)`, `location="US"`, job labels.
    - Effective cap = max(user cap, 1.2 × dry-run bytes, number of shards × 10 MiB + 10 MiB).
    - Placeholder lists are written as literals in the SQL so files paste into the console as-is; a unit test asserts they match `config.PLACEHOLDERS`.
19. **CSV reading.** Every ID column is VARCHAR in DuckDB.
20. **Promotion fields.** Also export `promotion_name` from event_params (optional flag). If view_promotion items carry real item ids, we partly do have homepage-promotion history, which softens limit #1. Profiling decides.

## 2. File tree

Root: `C:\Users\abelv\OneDrive\Desktop\School\Fall2026\ITCS5122\Project\`

```
app.py                      entrypoint: page config, data guard, st.navigation(position="top"), sidebar filters + AI settings, pg.run()
views/                      file pages (AppTest-switchable), UI composition only
  feature_picks.py          Q1+Q2: verdict card, KPIs, quadrant scatter, inspector, ranked tables
  holiday_shift.py          Q3: windows, category x period bars, movers
  device_split.py           Q4: product x device heatmap + lean
  outside_signal.py         Phase 2b: Trends vs on-site, alerts
  cannibalization.py        Phase 2a: spike days, pair tests
  slot_plan.py              decision synthesis + export (+ optional AI verdict)
  data_method.py            provenance, proof query, cleaning, mapping, metrics, limits, SQL, manifest
ga4app/
  __init__.py
  config.py                 paths, env names, PERIODS/CAMPAIGNS, PLACEHOLDERS, thresholds, palette, models, decisions defaults
  contracts.py              snapshot schemas, validate_frame, atomic parquet/json writers, manifest helpers
  data.py                   load_snapshot (pure) + get_snapshot (st.cache_resource)
  taxonomy.py               category path parsing, group mapping, brand, name_key
  metrics.py                filtering, aggregation, EB shrinkage, quadrants, action scores, period shift, movers, device lean, campaign scores, slot plan
  phase2.py                 exposure spikes, cannibalization tests, BH, trend index, trending alerts, lead/lag
  trends.py                 trendspy/pytrends/manual-CSV adapters, normalization, provenance
  charts.py                 Plotly figure builders + template (no Streamlit imports)
  llm.py                    OpenRouter streaming client (httpx), key resolution, fake transport
  narrative.py              payload, prompt, rule-based verdict, citation check, md escaping, stream wrapper
  annotate.py               chart notes: anchors, number check, rule-based and AI notes (section 7b)
  state.py                  widget keys, Filters from widgets, presets/callbacks, sidebar renderer
  ui.py                     shared components: header, KPI row, chart+table twin, verdict block, inspector, banners
  bq.py                     credentials, client, dry run, capped run -> Arrow, proof query
  synthetic.py              synthetic snapshots with planted truth
sql/                        proof_counts.sql, profile_*.sql (7), export_item_daily.sql, export_item_attrs.sql, export_promo_daily.sql
scripts/                    bq_auth.py, profile_bigquery.py, export_bigquery.py, build_site_aggregates.py,
                            fetch_trends.py, build_snapshots.py, verify_snapshots.py, make_synthetic_data.py
config/trends_keywords.csv  group_type,group_value,keyword,notes (finalized after profiling)
data/raw/                   bq_item_daily.parquet, bq_item_attrs.parquet, bq_promo_daily.parquet, export_log.json, trends/<keyword>.json
data/trends_manual/         optional Google Trends "multiTimeline.csv" downloads
data/                       item_daily, product_dim, promo_daily, site_daily, site_funnel, trends_daily (.parquet);
                            overview.json, profile.json, bq_proof.json, trends_meta.json, manifest.json
tests/                      conftest.py, test_*.py (see section 8), fixtures/mini_events.csv
.streamlit/config.toml, .streamlit/secrets.toml.example
requirements.txt            streamlit==1.66.0, plotly==7.1.0, pandas==3.0.2, numpy==2.3.5, pyarrow==26.0.0, scipy==1.16.3, httpx==0.28.1
requirements-pipeline.txt   google-cloud-bigquery==3.46.2, pydata-google-auth==1.9.1, duckdb==1.5.6, trendspy==0.1.6, pytrends==4.9.2 (optional)
requirements-dev.txt        pytest==9.1.1
pyproject.toml              [tool.pytest.ini_options] pythonpath=["."], testpaths=["tests"], markers=["app"]
.gitignore                  ga4_events_export.csv, .streamlit/secrets.toml, *.tmp, __pycache__, .pytest_cache
README.md
```

**`.streamlit/config.toml`**
- `[theme]`: base light, primaryColor `#2F3B7D`, backgroundColor `#FFFFFF`, secondaryBackgroundColor `#ECEEF8`, textColor `#1F2433`.
  - Fonts from Google Fonts: Inter for text (the font the charts already ask for) and Plus Jakarta Sans for headings (weights 800, 700, 700, 600, 600, 600).
  - Inter is wider than Streamlit's default font, so `baseFontSize = 15` and `metricValueFontSize = "1.875rem"` (weight 700). At 16px the KPI tiles cut off "2,528,496" and the seven nav items no longer fit on one line at 1440px.
  - The page background stays white: the charts paint their own white surface, and the theme stays light-only for the same reason.
- `[theme.sidebar]`: navy `#2F3B7D` background, `#3C4A92` inputs, white text, `#5A67AD` borders, pale yellow `#F8E79A` for selected options, toggles and sliders (8:1 on navy; coral would be 3.4:1).
- `app.py`: a Material icon for each page in the top nav, `page_icon=":material/storefront:"`, and `st.logo` from `assets/`. `logo.svg` is a 250x40 wordmark (Streamlit's 200x32 logo box at 1.25x), drawn with light text for the navy sidebar. `logo_icon.svg` is the square mark shown in the header when the sidebar is collapsed. The logo is set after the snapshot guard, because the error page has no sidebar.
- `[global]`: `disableWidgetStateDuplicationWarning = true`.
- `[browser]`: `gatherUsageStats = false`.
- If OneDrive sync triggers spurious reruns, set `server.fileWatcherType = "poll"` or `"none"`.

**`secrets.toml.example`**: `OPENROUTER_API_KEY = "sk-or-..."` and a `[gcp]` section with `project_id = "..."`.

## 3. Data contracts

Arrow types are listed; the pandas type after loading is noted where it changes. A `?` means nullable. Every writer goes to a temp file then `os.replace`, which also protects against OneDrive locks.

**Raw BigQuery exports** (`data/raw/`)
- **`bq_item_daily.parquet`**
  - Grain: unique on (event_date, device_category, item_id, item_name).
  - Columns: `event_date date32`, `device_category string`, `item_id string?`, `item_name string?`.
  - `int64` counts: `view_rows`, `view_events`, `primary_views`, `add_to_carts`, `checkouts`, `purchases`, `purchase_events_raw`, `purchases_missing_txn`, `units`, `promo_views`, `promo_clicks`.
  - `revenue_usd float64`.
- **`bq_item_attrs.parquet`**: `item_id? item_name? item_category? item_category2? item_brand?` (strings), `item_rows view_rows purchase_rows int64`, `median_price_usd float64?`, `first_seen last_seen date32`.
- **`bq_promo_daily.parquet`**: `event_date date32`, `device_category`, `event_name`, `promotion_id?`, `promotion_name?`, `creative_name?`, `creative_slot?`, `item_id?`, `item_name?`, `ep_promotion_name?` (strings), `item_rows int64`, `events int64`.

**App snapshots** (`data/`)
- **`item_daily.parquet`**
  - Primary key: (event_date, device_category, product_key).
  - `event_date date32` (loaded as `datetime64[ns]`); `device_category string` (loaded as Categorical `desktop, mobile, tablet, other`); `product_key string`.
  - `views int64` is a copy of the chosen view metric. Also: `view_rows`, `view_events`, `primary_views`, `add_to_carts`, `checkouts`, `purchases`, `purchase_events_raw`, `purchases_missing_txn`, `units`, `promo_views`, `promo_clicks` (all int64) and `revenue_usd float64`.
  - Invariants: everything ≥ 0; `view_events ≤ view_rows`; `purchases ≤ purchase_events_raw`.
- **`product_dim.parquet`**
  - `product_key string` (primary key), `item_name string` (display name: most views), `item_ids string` (pipe-joined), `n_item_ids int32`, `n_names int32`.
  - `category_raw string?`, `category_path string` (e.g. "Apparel > Men's / Unisex"), `category_group string` (one of `config.CATEGORY_GROUPS`), `brand string` (Google / YouTube / Android / Other).
  - `median_price_usd float64?`, `first_seen last_seen date32`, `total_views total_purchases int64`.
- **`promo_daily.parquet`** (optional): `event_date`, `device_category`, `promotion_name` (coalesced from the item and event-param values), `creative_slot`, `product_key?`, `promo_views int64`, `promo_clicks int64`.
- **`site_daily.parquet`** (from the CSV): `event_date date32`, `device_category`, `event_name`, `events users sessions int64`. Users and sessions are distinct within each cell.
- **`site_funnel.parquet`**: `event_date date32` (session start), `device_category`, `sessions view_sessions cart_sessions checkout_sessions purchase_sessions view_purchase_sessions int64`.
- **`trends_daily.parquet`**: `date date32`, `keyword`, `group_type` ("category" or "brand"), `group_value`, `geo`, `interest float64` (0–100; "<1" becomes 0.5), `is_partial bool`, `source` ("trendspy", "pytrends" or "manual_csv"), `fetched_at timestamp[us, UTC]`.

**JSON files**
- **`overview.json`**: `event_count`, `user_count`, `day_count`, `session_count`, `date_min`, `date_max`, `events_by_device{}`, `events_by_name{}`, `tracking_gaps{event_name: {gap_days, first_reliable_date}}`, `source_csv`, `generated_at`.
- **`bq_proof.json`**: `event_count`, `user_count`, `day_count`, `sql`, `job_id`, `bytes_processed`, `bytes_billed`, `project`, `run_at`.
- **`profile.json`**: `queries{name: rows}`, `bytes{name: n}`, `decisions{product_key, view_metric, featured_signal, notes[]}`, `generated_at`.
- **`trends_meta.json`**: `keywords[]`, `geo`, `timeframe`, `method_used{keyword: method}`, `errors{}`, `status` ("ok", "partial" or "unavailable").
- **`manifest.json`**: `schema_version`, `source` ("bigquery+csv" or "synthetic"), `generated_at`, `decisions{}`, `category_colors{group: hex}` (fixed by full-season views, so colours follow the entity), `files{name: {rows, sha256, built_from[]}}`, `periods{}`, `placeholders[]`, `shrinkage_m`.

## 4. BigQuery SQL

`CLEAN(x)` below is shorthand. In the files it is written out literally as `IF(TRIM(x) IN ('', '(not set)', '<Other>', '(data deleted)', 'Not available in demo dataset'), NULL, TRIM(x))`. A NULL input passes through as NULL. No temp UDF is used, so each file pastes into the console unchanged. `T` means `` `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` `` and `W` means `_TABLE_SUFFIX BETWEEN '20201101' AND '20210131'`.

```sql
-- proof_counts.sql  (verbatim pitch query; W would not change results)
SELECT COUNT(*) AS event_count, COUNT(DISTINCT user_pseudo_id) AS user_count,
       COUNT(DISTINCT event_date) AS day_count FROM T;

-- profile_items_per_event.sql
SELECT event_name, COUNT(*) AS events, COUNTIF(ARRAY_LENGTH(items) > 0) AS events_with_items,
       AVG(ARRAY_LENGTH(items)) AS avg_items, MAX(ARRAY_LENGTH(items)) AS max_items,
       APPROX_QUANTILES(ARRAY_LENGTH(items), 100)[OFFSET(99)] AS p99_items
FROM T WHERE W GROUP BY event_name ORDER BY events DESC;

-- profile_item_fields.sql  (placeholders/completeness per event; raw top values expose unknown placeholders)
SELECT e.event_name, COUNT(*) AS item_rows,
  COUNTIF(CLEAN(it.item_id) IS NULL) AS bad_item_id, COUNTIF(CLEAN(it.item_name) IS NULL) AS bad_item_name,
  COUNTIF(CLEAN(it.item_category) IS NULL) AS bad_item_category,
  COUNTIF(it.quantity IS NULL) AS null_quantity, COUNTIF(it.price_in_usd IS NULL) AS null_price,
  COUNTIF(it.item_revenue_in_usd IS NULL) AS null_revenue, COUNTIF(it.item_revenue_in_usd = 0) AS zero_revenue,
  COUNT(DISTINCT CLEAN(it.item_id)) AS distinct_ids, COUNT(DISTINCT CLEAN(it.item_name)) AS distinct_names,
  APPROX_TOP_COUNT(it.item_id, 8) AS top_raw_ids, APPROX_TOP_COUNT(it.item_name, 8) AS top_raw_names,
  APPROX_TOP_COUNT(it.promotion_name, 10) AS top_promotion_names, APPROX_TOP_COUNT(it.creative_slot, 10) AS top_slots,
  APPROX_TOP_COUNT(it.item_list_name, 8) AS top_list_names
FROM T AS e, UNNEST(e.items) AS it WHERE W GROUP BY e.event_name ORDER BY item_rows DESC;

-- profile_categories.sql
SELECT it.item_category, it.item_category2,
  COUNTIF(e.event_name = 'view_item') AS view_rows, COUNTIF(e.event_name = 'purchase') AS purchase_rows,
  COUNT(DISTINCT CLEAN(it.item_id)) AS items,
  ARRAY_AGG(DISTINCT CLEAN(it.item_name) IGNORE NULLS LIMIT 5) AS example_names
FROM T AS e, UNNEST(e.items) AS it WHERE W AND e.event_name IN ('view_item','purchase')
GROUP BY 1, 2 ORDER BY view_rows DESC LIMIT 500;

-- profile_keys.sql  (id<->name cardinality + purchase-row coverage by id vs by name)
WITH r AS (
  SELECT e.event_name, CLEAN(it.item_id) AS item_id,
         LOWER(REGEXP_REPLACE(CLEAN(it.item_name), r'\s+', ' ')) AS name_key
  FROM T AS e, UNNEST(e.items) AS it WHERE W AND e.event_name IN ('view_item','add_to_cart','purchase')),
pairs AS (SELECT item_id, name_key, COUNT(*) AS n FROM r WHERE item_id IS NOT NULL AND name_key IS NOT NULL GROUP BY 1, 2),
by_id AS (SELECT item_id, COUNT(*) AS names, SUM(n) AS n FROM pairs GROUP BY 1),
by_name AS (SELECT name_key, COUNT(*) AS ids, SUM(n) AS n FROM pairs GROUP BY 1),
vi AS (SELECT DISTINCT item_id FROM r WHERE event_name = 'view_item' AND item_id IS NOT NULL),
vn AS (SELECT DISTINCT name_key FROM r WHERE event_name = 'view_item' AND name_key IS NOT NULL),
purch AS (SELECT vi.item_id IS NOT NULL AS id_viewed, vn.name_key IS NOT NULL AS name_viewed
          FROM r AS p LEFT JOIN vi ON p.item_id = vi.item_id LEFT JOIN vn ON p.name_key = vn.name_key
          WHERE p.event_name = 'purchase')
SELECT (SELECT COUNT(*) FROM by_id) AS distinct_ids, (SELECT COUNT(*) FROM by_name) AS distinct_names,
  (SELECT COUNTIF(names > 1) FROM by_id) AS ids_multi_name, (SELECT COUNTIF(ids > 1) FROM by_name) AS names_multi_id,
  (SELECT SAFE_DIVIDE(SUM(IF(ids > 1, n, 0)), SUM(n)) FROM by_name) AS row_share_multi_id_names,
  (SELECT COUNT(*) FROM purch) AS purchase_rows, (SELECT COUNTIF(id_viewed) FROM purch) AS purchase_rows_id_viewed,
  (SELECT COUNTIF(name_viewed) FROM purch) AS purchase_rows_name_viewed,
  (SELECT ARRAY_AGG(STRUCT(name_key, ids, n) ORDER BY n DESC LIMIT 25) FROM by_name WHERE ids > 1) AS top_multi_id_names;

-- profile_purchases.sql
WITH p AS (SELECT IF(TRIM(ecommerce.transaction_id) IN ('', '(not set)'), NULL, TRIM(ecommerce.transaction_id)) AS txn,
                  ecommerce.purchase_revenue_in_usd AS rev, ARRAY_LENGTH(items) AS n_items
           FROM T WHERE W AND event_name = 'purchase')
SELECT COUNT(*) AS purchase_events, COUNTIF(txn IS NULL) AS missing_txn, COUNT(DISTINCT txn) AS distinct_txn,
  COUNTIF(txn IS NOT NULL) - COUNT(DISTINCT txn) AS duplicate_events,
  COUNTIF(IFNULL(rev, 0) = 0) AS zero_revenue_events, COUNTIF(n_items = 0) AS purchases_without_items,
  ROUND(SUM(rev), 2) AS purchase_revenue_usd_raw FROM p;

-- profile_devices.sql  (cross-check vs CSV: desktop 2,498,330 / mobile 1,704,069 / tablet 93,185)
SELECT LOWER(IFNULL(device.category, 'unknown')) AS device_category, COUNT(*) AS events,
       COUNTIF(event_name = 'view_item') AS view_item_events, COUNTIF(event_name = 'purchase') AS purchase_events
FROM T WHERE W GROUP BY 1 ORDER BY events DESC;

-- profile_view_item_composition.sql  (OPTIONAL --with-event-params; scans event_params, about +1-3 GiB)
WITH v AS (
  SELECT (SELECT LOWER(ep.value.string_value) FROM UNNEST(event_params) AS ep WHERE ep.key = 'page_title' LIMIT 1) AS page_title,
         ARRAY(SELECT AS STRUCT LOWER(TRIM(x.item_name)) AS name_key FROM UNNEST(items) AS x WITH OFFSET AS o ORDER BY o) AS its
  FROM T WHERE W AND event_name = 'view_item')
SELECT COUNT(*) AS view_events, COUNTIF(ARRAY_LENGTH(its) > 1) AS multi_item_events,
  COUNTIF(ARRAY_LENGTH(its) > 1 AND (SELECT COUNT(DISTINCT name_key) FROM UNNEST(its)) = 1) AS multi_item_same_name,
  COUNTIF(ARRAY_LENGTH(its) > 0 AND STRPOS(page_title, its[SAFE_OFFSET(0)].name_key) > 0) AS first_item_in_title,
  COUNTIF(EXISTS(SELECT 1 FROM UNNEST(its) AS i WHERE STRPOS(page_title, i.name_key) > 0)) AS any_item_in_title
FROM v;
```

```sql
-- export_item_daily.sql   Grain: event_date x device_category x item_id x item_name (cleaned)
WITH base AS (
  SELECT PARSE_DATE('%Y%m%d', event_date) AS event_date, event_name, event_timestamp,
         LOWER(IFNULL(device.category, 'unknown')) AS device_category,
         IF(TRIM(ecommerce.transaction_id) IN ('', '(not set)'), NULL, TRIM(ecommerce.transaction_id)) AS transaction_id,
         FARM_FINGERPRINT(CONCAT(user_pseudo_id, '|', CAST(event_timestamp AS STRING), '|', event_name)) AS event_key,
         items
  FROM T
  WHERE W AND event_name IN ('view_item','add_to_cart','begin_checkout','purchase','view_promotion','select_promotion')),
first_purchase AS (            -- one counted purchase event per valid transaction_id
  SELECT event_key FROM base WHERE event_name = 'purchase' AND transaction_id IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY event_timestamp, event_key) = 1),
events AS (
  SELECT b.event_date, b.event_name, b.device_category, b.event_key, b.items,
         b.event_name = 'purchase' AND (b.transaction_id IS NULL OR fp.event_key IS NOT NULL) AS counted_purchase,
         b.event_name = 'purchase' AND b.transaction_id IS NULL AS purchase_missing_txn
  FROM base AS b LEFT JOIN first_purchase AS fp ON b.event_name = 'purchase' AND b.event_key = fp.event_key),
item_rows AS (
  SELECT e.event_date, e.device_category, e.event_name, e.event_key, e.counted_purchase, e.purchase_missing_txn,
         off AS item_offset, CLEAN(it.item_id) AS item_id, CLEAN(it.item_name) AS item_name,
         it.quantity, it.item_revenue_in_usd
  FROM events AS e, UNNEST(e.items) AS it WITH OFFSET AS off)
SELECT event_date, device_category, item_id, item_name,
  COUNTIF(event_name = 'view_item')                                     AS view_rows,
  COUNT(DISTINCT IF(event_name = 'view_item', event_key, NULL))         AS view_events,
  COUNTIF(event_name = 'view_item' AND item_offset = 0)                 AS primary_views,
  COUNT(DISTINCT IF(event_name = 'add_to_cart', event_key, NULL))       AS add_to_carts,
  COUNT(DISTINCT IF(event_name = 'begin_checkout', event_key, NULL))    AS checkouts,
  COUNT(DISTINCT IF(counted_purchase, event_key, NULL))                 AS purchases,
  COUNT(DISTINCT IF(event_name = 'purchase', event_key, NULL))          AS purchase_events_raw,
  COUNT(DISTINCT IF(purchase_missing_txn, event_key, NULL))             AS purchases_missing_txn,
  IFNULL(SUM(IF(counted_purchase, quantity, NULL)), 0)                  AS units,
  ROUND(IFNULL(SUM(IF(counted_purchase, item_revenue_in_usd, NULL)), 0), 2) AS revenue_usd,
  COUNT(DISTINCT IF(event_name = 'view_promotion', event_key, NULL))    AS promo_views,
  COUNT(DISTINCT IF(event_name = 'select_promotion', event_key, NULL))  AS promo_clicks
FROM item_rows WHERE item_id IS NOT NULL OR item_name IS NOT NULL
GROUP BY event_date, device_category, item_id, item_name;
```

If the dry run shows unused `items` subfields being scanned (more than about 3 GiB), project only the needed subfields in `base`: `ARRAY(SELECT AS STRUCT x.item_id, x.item_name, x.quantity, x.item_revenue_in_usd FROM UNNEST(items) x WITH OFFSET o ORDER BY o) AS items`.

```sql
-- export_item_attrs.sql   (attribute combos with counts; canonical name/category resolved in Python)
WITH r AS (
  SELECT PARSE_DATE('%Y%m%d', e.event_date) AS event_date, e.event_name,
         CLEAN(it.item_id) AS item_id, CLEAN(it.item_name) AS item_name, CLEAN(it.item_category) AS item_category,
         CLEAN(it.item_category2) AS item_category2, CLEAN(it.item_brand) AS item_brand, it.price_in_usd
  FROM T AS e, UNNEST(e.items) AS it WHERE W AND e.event_name IN ('view_item','add_to_cart','purchase'))
SELECT item_id, item_name, item_category, item_category2, item_brand,
  COUNT(*) AS item_rows, COUNTIF(event_name = 'view_item') AS view_rows, COUNTIF(event_name = 'purchase') AS purchase_rows,
  APPROX_QUANTILES(IF(price_in_usd > 0, price_in_usd, NULL), 2 IGNORE NULLS)[SAFE_OFFSET(1)] AS median_price_usd,
  MIN(event_date) AS first_seen, MAX(event_date) AS last_seen
FROM r WHERE item_id IS NOT NULL OR item_name IS NOT NULL
GROUP BY item_id, item_name, item_category, item_category2, item_brand;

-- export_promo_daily.sql   (ep_promotion_name line only with --with-event-params)
WITH pr AS (
  SELECT PARSE_DATE('%Y%m%d', e.event_date) AS event_date, LOWER(IFNULL(e.device.category, 'unknown')) AS device_category,
         e.event_name, CLEAN(it.promotion_id) AS promotion_id, CLEAN(it.promotion_name) AS promotion_name,
         CLEAN(it.creative_name) AS creative_name, CLEAN(it.creative_slot) AS creative_slot,
         CLEAN(it.item_id) AS item_id, CLEAN(it.item_name) AS item_name,
         (SELECT ep.value.string_value FROM UNNEST(e.event_params) AS ep WHERE ep.key = 'promotion_name' LIMIT 1) AS ep_promotion_name,
         FARM_FINGERPRINT(CONCAT(e.user_pseudo_id, '|', CAST(e.event_timestamp AS STRING))) AS event_key
  FROM T AS e LEFT JOIN UNNEST(e.items) AS it
  WHERE W AND e.event_name IN ('view_promotion','select_promotion'))
SELECT event_date, device_category, event_name, promotion_id, promotion_name, creative_name, creative_slot,
       item_id, item_name, ep_promotion_name, COUNT(*) AS item_rows, COUNT(DISTINCT event_key) AS events
FROM pr GROUP BY 1,2,3,4,5,6,7,8,9,10;
```

**Bytes control (`ga4app/bq.py`)**
1. Dry-run every query with `dry_run=True, use_query_cache=False`.
2. Print GiB processed and the referenced-table count.
3. Abort if the per-query or total estimate exceeds `--max-gib` (5 for export, 3 for profiling) unless `--yes` is passed.
4. Run with `maximum_bytes_billed = max(cap, ceil(1.2 × dry_bytes), n_tables × 10 MiB + 10 MiB)`, `use_query_cache=True`, `location="US"`, `labels={"app": "itcs5122-team9", "step": name}`.

- Never use `SELECT *`. `event_params` appears only in the two optional queries.
- `_TABLE_SUFFIX` only documents the window, since all 92 shards are needed anyway.
- Log every job (job_id, bytes processed and billed, cache hit, SHA-256 of the SQL) to `data/raw/export_log.json`.
- Expected total: about 10–30 GiB of the 1 TiB monthly free tier.

## 5. Metric formulas (`ga4app/metrics.py`, `ga4app/phase2.py`)

**Notation**
- Scope S = date range × devices × categories.
- For each product i: V = views (chosen metric), P = counted purchases, R = revenue.
- Raw rate: `cr_raw = P/V` for V > 0.
- Eligible: V ≥ min_views.
- Purchases-exceed-views flag: P > V. The posterior then uses `V_eff = max(V, P)`.

**Pooled rates**
- Site average: `p̄ = ΣP/ΣV` over all products in S with V > 0.
- Category prior mean: `μ_c` = the same ratio within category c ∩ S.

**Shrinkage strength (method of moments; full season, all devices, products with V ≥ 100)**
- `τ² = [Σ V_i(r_i − μ_c(i))² − Σ μ_c(i)(1 − μ_c(i))] / ΣV_i`
- `m = clamp(μ̄(1 − μ̄)/τ² − 1, 100, 20000)`, using the overall pooled rate μ̄. If `τ² ≤ 0`, `m = 20000`.
- Store `m` in the manifest and show it on Data & Method.

**Posterior**
- `a = P + mμ_c`, `b = max(V − P, 0) + m(1 − μ_c)`.
- `cr_adj = a/(a + b)`; 80% interval from `scipy.stats.beta.ppf([.1, .9], a, b)`.
- Confidence:
  - "above" if the 10% bound is above p̄;
  - "below" if the 90% bound is below p̄;
  - otherwise "uncertain".

**Quadrants**
- `T_v` = median of V over eligible products (or p75, p90, custom). `T_c = p̄`.
- With y = plotted metric (`cr_raw` or `cr_adj`):

| | y ≥ T_c | y < T_c |
|---|---|---|
| V ≥ T_v | Winners | Wasted exposure |
| V < T_v | Hidden gems | Low priority |

**Action scores**
- Retire (Wasted exposure): `forgone = V·max(p̄ − cr_adj, 0)`, read as "purchases lost compared with an average product".
- Promote (Hidden gems): `lift_per_1k = 1000·max(cr_adj − p̄, 0)`, read as "extra purchases per 1,000 extra views". Requires confidence other than "below".
- Winners are ranked by `cr_adj`. Revenue per view: `rpv = R/V`.
- Codes: W1…, G1…, X1… (top 5 of each).
- Verdict "Feature" list: top 3 by `cr_adj` among Hidden gems ∪ Winners with confidence "above". Fall back to "uncertain" ones if fewer than 3.
- Swap gain: `1000·(cr_adj_G − cr_adj_X)`.

**Period shift (Q3)**
- Per category group and period: `cr = ΣP/ΣV` with a 95% Wilson interval and `lift = cr_focus/cr_base − 1`.
- Two-proportion z-test p-value, only when both periods have P ≥ 10; otherwise flag as low sample.
- Note on the page that events aren't independent, so the intervals are optimistic.

**Quadrant movers**
- Classify each product per period with that period's own thresholds and category priors (same m).
- `Δ = logit(cr_adj_f) − logit(cr_adj_b)`, `se² ≈ 1/a + 1/b` per period, `z = Δ/√(se_b² + se_f²)`.
- Rising or Fading if `|z| ≥ 1.96` and the relative change is at least 25%.
- Also a 4×4 transition-count matrix.

**Device lean (Q4)**
- Device baseline `p̄_d`. Expected purchases `E_{i,d} = V_{i,d}·p̄_d`.
- `RR = (O_m/E_m)/(O_d/E_d)` (mobile vs desktop).
- p-value: `binomtest(O_m, O_m + O_d, E_m/(E_m + E_d))`, two-sided.
- Calls:
  - Mobile-first if `RR ≥ 1.25` and `p < 0.10`;
  - Desktop-first if `RR ≤ 0.8` and `p < 0.10`;
  - otherwise "Both";
  - "Not enough data" if `O_m + O_d < 10`.
- Heatmap cell: `index = cr_adj_{i,d}/p̄_d`, where `cr_adj_{i,d} = (P + m·μ_{i,d})/(V + m)` and `μ_{i,d} = cr_adj_i·p̄_d/p̄`. Colour on `log2(index)` clipped to ±1.

**Campaign score (Slot Plan)**
- Season prior: `s_i` = full-season `cr_adj`.
- Category uplift `u_{c,w} = cr_{c,w}/cr_{c,season}`, using the site uplift instead when `V_{c,w} < 2000`.
- `μ_{i,w} = s_i·u_{c,w}`, then `score = (P_{i,w} + m·μ_{i,w})/(V_{i,w} + m)`, reported per 1,000 views.
- Slot selection is greedy by score per campaign. It skips excluded products and, when the toggle is on, any product whose flagged cannibalization partner is already picked. It optionally prevents the same product appearing in more than one campaign.

**Cannibalization (Phase 2a)**
- Scope: within a category, the top K = 12 products by views; days d in S.
- Spike days for A: `share_A(d) = V_A(d)/V_c(d)`. A spike day has `share_A(d) ≥ ratio × median_d share_A` (ratio 2.0; 1.5 or 3 selectable) and `V_A(d) ≥ 30`.
  - Use the promo-impression share instead if promotions carry item ids.
  - Guards: at least 3 spike days and at least 10 other days.
- For each other product B, the peers are the category minus {A, B}:
  - `cr_P(d) = ΣP_peers/ΣV_peers`, `e_B(d) = V_B(d)·cr_P(d)`.
  - `O_s, E_s` are B's observed and expected purchases on spike days; `O_n, E_n` on other days.
- Test: under H0, `O_s | n=O_s+O_n ~ Binomial(n, π0 = E_s/(E_s+E_n))`.
  - `RR = (O_s/E_s)/(O_n/E_n)`, with a Clopper–Pearson interval mapped to RR via `π/(1−π)·E_n/E_s`.
  - One-sided p-value for a dip: `binomtest(..., alternative="less")`.
  - q-values: `scipy.stats.false_discovery_control(p, method="bh")` across all tested pairs.
- Guards: `n ≥ 10`, `n·π0 ≥ 3`, and B's views on spike days ≥ 200; otherwise "Insufficient data".
- Secondary "steal" check: Fisher exact test (`alternative="less"`) on the 2×2 table [B purchases vs other-peer purchases] × [spike vs other days].
- Status:
  - "Likely cannibalized" if `RR ≤ 0.8` and `q ≤ 0.10`;
  - "Possible" if `RR ≤ 0.8` and `p ≤ 0.05`;
  - otherwise "No evidence".
- Why this controls for seasonality: shares ignore traffic surges, and peer-relative conversion absorbs shocks common to the whole category on a given day.

**Trending alert (Phase 2b)**
- `M_ext = mean(interest, focus)/mean(interest, base)`, where base is the Nov 1–22 baseline. Unreliable if the base mean is below 5.
- `M_site = (V_g,focus/days)/(V_g,base/days)`. `gap = M_ext/M_site`.
- Flag when `M_ext ≥ 1.20`, `gap ≥ 1.15`, and `V_g,base ≥ 1000`.
- Promote list: products in the flagged group with `cr_adj > μ_g` and `V < T_v`, ranked by `lift_per_1k`.
- Lead/lag: weekly (W-SUN) percent changes, correlations at lags 0–3, labelled descriptive only (about 13 weeks of data).

## 6. Pages (one row of controls per page; every chart has a "Show data" twin with download)

**Global (app.py)**
- `st.set_page_config(layout="wide")`.
- `get_snapshot()`. If snapshots are missing, show an error with exact commands and `st.stop()`. Show the synthetic banner when relevant.
- `pg = st.navigation([...7 file pages...], position="top")`; the page ID comes from page identity; then `state.render_sidebar(snapshot, page_id)` and `pg.run()`.
- Sidebar controls:
  - Window presets (`segmented_control`): Full season, Baseline, Cyber Week, BF weekend, Cyber Monday, December, January. Choosing one sets the dates; editing the dates clears it to "custom".
  - Date range: bound to query params, with a `len==2` guard.
  - Devices (multi-select pills; none selected means all, with a note).
  - Category groups (multiselect).
  - Min views (`select_slider`: 0, 25, 50, 100, 200, 300, 500, 1000, 2000, 5000; default 200).
  - Reset button and a data-freshness caption.
  - Expander "AI narrator":
    - model selectbox: `anthropic/claude-opus-5.5` (default), `anthropic/claude-sonnet-5.5`, `anthropic/claude-haiku-5.5`, `anthropic/claude-fable-5.1`, `openrouter/free`, `google/gemma-4-31b-it:free`, or a custom ID;
    - key-source status;
    - session-only password input;
    - "Auto-write when filters change" toggle, off by default.

**1. Feature Picks (Q1, Q2; default page)**
- Header with a scope caption.
- Verdict card: rule-based immediately; "Write AI verdict" streams the AI version.
  - Header row: a gavel icon, the title ("Verdict"; "Verdict on the plan" on the Slot Plan) and a badge for the source (grey "Rule-based", violet "AI · model", "Writing…" while streaming). The body drops the text's leading `**Verdict:**`, also while streaming, since the header says it; the generated text keeps it.
  - "Write AI verdict" and "Write AI notes" are primary buttons with the `auto_awesome` icon, so the AI actions stand out.
- KPI tiles: eligible products; average conversion (delta vs full season); views; purchases; forgone purchases.
- Page controls: y-axis Raw/Adjusted; high-traffic cutoff; colour by Category or Highlight one; bubble size Revenue or Purchases.
- Main chart, the quadrant bubble scatter:
  - Axes: log x for views; y as a percentage, clipped at the 99th percentile with an overflow marker.
  - Thresholds: dashed 1px ink lines with labels, the only dashed lines in the app.
  - Background: 6–10% quadrant tints.
  - Labels: corner badges HIDDEN GEMS, WINNERS, LOW PRIORITY, WASTED EXPOSURE in paper coordinates.
  - Markers: 2px white ring around each marker; uncertain points at 45% opacity; direct labels only for codes of the top 3 per actionable quadrant.
  - Interaction: hover tooltip; `on_select` callback sets `inspect_key`.
- Right column: product inspector (stats, posterior interval, daily views and purchases with window shading, device bars, period rates, flags, "Add to slot plan").
- Tabs:
  - Hidden gems: promote (Q2), by lift per 1k views.
  - Wasted exposure: retire (Q1), by forgone purchases.
  - Winners: keep.
  - All products: the efficiency ranking.
  - Every tab uses single-row selection that sets the inspected product, plus columns for code, conversion, interval, revenue per view, device lean and flags.

**2. Holiday Shift (Q3)**
- The date filter is disabled here, with a note that the page compares fixed windows.
- Comparison selector: Baseline vs Cyber Week (default), Baseline vs December, or Cyber Week vs December.
- KPI tiles: conversion, views per day, purchases per day, each with deltas.
- Daily conversion line with shaded windows (single axis).
- Grouped bars: category × period, 95% intervals, lift labels for significant changes, ordinal navy colours.
- Movers: transition matrix plus top risers and fallers, e.g. "items that became Hidden gems in Cyber Week".
- Tracking-gap caveat.

**3. Device Split (Q4)**
- The device filter is disabled.
- Candidates: top Winners + Hidden gems, or top by adjusted rate, or a manual pick; N from 10 to 40.
- KPI tiles per device.
- Heatmap of products × devices: diverging index colour, cell text = conversion, hover shows views and purchases.
- Lean column and summary: "Mobile hero: …; Desktop hero: …".
- Session-funnel context from `site_funnel`.

**4. Outside Signal (Phase 2b)**
- Alert cards for flagged categories, each with promote-ahead products.
- Small multiples per category: Google interest and on-site views, both indexed to baseline = 100 on one axis (no dual axis).
- Table: M_ext, M_site, gap, lead/lag.
- Provenance: method, geo, fetch time.
- If no data, show fetch instructions; never fabricate.

**5. Cannibalization Check (Phase 2a)**
- Product A selector (default: top pick); spike ratio; signal (views share or promo impressions).
- A's share line with spike-day markers.
- One-sentence summary.
- Dot plot of relative risk with intervals (log x, reference line at 1), using status colours plus icon and label.
- Table: relative risk, dip %, p, q, Fisher p, status.
- Category roll-up of flagged pairs.
- Method and correlation-is-not-causation note.

**6. Slot Plan (the decision)**
- Controls: slots per campaign (default 4 each); avoid cannibalizing pairs (on); allow repeats (off).
- Three columns (Black Friday, Cyber Monday, December gifting), each with:
  - slot cards: product, score per 1,000 views, comparison with average, device-lean badge, trend badge, warning;
  - a "Retire from homepage" list.
- CSV and Markdown downloads (`on_click="ignore"`) and an optional AI verdict for the plan.

**7. Data & Method**
- Source facts: 4,295,584 events, 270,154 users, 92 days, computed from the CSV and from `bq_proof.json`, with a match check.
- The proof SQL and its job metadata, plus a "Re-run live" button that only works locally with credentials (1 GiB cap).
- Grain and row counts; cleaning steps with counts from the profile (placeholder %, duplicates, missing transaction IDs, product-key and view-metric decisions).
- Category mapping table (raw value to group, with views).
- Metric formulas (`st.latex`).
- Season windows and why; tracking-gap chart.
- Limits: the pitch's 3, plus tracking gaps, obfuscation and Trends normalization.
- SQL expanders, manifest table, and the competitive-framing note.

## 7. AI narrative (`ga4app/narrative.py`, `ga4app/llm.py`, `ui.verdict_block`)

**Payload** (compact JSON, rounded numbers)
- `prompt_version` and `scope`: date start/end, window label, devices, categories, min views, eligible count, average conversion, views threshold and rule.
- `winners`, `hidden_gems`, `wasted_exposure`: top 5 each. Each entry has code, name, category, views, purchases, conversion, adjusted conversion, comparison with average, revenue, device lean, and either `lift_per_1k` or `forgone`.
- `holiday_shift`: each campaign window vs the pre-holiday baseline, site-wide and per category, significant ones only (`data.holiday_shift`).
- `device`: rate per device.
- `outside_signal`: flagged categories only, each labelled with its base and focus window (`data.campaign_alerts`: the Trends window of each campaign).
- `cannibalization`: flagged pairs among the picks.
- `slot_plan` (Slot plan page only): per campaign, its window, days and slots (name, code when coded, `per_1k_views`, `x_window_avg`). The prompt then asks for a verdict on the plan, and the rule-based verdict summarises each campaign's lead slot.
- Both pages get `holiday_shift` and `outside_signal` through `data.verdict_context`.
- `caveats`: no placement data (candidates, not causation); no inventory or margin data; add-to-cart not tracked before Nov 26.

**System prompt outline**
1. Role: merchandising analyst briefing the Head of Merchandising on limited holiday homepage slots.
2. Grounding: use only the JSON. Never invent products, numbers, causes, stock or margin. Follow every product name with its code in brackets, using only codes from the JSON.
3. Definitions: conversion = purchases ÷ product views. Percentages with 1–2 decimals.
4. Logic:
   - Feature = Hidden gems and Winners with the highest adjusted conversion.
   - Retire = Wasted exposure with the most forgone purchases.
   - Mention device lean only when it is mobile-first or desktop-first.
   - Mention trend or cannibalization flags only if present.
5. Format:
   - First line: `**Verdict:** Feature A [G1], B [W2], C [G3]. Retire X [X1] from the homepage — it drives views but almost no purchases.`
   - Then 3–5 "why" bullets with numbers, then a one-line "Watch-outs".
   - At most about 170 words; no headings, tables, LaTeX or emoji.
6. If fewer than 3 candidates qualify, say the evidence is thin and suggest lowering min views.

**Request body**
- `{"model": m, "models": fallbacks, "messages": [...], "stream": true, "max_tokens": 4000, "reasoning": {"effort": "low", "exclude": true}}`.
- Never send temperature, top_p or top_k.
- Headers: `Authorization: Bearer`, `HTTP-Referer`, `X-OpenRouter-Title: "Team 9 – Which Products Earn the Homepage?"`.
- Default fallbacks for paid models: `["anthropic/claude-sonnet-5.5"]`; none for free models.

**Streaming client (`stream_chat`, httpx)**
1. Open the stream; on 429/502/503, retry once (honouring Retry-After up to 10 s).
2. On a non-200 status, read the JSON error and raise `OpenRouterError(status, code, message)`.
3. While reading lines: skip blank lines and `:` comments; stop on `data: [DONE]`.
4. Parse each data line as JSON:
   - a top-level `error` yields an error event and stops;
   - otherwise yield the text from `choices[0].delta.content`, plus model, usage and finish reason.
5. Closing the generator closes the connection, which cancels billing on supported providers.

**Rendering and caching**
- `verdict_stream()` wraps `stream_chat`, escapes `$`, and fills a sink with the full text, model and usage.
- On any error it yields an "AI narrator unavailable (reason)" notice followed by the rule-based text, and does not cache.
- `st.write_stream(verdict_stream(...))` runs inside a placeholder that already shows the rule-based verdict.
- After it finishes: `unknown_citations()`. Any unknown code produces a warning and the rule-based verdict is shown instead.
- Cache key: `sha256(PROMPT_VERSION + model + canonical JSON(payload))`, stored in `st.session_state["ai_cache"]` and in a shared `st.cache_resource` dict (outputs aren't sensitive).
- Key resolution order: session input, then `st.secrets` (catching KeyError and FileNotFoundError), then the `OPENROUTER_API_KEY` environment variable. The key is never displayed or used as a cache argument.
- `GA4APP_LLM_FAKE=1` swaps in a canned `MockTransport` for AppTest.

**Rule-based fallback** (deterministic template from the same payload)
- Verdict line: features G1, G2, W1; retire X1 "drew {views} views but converted at {cr} vs the {avg} average (≈{forgone} purchases forgone)".
- Bullets: lift per 1k views for gems; why winners stay; device lean; trend flag; cannibalization flag.
- Caveat line.

## 7b. AI chart notes (`ga4app/annotate.py`, `charts._draw_notes`, `ui.chart_notes_*`)

Built from [`PLAN_ai_chart_notes.md`](PLAN_ai_chart_notes.md). Numbered notes on the Feature Picks scatter, the Holiday Shift bars and the Device Split heatmap. The split of work is the safety rule:

- **The model (or the rule-based template) chooses what to point at and what to say**, from the anchors the app offers.
- **The chart function decides where the note goes**, because it owns the coordinates.
- **`validate_notes` checks what the note says**: the anchor must exist, and every number must come from that anchor's data or the chart's scope.

**Anchors** (`annotate.AnchorSet`: `chart`, `scope`, `anchors` = `[{"id", **facts}]`, `empty` caption, and `keys` = anchor → product key for the page, never sent)

| Chart | Builder | Anchor ID | Facts | Never offered |
|---|---|---|---|---|
| Scatter | `scatter_anchors(payload, products, conflict_keys)` | product code `G1` | the verdict payload entry (name, category, views, purchases, `conversion_pct`, `adjusted_conversion_pct`, 80% `interval_pct`, `vs_avg`, `confidence`, `lift_per_1k` or `forgone_purchases`, `views_undercount`, `device_lean`) + `quadrant` + `cannibal_pair` | uncoded and ineligible products; at most 15 |
| Bars | `holiday_anchors(groups, filters=...)` | `Apparel·cyber_week` (`charts`/`annotate` share `config.NOTE_ANCHOR_SEP`) | category, window, `conversion_pct`, `base_conversion_pct`, 95% `interval_pct`, `lift_pct`, `p_value`, `significant`, `lift_vs_rest_pct`, `p_value_vs_rest` | base-window bars; `low_sample` bars |
| Heatmap | `heatmap_anchors(charts.heatmap_cells(matrix, codes=...), lean, scope=...)` | `G1·mobile`, or `R3·mobile` (3rd row, 1-based) for an uncoded product | name, device, `conversion_pct` (the cell text), `x_usual` (the colour's multiple, 2 decimals), `vs_usual_pct` (the same as a percent difference), views, purchases, `other_devices_pct` (the product's other readable cells), `lean` | sparse tablet cells; cells with abs(log2(rel)) < 0.25 unless the product leans strongly; at most 30, strong-lean products' cells first, then the strongest |

- Facts use `narrative`'s rounding (percent with 2 decimals, counts as integers, multiples with 2 decimals), so the number check compares like with like.
- Scope: scatter = window, filters, average conversion, views threshold, `interval_level_pct` 80; bars = base and focus labels and dates, filters, `significance_level` 0.05, `interval_level_pct` 95; heatmap = window, filters, candidate list.
- `charts.heatmap_cells` is the per-cell computation that used to live inside `device_heatmap` (row order, `rel`, `z`, `sparse`, plus the anchor ID). The chart and the anchors both read it, so a note quotes exactly what the cell shows. The heatmap figure is byte-identical to before.
- **Anchors depend only on data controls** (sidebar filters, the high-traffic cutoff, the Holiday Shift comparison, the Device Split candidate list). Log/Linear, bubble size and colour are chart arguments only. **Raw/Adjusted** is a data control in this app: `metrics.classify` puts products in quadrants by the plotted metric, so it could re-code products. On the real and synthetic snapshots it gives the same coded products, hence the same anchors and cache key, in every scope tried (full season, min views 0, Cyber Week). When it does re-code products, the key changes and the notes are rewritten, which is right: a note must point at the bubble that carries its code.

**The model call** (`ai_notes`)
- Own prompt, `NOTES_PROMPT_VERSION = "notes-2026-11-18.2"` (tuned in the live run below): the verdict's analyst role; at most 4 anchors; each note has `anchor` (copied character for character), `kind` (`highlight` / `caution`), a `label` of at most 5 words and 30 characters saying why it matters, and a `detail` of 25 words or fewer that starts with the product (name and code) or category; plain English with no field names; copy numbers exactly, never compute new ones, keep multiples as multiples; never call anything the highest, lowest, most, least, best or worst (the model sees only the anchors); at least one caution when an anchor carries `views_undercount`, `cannibal_pair` or `confidence: "uncertain"`; reply with `{"notes": [...]}` only. Definitions cover the fields of all three charts.
- User message: `Annotate this chart. JSON:` + `{"chart", "question", "scope", "anchors"}`.
- Transport: `llm.stream_chat` unchanged (same model, fallbacks and key as the verdict), text events joined, no `response_format`. Any `OpenRouterError`, network error or empty reply gives the rule-based notes with `error` = the reason.
- A reply with no text because the model spent its whole token budget reasoning (`finish_reason: "length"`, seen with free models) gives the error "the model used up its token budget before answering; try again or choose another model".
- Offline: `llm.fake_transport(respond=...)` takes a function from the request body to text chunks (default: the verdict's `FAKE_TEXT_CHUNKS`); `annotate.fake_respond` notes the first two anchors without numbers, so it passes on any data. `GA4APP_LLM_FAKE=1` selects it.

**Validation** (`validate_notes`; rule-based notes go through it too)
1. Parse: strip code fences, decode the first JSON value at a `{` or `[` (`{"notes": [...]}` or a bare list). Unreadable: rule-based notes, error "AI notes could not be read".
2. Shape: `anchor`, `label`, `detail` must be non-empty strings; an unknown `kind` becomes `highlight`.
3. Anchor: compared after Unicode normalisation (NFKC) and whitespace collapsing, so an ID typed back with a non-breaking space still matches; unknown → dropped, duplicate → dropped, then the first `NOTES_MAX = 4` are kept.
4. Number check (`unmatched_numbers`): product codes (`[A-Z]\d{1,3}`) and the anchor ID are removed; every number (`\d+(?:,\d{3})*(?:\.\d+)?`) written with d decimals must satisfy abs(x − v) ≤ 0.5·10^−d for some v among this anchor's facts and the scope, including numbers inside strings (dates, "min 200 views"). Integers 1–5 pass as counting words; signs are compared by absolute value; "per 1,000" is read as the unit of `lift_per_1k`.
5. Length: label to 40 and detail to 200 characters, cut at a word so a number is never cut in half, ending with "...".
6. Stored unescaped; `charts._esc` escapes for Plotly, `narrative.md_safe` plus markdown escaping for the list.
7. No survivors (or an empty list): rule-based notes, error "no AI note passed the checks", with `dropped = [(anchor, reason)]`.

**Rule-based notes** (`rule_based_notes`, deterministic)
- Scatter: G1 "Best hidden gem", W1 "Top winner", X1 "Most purchases forgone" (caution), and a caution "Views may be under-counted" for the first featured pick (the verdict's Feature order) whose views look under-counted. When that pick is G1 or W1, its own note becomes the caution instead of a duplicate.
- Bars: the largest significant rise (highlight) and drop (caution), "significant at p < 0.05". The lift is written rounded like the bar's own label (+108%). Nothing significant: no notes and the caption "No significant shift to point out."
- Heatmap: the favoured-device cell of up to 2 mobile-first / desktop-first products ("Mobile-first hero"), then the strongest remaining cell of another product ("Strong on mobile", or a caution "Weak on desktop").

**Placement** (`ChartNote(anchor, kind, label, detail, n)` and `NotesResult` in `types.py`)
- `_draw_notes`: annotation `note:<source>`, text `<b>n</b> label` (cautions: `<b>n</b> <b>!</b> label`, coral border), detail on hover, white box, ink text.
- Scatter: at the drawn point (`log10(views)`, `log10(y_plot)` on a log y axis), so a clipped bubble's arrow goes to the edge; the box sits 40 px toward the centre horizontally and 30 px vertically (below in the top third), and grows toward the interior (`xanchor` right of a left box, left of a right box), so no box hangs over an edge; the arrow stops at the bubble's radius.
- Bars: at the focus bar's `x[j]` and whisker top, 40 px above; the arrow stops above the lift label; the y headroom grows from 1.18 to 1.35 when a note is drawn.
- Overlap (`_spread_notes`, scatter and bars): in note order, a box that would overlap an earlier one moves to the other side of its point (scatter only), then one box height (24 px) further out, staying inside the plot. The plot's pixel width follows the page, so it is guessed (700 px, a box up to 32% of it); a wrong guess only stacks boxes that would not have touched. Code labels (W1, G2) are not obstacles, so a box can still cover one.
- Heatmap: a rectangle around the cell and a numbered badge in its top-left corner; the label is in the list only.
- With no notes drawn, every figure is unchanged (tests compare the JSON).

**Page UI**
- `ui.chart_notes_controls(anchors, rule_notes=None, *, key_prefix)` above the chart: the "Point out what matters" toggle (`{P}notes_on`, off), "Write AI notes" (disabled without a model or anchors; it also turns the toggle on) and the cost note. The click runs `ai_notes` under a spinner, not streamed. Results whose source is `"ai"` are cached by `cache_key` = sha256(prompt version, model, chart, canonical JSON of scope + anchors) in `st.session_state["notes_cache"]` and a shared `st.cache_resource` dict. Without a click: the cached AI notes for these anchors, else the rule-based ones.
- `ui.chart_notes_list(result, anchors, *, key_prefix, inspect=False)` under it: the source caption ("AI notes · model · every number checked against the data" or "Rule-based notes from the numbers on this chart"), a warning with the dropped notes or the failure, then "**n** label: detail". Feature Picks adds an Inspect button per product note.
- `ui.chart_with_table(..., alt_extra=ui.notes_alt(notes))` puts the note sentences into the chart's alt text only.

**Live run (plan step 5), 2026-11-17.** Real snapshot; six anchor sets per model (scatter: full season and Cyber Week; heatmap: the same two scopes, Winners + Hidden gems, top 20; bars: baseline vs Cyber Week and vs December), each called exactly as `ai_notes` does, plus one verdict per model. Prompt versions: v1 = `notes-2026-11-17.1`, v2 = `notes-2026-11-18.1`, v3 = `notes-2026-11-18.2` (the later IDs carry the UTC date; the run was on the evening of Nov 17, US Eastern).

| Model | Prompt | Calls | Answered | Notes kept / dropped | Latency | Cost per call |
|---|---|---|---|---|---|---|
| `anthropic/claude-opus-5.5` | v2 | 1 (scatter, full season) | 1 | 4 / 0 | 5.2 s (first token 2.4 s) | \$0.0214 (3,509 prompt tokens \$0.0140 + 366 output \$0.0073; no reasoning tokens) |
| `anthropic/claude-haiku-5.5` | v1 | 6 | 6 | 23 / 1 | 4.1–6.2 s | \$0.0006–0.0010 |
| `anthropic/claude-haiku-5.5` | v2 | 6 | 6 | 23 / 1 | 4.2–7.0 s | \$0.0008–0.0010 |
| `anthropic/claude-haiku-5.5` | v3 | 6 | 6 | **24 / 0** | 5.0–7.4 s | \$0.0008–0.0012 |
| `openrouter/free` (a different free model each call) | v1 | 6 | 6 | 16 / 7 | 9–54 s | \$0 |
| `openrouter/free` | v2 | 6 | 4 (2 used up the token budget) | 16 / 0 | 21–43 s | \$0 |
| `openrouter/free` | v3 | 6 | 3 (3 used up the token budget) | 12 / 0 | 14–23 s | \$0 |
| `google/gemma-4-31b-it:free` | v1–v3 | 13 | 0 | – | – | HTTP 429 every time: "temporarily rate-limited upstream" (Google AI Studio's shared free pool) |

Verdict, same session: Haiku 5.5 \$0.0011–0.0012 and 6.5 s (4,030 prompt tokens), free models 8–15 s; no unknown product codes from either. Opus was not run for the verdict (see the balance note below); at Opus prices and the token counts above, a verdict should cost about \$0.02–0.03, inside `ui.COST_NOTE`'s \$0.02–0.05.

**Why notes were dropped, and what changed** (v1 → v2 → v3):
- Heatmap notes compared a cell with the product's other device ("9.82% on desktop vs 2.15% on mobile": 2.15 came from another anchor) or turned `x_usual` 1.24 into "24% above its usual rate". Both are true, so cells now carry `other_devices_pct` and `vs_usual_pct`.
- An anchor came back as `Lifestyle &\u00a0Gifts·cyber_week` (a non-breaking space): IDs now match after Unicode normalisation.
- "57.2 extra purchases per 1,000 views" (1,000 is the unit of `lift_per_1k`) and "its 95% range" (the interval's level) were false positives: "per 1,000" is now a unit and each scope carries `interval_level_pct`.
- Quality, not caught by the check: field names in the text ("adjusted_conversion_pct 6.51"), `lift_vs_rest_pct` read as "above the store average", holiday notes that never named their category, labels cut at 40 characters, and rank claims the model cannot know ("lowest mobile rate shown", false: it saw only the offered cells). The prompt now covers each; v3 Haiku notes all start with their product or category and no label was cut.
- Free models: 5 of the 12 v2–v3 calls spent all 4,000 tokens on reasoning and returned nothing; the app then shows the rule-based notes with a clear message. Every free-model note that did come back in v2–v3 passed the checks.

**Seen in the running app** (Haiku 5.5, `streamlit run` with the key, "Write AI notes" clicked in a browser on all three pages): notes drawn as `note:ai`, no warnings. Two neighbouring hidden gems had overlapping boxes and a box near the right edge covered the "Average" label, which led to `_spread_notes` and the inward-growing boxes above.

**Limits that remain:**
- The number check cannot tell whether a correct number is used correctly; a model can still misread a fact (e.g. call a 35% lift-beyond-the-rest "35 points").
- **OpenRouter reserves `max_tokens` × price before a call.** The app asks for 4,000 tokens (free reasoning models need them), so an Opus call needs about \$0.09 of available credit even though it costs about \$0.02. With less, every Opus or Sonnet call returns "out of OpenRouter credits, HTTP 402" and the page keeps the rule-based notes. The key used here had about \$0.035, enough for one Opus call with a lower cap, which is why Opus has one data point and the other paid figures are from Haiku.

## 8. Tests and end-to-end verification

**Unit tests** (`tests/`; synthetic data from a session-scoped fixture using `make_synthetic_snapshots(tmp, seed=7)`)
- **`test_contracts`**
  - Synthetic snapshots validate.
  - A missing column or wrong dtype raises.
  - The literal placeholder list in every SQL file equals `config.PLACEHOLDERS`.
- **`test_taxonomy`**, on real strings:
  - `Home/Apparel/Men's / Unisex/` maps to Apparel and keeps the segment "Men's / Unisex".
  - `Lifestyle/Drinkware/` maps to Drinkware.
  - `Home/Sale/` + "Google Tee Green" maps to Apparel.
  - `Home/Shop by Brand/Android/` + "Android Small Trace Journal Black" maps to Office & Stationery.
  - `Lifestyle/Bags/` + "Laptop Sleeve" maps to Bags.
  - Double spaces in names collapse in `name_key`.
- **`test_metrics`**
  - Pooled rate; Wilson interval (k=5, n=100 gives about [0.0215, 0.1118]).
  - Method of moments recovers the planted m within 30%.
  - Shrinkage is monotone; quadrant assignment on hand-made points; forgone and lift formulas.
  - Filter edges (inclusive dates); summing over devices equals the all-device total.
  - The P > V clip.
  - Planted hidden gems and wasted products land in the right quadrants; the planted mobile product is called Mobile-first.
  - Campaign score borrows from the season rate.
  - Slot plan skips conflicting pairs.
- **`test_phase2`**
  - The planted cannibal pair has relative risk below 0.8 and q below 0.1.
  - Null simulations (200 runs) have a false-positive rate at or below about alpha.
  - Spike detection; insufficient-data guards.
  - BH results match `false_discovery_control`.
  - The planted trending group is flagged; a non-trending one isn't.
- **`test_trends`**
  - The manual multiTimeline CSV parser handles the preamble, "<1" values, and Day vs Week headers.
  - Weekly aggregation; provenance fields are filled.
- **`test_llm`** (MockTransport)
  - The stream with comments, content, a usage-only final chunk and `[DONE]` gives the right text, model and usage.
  - A mid-stream `{"error":...}` gives an error event.
  - 401, 402 and 429 raise `OpenRouterError` with the status.
  - The request body has `models`, `reasoning` and `stream`, and no temperature; the headers are present.
- **`test_narrative`**
  - The rule-based verdict is deterministic and cites only payload codes.
  - `unknown_citations`; `$` escaping; the cache key ignores dict ordering.
  - The fallback path works when there is no key.
- **`test_charts`**
  - Scatter JSON: log x axis; 2 dashed threshold shapes; 4 quadrant badges in paper coordinates; `customdata` holds the product key; the colour map follows `manifest.category_colors`.
  - Heatmap z is clipped.
  - Nothing uses `use_container_width`.
- **`test_bq`** (mocked client)
  - The dry run gates the real run.
  - The effective cap respects the shard floor.
  - The arrow-to-parquet write is atomic; the export log is written.
- **`test_site_aggregates`**: DuckDB over `fixtures/mini_events.csv` (about 60 rows) gives the expected counts. Distinct users count correctly when IDs look like floats.
- **`test_build_snapshots`**: tiny raw parquet built in the test; both product-key strategies; view-metric selection; the manifest is written.

**AppTest** (`tests/test_app_pages.py`, `@pytest.mark.app`, `default_timeout=60`)
- `GA4APP_DATA_DIR` points at the synthetic directory; `OPENROUTER_API_KEY` is unset.
- For each of the 7 page files: `run()`, then `switch_page("views/x.py").run()`, then assert `not at.exception`.
- Feature Picks: the scatter spec assertions; at least 3 metrics; a dataframe is present; "**Verdict:**" text appears.
- Setting the devices pills to `["mobile"]` and rerunning gives no exception and changed metrics.
- Setting `session_state["inspect_key"]` makes the inspector show that name.
- With `GA4APP_LLM_FAKE=1` and `at.secrets[...]`, clicking the AI button and rerunning produces the fake text and fills `ai_cache`.
- Holiday Shift and Device Split show the disabled filter.
- Outside Signal shows the "no data" state when the trends file is removed.

**End-to-end, in order**
1. Install the three requirements files into the ITCS5122 env, run `pip check`, then `pytest -q`.
2. Run `python scripts/bq_auth.py --project <ID>`. It opens a browser for sign-in (run it in the user's own terminal or in the background) and must print 4295584 / 270154 / 92; it writes `bq_proof.json`.
3. Run `profile_bigquery.py`, review `profile.json` decisions (section 9), then set the `config.py` defaults.
4. Run `export_bigquery.py --dry-run-only`, then the real export.
5. Run `build_site_aggregates.py --csv ga4_events_export.csv`. `overview.json` must match the bq proof. Site-level purchase/view_item must be 1.18% (baseline), 1.85% (Cyber Week), 1.70% (December), 1.47% (full).
6. Run `fetch_trends.py`, or drop manual CSVs into `data/trends_manual/`.
7. Run `build_snapshots.py`.
8. Run `verify_snapshots.py`. Hard checks:
   - 92 dates;
   - devices limited to {desktop, mobile, tablet};
   - all invariants hold;
   - product_dim covers every product key;
   - `profile.purchases.purchase_events == site_daily purchase events (5,692)`;
   - total view_rows ÷ view_item events equals the profiled average items per event (±0.5%).

   Warnings:
   - "Other" above 15% of views;
   - revenue total far from about $308k;
   - top-revenue items not including the Zip Hoodie or Crewneck Sweatshirt.
9. Run `streamlit run app.py` and click through every page:
   - click a scatter point;
   - apply each preset;
   - copy the URL and reopen it;
   - generate the verdict with and without a key, and with `openrouter/free`.
10. Rerun AppTest with `GA4APP_DATA_DIR=data`.

## 9. Risks and unknowns, with decision rules after profiling

| # | Unknown | Decision rule |
|---|---|---|
| R1 | Product key | Use `item_id` if purchase-row coverage by id is at least 97% and names with multiple ids cover under 3% of rows. Otherwise use `name_key`. Record in `profile.decisions`. |
| R2 | View metric | If average items per view_item ≤ 1.05, use `view_events`. If multi-item events are at least 80% same-name, use `view_events`. If they are mixed products and the first item matches page_title in at least 80% of events, use `primary_views`. Otherwise use `view_events` and caveat it. |
| R3 | Category mapping | Path rules, then name keywords, then "Other". If "Other" exceeds 15% of views, extend the name rules from the top unmapped items. Colours go to the top 3 groups by full-season views and are frozen in the manifest. |
| R4 | Promotions | If at least 50% of view_promotion item rows carry ids that exist in view_item, the featured signal is promo impressions (soften limit #1). Otherwise use the views-share spike. |
| R5 | Purchases | Duplicates by transaction id are deduped. Missing-id purchases are kept for conversion and excluded from revenue only if revenue is null or zero. Show sensitivity (raw vs deduped) on Data & Method. |
| R6 | Tracking gaps | Auto-detect days below 10% of an event's median; hard-code `CART_TRACKING_START = 2020-11-26`. Cart metrics appear only in the inspector, with a caveat. |
| R7 | Devices | Values outside the three main ones go to "other"; drop them if under 0.1%. |
| R8 | Trends | If a series has more than 30% zero days, use a broader keyword. If every method fails, set status "unavailable" and disable the page features. If trendspy breaks under pandas 3, run `fetch_trends.py` in a throwaway venv (it only writes files). |
| R9 | Shrinkage m | Clamp to [100, 20000]; log and display it. If a category has under 20 products, use the global m. |
| R10 | Spike sparsity | If fewer than 30% of top-K products have 3 or more spike days at ratio 2.0, default to 1.5 and record it. |
| R11 | Windows | Recheck daily purchases in BigQuery; windows are config constants. |
| R12 | OAuth or quota | Fall back to gcloud ADC, then a service account, then console CSV export with `build_snapshots.py --from-csv`. Enable the BigQuery API if prompted. |
| R13 | LLM | Invalid model IDs give a 400 with the message shown plus fallback. Free models are limited to 50 requests/day. Opus rejects sampling params. Hallucinated codes trigger the rule-based verdict. |
| R14 | Environment | pandas date32 objects, `str` dtype, OneDrive locks and file-watcher churn, and the fact that the 450 MB CSV is not deployed. |
| R15 | Obfuscation | Interpret relative patterns only; say so on every relevant page. |

## 10. Parallel implementation units and interfaces

**Unit 0 (blocking, about 1 hour)**
- Files: requirements files, pyproject, .gitignore, `.streamlit/*`, `config.py`, `contracts.py`, `synthetic.py`, `scripts/make_synthetic_data.py`, `tests/conftest.py`, `test_contracts.py`.

**After Unit 0, all disjoint and parallel**

| Unit | Files owned | Depends on |
|---|---|---|
| U1 BigQuery | `bq.py`, `sql/*`, `scripts/bq_auth.py`, `profile_bigquery.py`, `export_bigquery.py`, `test_bq.py` | U0 |
| U2 Local build | `taxonomy.py`, `scripts/build_site_aggregates.py`, `build_snapshots.py`, `verify_snapshots.py`, tests + fixture CSV | U0 |
| U3 Trends | `trends.py`, `scripts/fetch_trends.py`, `config/trends_keywords.csv`, `test_trends.py` | U0 |
| U4 Metrics | `metrics.py`, `test_metrics.py` | U0 |
| U5 Phase 2 | `phase2.py`, `test_phase2.py` | U0 (uses `metrics.Filters`/`Period` types only) |
| U6 Charts | `charts.py`, `test_charts.py` | U0 + the frame columns below |
| U7 AI | `llm.py`, `narrative.py`, `test_llm.py`, `test_narrative.py` | U0 |
| U8 Shell + decision pages | `app.py`, `data.py`, `state.py`, `ui.py`, `views/feature_picks.py`, `views/slot_plan.py` | U4–U7 interfaces (can stub) |
| U9 Remaining pages | `views/holiday_shift.py`, `device_split.py`, `outside_signal.py`, `cannibalization.py`, `data_method.py`, `test_app_pages.py`, README | U8 `ui`/`state` |

Then a final sequential step: run the real-data pipeline with the user present for the sign-in.

**Interfaces**

```python
# config.py
DATA_DIR_ENV="GA4APP_DATA_DIR"; LLM_FAKE_ENV="GA4APP_LLM_FAKE"; DEVICES=("desktop","mobile","tablet")
PLACEHOLDERS=("","(not set)","<Other>","(data deleted)","Not available in demo dataset")
@dataclass(frozen=True) class Period: key:str; label:str; start:date; end:date
PERIODS: dict[str, Period]   # full, baseline(11-01..11-22), cyber_week(11-23..11-30), black_friday(11-26..11-29),
                             # cyber_monday(11-30..12-01), december(12-02..12-18), post_cutoff(12-19..12-31), january
CAMPAIGNS={"Black Friday":"black_friday","Cyber Monday":"cyber_monday","December gifting":"december"}
CART_TRACKING_START=date(2020,11,26); MIN_VIEWS_OPTIONS=(...); DEFAULT_MIN_VIEWS=200; SHRINK_M_BOUNDS=(100,20000)
CATEGORY_GROUPS=("Apparel","Drinkware","Bags","Office & Stationery","Electronics & Accessories","Lifestyle & Gifts","Other")
PALETTE={...}; QUADRANT_STYLE={...}; PERIOD_COLORS=("#9AA6D9","#6270BD","#2F3B7D")
LLM_MODELS=(...); DEFAULT_MODEL="anthropic/claude-opus-5.5"; DEFAULT_FALLBACKS=("anthropic/claude-sonnet-5.5",)

# contracts.py
SCHEMAS: dict[str, dict[str, str]]
class ContractError(ValueError): ...
def validate_frame(df: pd.DataFrame, name: str) -> None
def write_parquet_atomic(obj: pd.DataFrame | pa.Table, path: Path) -> str  # sha256
def write_json_atomic(obj: dict, path: Path) -> None

# synthetic.py
def make_synthetic_snapshots(out_dir: Path, *, n_products: int = 240, seed: int = 7) -> dict  # planted truth

# data.py
@dataclass(frozen=True) class Snapshot: item_daily; product_dim; site_daily; site_funnel|None; promo_daily|None;
                                         trends_daily|None; overview: dict; profile: dict; manifest: dict; version: str;
                                         item_daily_totals|None   # per day, before the ITEM_DATA_END cut
def resolve_data_dir() -> Path
def split_item_daily(item_daily) -> tuple[pd.DataFrame, pd.DataFrame]  # (rows <= ITEM_DATA_END, per-day totals)
def load_snapshot(data_dir: Path) -> Snapshot          # pure; validates; converts dtypes; drops rows after the cut
def product_devices(snapshot, key: str, f: Filters) -> pd.DataFrame  # one product's device_matrix rows
def get_snapshot() -> Snapshot                          # st.cache_resource(data_dir, manifest sha)

# taxonomy.py
def parse_category_path(raw: str | None) -> tuple[str, ...]
def category_group(raw_category: str | None, item_name: str | None) -> str
def brand_of(item_name: str | None) -> str
def name_key(item_name: str | None) -> str | None

# metrics.py
@dataclass(frozen=True) class Filters: start: date; end: date; devices: tuple[str,...]; categories: tuple[str,...]; min_views: int
@dataclass(frozen=True) class Thresholds: views: float; conversion: float; views_rule: str
def filter_daily(item_daily, product_dim, f: Filters, *, ignore_dates=False, ignore_devices=False) -> pd.DataFrame
def product_table(daily_f, product_dim) -> pd.DataFrame
def estimate_prior_strength(products_full, *, min_views=100, bounds=SHRINK_M_BOUNDS) -> float
def shrink(products, m: float, *, interval=0.8) -> pd.DataFrame
def thresholds(products, min_views: int, views_rule: str | float = "median") -> Thresholds
def classify(products, th: Thresholds, *, y: str = "cr_raw") -> pd.DataFrame
def score_actions(products, th: Thresholds, *, top_n_codes: int = 5) -> pd.DataFrame
def wilson_ci(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]
def period_comparison(item_daily, product_dim, periods: Sequence[Period], f: Filters, *, by="category_group") -> pd.DataFrame
def quadrant_movers(item_daily, product_dim, base: Period, focus: Period, f: Filters, m: float) -> pd.DataFrame
def device_matrix(item_daily, product_dim, keys: Sequence[str], f: Filters, m: float) -> pd.DataFrame
def device_lean(matrix, *, rr_hi=1.25, rr_lo=0.8, alpha=0.10, min_purchases=10) -> pd.DataFrame
def campaign_scores(item_daily, product_dim, campaign: Period, f: Filters, m: float) -> pd.DataFrame
def slot_plan(scores: Mapping[str, pd.DataFrame], n_slots: Mapping[str, int], *, conflicts=frozenset(),
              exclude=frozenset(), must_include=(), allow_repeats=False) -> pd.DataFrame  # attrs["unplaced"]
# products frame columns: product_key,item_name,category_group,brand,views,purchases,units,revenue_usd,promo_views,
#   cr_raw,prior_mean,cr_adj,ci_low,ci_high,confidence,eligible,quadrant,forgone_purchases,lift_per_1k,rpv,action_rank,code,
#   views_share,purchases_exceed_views

# phase2.py
def exposure_spikes(item_daily, product_dim, f, *, signal="views", ratio=2.0, min_daily=30) -> pd.DataFrame
def cannibalization_tests(item_daily, product_dim, spikes, f, *, top_k=12, min_spike_days=3, min_other_days=10,
                          min_purchases=10, min_expected=3.0, rr_flag=0.8, q_flag=0.10) -> pd.DataFrame
def conflict_pairs(tests) -> set[frozenset[str]]
def trend_index(trends_daily, item_daily, product_dim, keyword_map, base: Period, f, *, freq="W-SUN") -> pd.DataFrame
def trending_alerts(trends_daily, item_daily, product_dim, keyword_map, base, focus, f, products_scored, *,
                    ext_min=1.2, gap_min=1.15, min_base_views=1000) -> pd.DataFrame
def lead_lag(index_df, max_lag: int = 3) -> pd.DataFrame

# charts.py  (all return plotly.graph_objects.Figure)
def quadrant_scatter(products, th, *, y="cr_raw", size="revenue_usd", color_map, highlight=None, y_scale="linear")
def period_bars(period_df); def daily_conversion(daily, periods); def transition_matrix(movers)
def device_heatmap(matrix, lean); def trends_small_multiples(index_df, periods); def spike_share(spikes, key, periods)
def rr_dotplot(tests); def product_timeline(daily_one, periods)
def slot_candidates(scores, picked=(), *, top=12); def device_bars(matrix_one); def short_label(key) -> str

# llm.py
@dataclass class StreamEvent: kind: Literal["text","meta","error","done"]; text: str = ""; model: str | None = None;
                              usage: dict | None = None; finish_reason: str | None = None; status: int | None = None; code: str | None = None
class OpenRouterError(RuntimeError): status: int | None; code: str | None
def resolve_api_key(session_value: str | None = None) -> tuple[str | None, str]
def build_request(messages, *, model, fallbacks=(), max_tokens=4000, reasoning_effort="low") -> dict
def stream_chat(messages, *, api_key, model, fallbacks=(), max_tokens=4000, reasoning_effort="low",
                transport: httpx.BaseTransport | None = None, timeout_s: float = 90) -> Iterator[StreamEvent]

# narrative.py
PROMPT_VERSION: str; SYSTEM_PROMPT: str; DEFAULT_CAVEATS: tuple[str, ...]
def build_payload(products, th, f, *, window_label, period_shift=None, device_lean=None, alerts=None, cannibal=None, top_n=5, device_summary=None, slot_plan=None) -> dict  # slot_plan may carry views_w / thin_window; products may carry views_undercount
def cache_key(payload: dict, model: str) -> str
def rule_based_verdict(payload: dict) -> str
def unknown_citations(text: str, payload: dict) -> list[str]
def md_safe(text: str) -> str
def verdict_stream(payload, *, api_key, model, fallbacks=(), sink: dict, transport=None) -> Iterator[str]

# bq.py
SCOPES=("https://www.googleapis.com/auth/bigquery",)
def get_credentials(*, reauth=False, use_local_webserver=True)   # pydata_google_auth + ReadWriteCredentialsCache(dirname="itcs5122_team9"); falls back to google.auth.default()
def get_client(project: str, credentials=None) -> "bigquery.Client"   # location="US"
def read_sql(name: str) -> str
@dataclass class DryRun: bytes_processed: int; referenced_tables: int
def dry_run(client, sql: str) -> DryRun
def effective_cap(dr: DryRun, cap_bytes: int) -> int
def run_to_arrow(client, sql: str, *, cap_bytes: int, labels: dict) -> tuple["pa.Table", dict]

# state.py / ui.py (Streamlit only)
def render_sidebar(snapshot, page_id: str) -> Filters;  def current_filters() -> Filters
def load_query_params() -> None;  def sync_query_params() -> None;  def query_params_for(f, preset) -> dict
def verdict_block(payload: dict, *, key_prefix: str) -> None;  def data_table(df, *, key, column_config, on_select=None)
def chart_with_table(fig, df, *, key, alt, on_select=None);  def product_inspector(snapshot, f, key: str | None)
```

Widget key prefixes avoid collisions between units:
- Global: `dates`, `preset`, `dev`, `cat`, `minv` (the URL parameters).
- AI settings: `ai_*`.
- Per page: `fp_`, `hs_`, `ds_`, `os_`, `cc_`, `sp_`, `dm_`.
- Shared: `inspect_key`.

## Sources
- [S-UrwaBatoolAbdi/ga4-ecommerce-analysis](https://github.com/S-UrwaBatoolAbdi/ga4-ecommerce-analysis) (README, results/07_product_insights.csv, results/07b_zero_orders_check.csv, sql/07_product_insights_v2.sql)
- [GGzzzW/ga4-ecommerce-analysis](https://github.com/GGzzzW/ga4-ecommerce-analysis) · [xinhaidi021005-sudo/GA4-Ecommerce-Analytics](https://github.com/xinhaidi021005-sudo/GA4-Ecommerce-Analytics) · [Meliorum item SQL](https://www.meliorum.com.au/blog/the-ga4-update-matching-big-query-data-with-google-analytics-product-performance)
- [GA4 BigQuery export schema](https://support.google.com/analytics/answer/7029846) · [GA4 sample dataset page](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset) · [HF gizdevans/ga4-bq-flattened-data](https://huggingface.co/datasets/gizdevans/ga4-bq-flattened-data)
- [pydata-google-auth API](https://pydata-google-auth.readthedocs.io/en/latest/api.html) · [auth.py](https://github.com/pydata/pydata-google-auth/blob/main/pydata_google_auth/auth.py) · [QueryJobConfig](https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.QueryJobConfig) · [RowIterator](https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.RowIterator) · [BigQuery pricing](https://cloud.google.com/bigquery/pricing) · [winget Google.CloudSDK](https://wingetly.io/apps/google/cloud-sdk)
- [trendspy on PyPI](https://pypi.org/project/trendspy/) · [pytrends is dead (2026)](https://apiserpent.com/blog/pytrends-dead-google-trends-data-2026) · [Trends API landscape 2026](https://konabayev.com/blog/google-trends-api-landscape-2026/) · [Official alpha vs alternatives](https://webscraping.cc/blog/google-trends-api/)
- [openai-python _streaming.py](https://raw.githubusercontent.com/openai/openai-python/main/src/openai/_streaming.py) · [openai-python CHANGELOG](https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md) · [releases](https://github.com/openai/openai-python/releases)
- [OpenRouter streaming](https://openrouter.ai/docs/api/reference/streaming) · [OpenRouter API overview](https://openrouter.ai/docs/api/reference/overview)

### Critical Files for Implementation
- C:\Users\abelv\OneDrive\Desktop\School\Fall2026\ITCS5122\Project\sql\export_item_daily.sql
- C:\Users\abelv\OneDrive\Desktop\School\Fall2026\ITCS5122\Project\ga4app\metrics.py
- C:\Users\abelv\OneDrive\Desktop\School\Fall2026\ITCS5122\Project\ga4app\phase2.py
- C:\Users\abelv\OneDrive\Desktop\School\Fall2026\ITCS5122\Project\app.py
- C:\Users\abelv\OneDrive\Desktop\School\Fall2026\ITCS5122\Project\views\feature_picks.py
"""Phase 2 analytics: exposure spikes, cannibalization pair tests and the Google Trends alert.

Phase 2a (cannibalization). The GA4 sample has no product-level placement history:
promotion events carry banner names, not products. So a product's "featured" days are
proxied by short spikes in its share of its category's views, measured against the
product's rolling four-week median share (``exposure_spikes``).

For every ordered pair (A, B) among a category's top-K products, ``cannibalization_tests``
compares B's purchases on A's spike days with what B's same-day category peers predict.
Shares ignore traffic surges, and peer-relative conversion absorbs shocks common to the
whole category on a given day, so the comparison is seasonality-neutral. Pairs get an exact
conditional binomial test, with Benjamini-Hochberg FDR control across all tested pairs.

Phase 2b ("trending outside our walls"). Google search momentum per category or brand is
compared with on-site view growth (``trend_index``, ``trending_alerts``). Each Trends
request is normalised to its own peak, so a keyword is only ever compared with itself.

Pure pandas/numpy/scipy: no Streamlit, and no dependency on ``metrics``. The few pooled
rates needed are computed here, and scored products are passed in.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .config import Period
from .types import Filters, Thresholds

_C = config.CANNIBAL
_T = config.TRENDING
# Centred rolling window (days) for the spike baseline: two weeks either side of the day.
SPIKE_WINDOW_DAYS = 29

# --------------------------------------------------------------------------- output contracts
# Column -> dtype of every frame this module returns. Empty results keep the same columns
# and dtypes, so pages can filter them without special cases.

_SPIKE_SCHEMA: dict[str, str] = {
    "category_group": "str",
    "product_key": "str",
    "event_date": "datetime64[ns]",
    "views": "int64",  # product views that day, in scope
    "category_views": "int64",
    "exposure": "int64",  # the spike signal: views, or promo impressions when signal="promo"
    "category_exposure": "int64",
    "share": "float64",  # exposure / category_exposure
    "median_share": "float64",  # median share over the product's in-scope days with exposure
    "baseline_share": "float64",  # the local (rolling) median share the spike rule compares with
    "share_ratio": "float64",  # share / baseline_share
    "is_spike": "bool",
    "signal": "str",  # the signal actually used: "views" or "promo"
}
_TEST_SCHEMA: dict[str, str] = {
    "category_group": "str",
    "product_a": "str",  # the spiking ("featured") product
    "product_b": "str",  # the same-category peer whose purchases are tested
    "item_name_a": "str",
    "item_name_b": "str",
    "n_spike_days": "int64",
    "n_other_days": "int64",
    "spike_views_b": "int64",  # B's views on A's spike days
    "O_s": "int64",  # B's purchases on A's spike days
    "E_s": "float64",  # ... expected from B's same-day peers (category minus A and B)
    "O_n": "int64",  # B's purchases on A's other days
    "E_n": "float64",
    "n": "int64",  # O_s + O_n
    "pi0": "float64",  # E_s / (E_s + E_n): share of B's purchases expected on spike days
    "rr": "float64",  # (O_s / E_s) / (O_n / E_n); NaN unless tested
    "rr_low": "float64",  # Clopper-Pearson interval mapped to RR
    "rr_high": "float64",
    "dip": "float64",  # 1 - rr (0.6 = B sold 60% less than expected)
    "p_value": "float64",  # one-sided exact binomial test for a dip
    "q_value": "float64",  # Benjamini-Hochberg across all tested pairs
    "fisher_p": "float64",  # secondary "steal" check: B's share of peer purchases drops
    "tested": "bool",
    "status": "str",
    "reason": "str",
}
_INDEX_SCHEMA: dict[str, str] = {
    "group_type": "str",  # "category" or "brand"
    "group": "str",  # category_group or brand, spelled as in product_dim
    "keyword": "str",
    "week": "datetime64[ns]",  # period end label (the Sunday for "W-SUN")
    "n_days": "int64",  # calendar days of on-site data in the week
    "interest": "float64",  # mean daily Google interest (0-100, this keyword's own scale)
    "site_views": "float64",  # mean daily on-site views of the group
    "base_interest": "float64",  # baseline means used for the indices
    "base_site_views": "float64",
    "google_index": "float64",  # 100 = baseline mean
    "site_views_index": "float64",
}
_ALERT_SCHEMA: dict[str, str] = {
    "group_type": "str",
    "group": "str",
    "keywords": "str",  # comma-joined keywords behind the group
    "n_keywords": "int64",
    "interest_base": "float64",
    "interest_focus": "float64",
    "M_ext": "float64",  # Google momentum: mean interest focus / baseline
    "views_base": "int64",
    "views_focus": "int64",
    "days_base": "int64",
    "days_focus": "int64",
    "M_site": "float64",  # on-site momentum: views per day focus / baseline
    "gap": "float64",  # M_ext / M_site
    "reliable": "bool",  # baseline interest high enough to read
    "flagged": "bool",
    "promote_ahead": "object",  # list[str] of product_keys (flagged groups only)
    "promote_names": "object",  # list[str] of display names, same order
    "reason": "str",
}
_LEAD_LAG_SCHEMA: dict[str, str] = {
    "group_type": "str",
    "group": "str",
    "keyword": "str",
    "lag": "int64",  # weeks Google leads on-site views
    "corr": "float64",  # Pearson r of weekly % changes; descriptive only
    "n_weeks": "int64",  # week pairs behind the correlation
}

SPIKE_COLUMNS: tuple[str, ...] = tuple(_SPIKE_SCHEMA)
TEST_COLUMNS: tuple[str, ...] = tuple(_TEST_SCHEMA)
TREND_INDEX_COLUMNS: tuple[str, ...] = tuple(_INDEX_SCHEMA)
ALERT_COLUMNS: tuple[str, ...] = tuple(_ALERT_SCHEMA)
LEAD_LAG_COLUMNS: tuple[str, ...] = tuple(_LEAD_LAG_SCHEMA)

LIKELY = "Likely cannibalized"
POSSIBLE = "Possible"
NO_EVIDENCE = "No evidence"
INSUFFICIENT = "Insufficient data"
STATUSES = (LIKELY, POSSIBLE, NO_EVIDENCE, INSUFFICIENT)
_STATUS_ORDER = {s: i for i, s in enumerate(STATUSES)}

# Accepted spike signals -> item_daily column. "views_share" is profile.decisions.featured_signal.
_SIGNALS = {
    "views": "views",
    "views_share": "views",
    "promo": "promo_views",
    "promo_views": "promo_views",
    "promo_impressions": "promo_views",
}
_GROUP_COLUMN = {"category": "category_group", "brand": "brand"}

# --------------------------------------------------------------------------- shared helpers


def _empty(schema: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in schema.items()})


def _conform(df: pd.DataFrame, schema: Mapping[str, str]) -> pd.DataFrame:
    """Columns in contract order, with contract dtypes (object columns are left alone)."""
    out = df.loc[:, list(schema)].reset_index(drop=True)
    for col, dtype in schema.items():
        if dtype != "object":
            out[col] = out[col].astype(dtype)
    return out


def _as_datetime(values: pd.Series) -> pd.Series:
    """Dates (date objects, strings or timestamps) as tz-naive ``datetime64[ns]``."""
    out = pd.to_datetime(values)
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _as_period(p: Period | str) -> Period:
    if isinstance(p, str):
        if p not in config.PERIODS:
            raise ValueError(f"unknown period {p!r}; expected one of {sorted(config.PERIODS)}")
        return config.PERIODS[p]
    return p


def _within(dates: pd.Series, period: Period) -> pd.Series:
    return (dates >= pd.Timestamp(period.start)) & (dates <= pd.Timestamp(period.end))


def _dim(product_dim: pd.DataFrame) -> pd.DataFrame:
    """product_key, item_name, category_group and brand as plain strings, one row per key."""
    dim = product_dim.drop_duplicates("product_key")
    out = pd.DataFrame({"product_key": dim["product_key"].astype(str)})
    name = dim["item_name"] if "item_name" in dim.columns else dim["product_key"]
    out["item_name"] = name.fillna(dim["product_key"]).astype(str)
    for col in ("category_group", "brand"):
        out[col] = dim[col].fillna("Other").astype(str) if col in dim.columns else "Other"
    return out.reset_index(drop=True)


def _scoped(
    item_daily: pd.DataFrame | None,
    product_dim: pd.DataFrame | None,
    f: Filters,
    values: tuple[str, ...],
    *,
    use_dates: bool = True,
) -> pd.DataFrame:
    """Rows of ``item_daily`` inside ``f`` (dates optional, devices, categories) plus group columns.

    Empty ``f.devices`` means ``config.DEVICES`` (``Filters.device_set``). ``f.min_views`` is
    not applied: Phase 2 works on daily totals, not on the eligible-product list.
    """
    cols = ["event_date", "product_key", "category_group", "brand", *values]
    if item_daily is None or product_dim is None or len(item_daily) == 0 or len(product_dim) == 0:
        return pd.DataFrame(columns=cols)
    keep = [c for c in ("event_date", "device_category", "product_key", *values) if c in item_daily.columns]
    d = item_daily.loc[:, keep]
    d = d.assign(event_date=_as_datetime(d["event_date"]), product_key=d["product_key"].astype(str))
    for col in values:
        if col not in d.columns:
            d[col] = 0
    if use_dates:
        d = d[_within(d["event_date"], Period("scope", "scope", f.start, f.end))]
    if "device_category" in d.columns:
        d = d[d["device_category"].astype(str).isin(f.device_set())]
    d = d.merge(_dim(product_dim)[["product_key", "category_group", "brand"]], on="product_key", how="inner")
    if f.categories:
        d = d[d["category_group"].isin(f.categories)]
    return d.loc[:, cols].reset_index(drop=True)


def _fmt_p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"= {x:.3f}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


# --------------------------------------------------------------------------- Phase 2a: spikes


def exposure_spikes(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame,
    f: Filters | None = None,
    *,
    signal: str = "views",
    ratio: float = _C["spike_ratio"],
    min_daily: int = _C["min_daily_views"],
    window: int | None = SPIKE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Daily exposure share of each product in its category, with spike-day flags.

    One row per (product, day) for every in-scope day on which the product's category had
    traffic; days the product was not viewed have share 0. ``share = exposure / category
    exposure``. ``median_share`` is the product's median share over its in-scope days with
    exposure, and ``baseline_share`` is the same median over a centred ``window``-day rolling
    window (``window=None``: the whole scope, i.e. ``median_share``). A spike day has
    ``share >= ratio * baseline_share`` and ``exposure >= min_daily``.

    The local baseline matters on the real data: many products' shares shift level for weeks
    (launches, sell-outs, seasonal ranges). Against a season-long median a 5-week plateau
    becomes 35 "spike days", which turns a time trend into a fake cannibalization signal.
    Short bursts, the proxy for being featured, are flagged either way.

    ``signal="promo"`` uses ``promo_views`` when any are in scope and otherwise falls back to
    views (real data: promotions carry no products). The ``signal`` column records which was used.
    """
    if signal not in _SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; expected one of {sorted(_SIGNALS)}")
    f = Filters() if f is None else f
    d = _scoped(item_daily, product_dim, f, ("views", "promo_views"))
    if d.empty:
        return _empty(_SPIKE_SCHEMA)
    used = "promo" if _SIGNALS[signal] == "promo_views" and d["promo_views"].sum() > 0 else "views"
    keys = ["category_group", "product_key", "event_date"]
    daily = d.groupby(keys, as_index=False, observed=True)[["views", "promo_views"]].sum()
    daily["exposure"] = daily["promo_views" if used == "promo" else "views"]

    cat = daily.groupby(["category_group", "event_date"], as_index=False, observed=True).agg(
        category_views=("views", "sum"), category_exposure=("exposure", "sum")
    )
    cat = cat[(cat["category_views"] > 0) | (cat["category_exposure"] > 0)]
    totals = daily.groupby(["category_group", "product_key"], as_index=False, observed=True)["exposure"].sum()
    products = totals.loc[totals["exposure"] > 0, ["category_group", "product_key"]]
    grid = products.merge(cat, on="category_group").merge(
        daily[[*keys, "views", "exposure"]], on=keys, how="left"
    )
    if grid.empty:
        return _empty(_SPIKE_SCHEMA)
    grid[["views", "exposure"]] = grid[["views", "exposure"]].fillna(0).astype("int64")
    grid["share"] = (grid["exposure"] / grid["category_exposure"].where(grid["category_exposure"] > 0)).fillna(0.0)
    grid = grid.sort_values(keys, ignore_index=True)
    seen = grid["share"].where(grid["exposure"] > 0)  # the medians skip days without exposure
    grid["median_share"] = seen.groupby(grid["product_key"]).transform("median")
    if window is None or window <= 1:
        grid["baseline_share"] = grid["median_share"]
    else:
        local = seen.groupby(grid["product_key"]).transform(
            lambda s: s.rolling(int(window), center=True, min_periods=max(3, int(window) // 4)).median()
        )
        # Fewer than min_periods exposure days nearby: borrow the scope median.
        grid["baseline_share"] = local.fillna(grid["median_share"])
    grid["share_ratio"] = grid["share"] / grid["baseline_share"]
    # Relative tolerance so an exact 2.0x share is not lost to float rounding.
    grid["is_spike"] = (grid["share"] >= ratio * grid["baseline_share"] * (1 - 1e-12)) & (grid["exposure"] >= min_daily)
    grid["signal"] = used
    return _conform(grid.sort_values(keys), _SPIKE_SCHEMA)


def spike_coverage(
    spikes: pd.DataFrame,
    *,
    top_k: int = _C["top_k"],
    min_spike_days: int = _C["min_spike_days"],
) -> float:
    """Share of each category's top-K products (by views) with at least ``min_spike_days`` spikes."""
    if spikes is None or len(spikes) == 0:
        return float("nan")
    per = spikes.groupby(["category_group", "product_key"], as_index=False, observed=True).agg(
        views=("views", "sum"), n_spikes=("is_spike", "sum")
    )
    per = per.sort_values(["category_group", "views", "product_key"], ascending=[True, False, True])
    top = per.groupby("category_group", observed=True).head(top_k)
    return float((top["n_spikes"] >= min_spike_days).mean()) if len(top) else float("nan")


def suggest_spike_ratio(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame,
    f: Filters | None = None,
    *,
    signal: str = "views",
    top_k: int = _C["top_k"],
    min_spike_days: int = _C["min_spike_days"],
    min_daily: int = _C["min_daily_views"],
    window: int | None = SPIKE_WINDOW_DAYS,
    min_coverage: float = 0.30,
) -> tuple[float, float]:
    """Decision rule R10: keep the default ratio unless spikes are too sparse to test.

    Returns ``(ratio, coverage)``, where coverage is measured at the default ratio (2.0). If
    under ``min_coverage`` of the top-K products have ``min_spike_days`` spikes, the ratio drops
    to the smallest option (1.5).
    """
    default = float(_C["spike_ratio"])
    spikes = exposure_spikes(item_daily, product_dim, f, signal=signal, ratio=default, min_daily=min_daily, window=window)
    coverage = spike_coverage(spikes, top_k=top_k, min_spike_days=min_spike_days)
    if np.isnan(coverage) or coverage >= min_coverage:
        return default, coverage
    return float(min(_C["spike_ratio_options"])), coverage


# --------------------------------------------------------------------------- Phase 2a: pair tests


def _spike_days(spikes: pd.DataFrame | None) -> pd.DataFrame:
    """(product_key, event_date) of every flagged spike day."""
    if spikes is None or len(spikes) == 0 or not {"product_key", "event_date", "is_spike"} <= set(spikes.columns):
        return pd.DataFrame({"product_key": pd.Series(dtype="str"), "event_date": pd.Series(dtype="datetime64[ns]")})
    hit = spikes.loc[spikes["is_spike"].fillna(False).astype(bool), ["product_key", "event_date"]]
    return pd.DataFrame({"product_key": hit["product_key"].astype(str), "event_date": _as_datetime(hit["event_date"])})


def _pair_counts(category: str, g: pd.DataFrame, spike_days: pd.DataFrame, top_k: int) -> pd.DataFrame | None:
    """Observed and expected purchase sums for every ordered pair in one category's top-K.

    B's expected purchases on day d are ``V_B(d) * cr_peers(d)``, where the peers are the
    category minus {A, B}. Days on which the peers had no views carry no expectation, so they
    are dropped for that pair.
    """
    by_day = g.groupby("event_date")["views"].sum()
    days = by_day.index[by_day.to_numpy() > 0]
    g = g[g["event_date"].isin(days)]
    totals = g.groupby("product_key")["views"].sum()
    totals = totals[totals > 0]
    top = sorted(totals.index, key=lambda k: (-totals[k], k))[:top_k]
    if len(top) < 2:
        return None
    wide = g.set_index(["product_key", "event_date"])[["views", "purchases"]].unstack("event_date", fill_value=0)
    views = wide["views"].reindex(columns=days, fill_value=0)
    purchases = wide["purchases"].reindex(columns=days, fill_value=0)
    v_cat = views.to_numpy(float).sum(axis=0)
    p_cat = purchases.to_numpy(float).sum(axis=0)
    v_top = views.loc[top].to_numpy(float)
    p_top = purchases.loc[top].to_numpy(float)

    spiking = np.zeros(v_top.shape, dtype=bool)
    sp = spike_days[spike_days["product_key"].isin(top)]
    rows = pd.Index(top).get_indexer(sp["product_key"])
    cols = pd.DatetimeIndex(days).get_indexer(sp["event_date"])
    ok = (rows >= 0) & (cols >= 0)
    spiking[rows[ok], cols[ok]] = True

    a, b = np.nonzero(~np.eye(len(top), dtype=bool))
    v_b, p_b = v_top[b], p_top[b]
    peer_v = v_cat - v_top[a] - v_b
    peer_p = p_cat - p_top[a] - p_b
    valid = peer_v > 0
    expected = v_b * np.divide(peer_p, peer_v, out=np.zeros_like(peer_v), where=valid)
    on = spiking[a] & valid
    off = ~spiking[a] & valid
    keys = np.asarray(top, dtype=object)
    return pd.DataFrame(
        {
            "category_group": category,
            "product_a": keys[a],
            "product_b": keys[b],
            "n_spike_days": on.sum(axis=1),
            "n_other_days": off.sum(axis=1),
            "spike_views_b": np.rint((v_b * on).sum(axis=1)).astype(np.int64),
            "O_s": np.rint((p_b * on).sum(axis=1)).astype(np.int64),
            "E_s": (expected * on).sum(axis=1),
            "O_n": np.rint((p_b * off).sum(axis=1)).astype(np.int64),
            "E_n": (expected * off).sum(axis=1),
            "peer_s": np.rint((peer_p * on).sum(axis=1)).astype(np.int64),
            "peer_n": np.rint((peer_p * off).sum(axis=1)).astype(np.int64),
        }
    )


def _odds(p: np.ndarray) -> np.ndarray:
    return np.divide(p, 1 - p, out=np.full_like(p, np.inf), where=p < 1)


def _pair_reason(r, code: str, n_tested: int, g: Mapping[str, float]) -> str:
    """Plain-English explanation of one pair's status."""
    a, b = r.item_name_a, r.item_name_b
    if code == "spike_days":
        return f"{a} had {_plural(r.n_spike_days, 'spike day')} in this window; at least {g['min_spike_days']} are needed."
    if code == "other_days":
        return (
            f"Only {_plural(r.n_other_days, 'non-spike day')} in this window; at least "
            f"{g['min_other_days']} are needed for a baseline. Widen the date range."
        )
    if code == "purchases":
        return f"{b} had {_plural(r.n, 'purchase')} on the compared days; at least {g['min_purchases']} are needed."
    if code == "no_views":
        which = "spike" if r.E_s <= 0 else "other"
        return f"Nothing was expected of {b} on {a}'s {which} days (no views, or its peers sold nothing), so there is no baseline."
    if code == "expected":
        return (
            f"Only {r.n * r.pi0:.1f} purchases of {b} were expected on {a}'s spike days; "
            f"at least {g['min_expected']:g} are needed."
        )
    if code == "spike_views":
        return f"{b} had {r.spike_views_b:,} views on {a}'s spike days; at least {g['min_spike_views']:,} are needed."
    # "Usual pace" = B's own rate relative to its same-day peers on the other days, so
    # O_s / usual == rr and a best seller is not compared with average products.
    usual = r.E_s * r.O_n / r.E_n
    sold = f"{b} sold {r.O_s:,} vs {usual:.1f} expected at its usual pace against same-day category peers"
    if r.status == LIKELY:
        return f"On {a}'s {r.n_spike_days} spike days, {sold} ({r.dip:.0%} lower; q {_fmt_p(r.q_value)})."
    if r.status == POSSIBLE:
        return (
            f"On {a}'s {r.n_spike_days} spike days, {sold} ({r.dip:.0%} lower; p {_fmt_p(r.p_value)}), "
            f"but this does not survive the correction for {n_tested} tested pairs (q {_fmt_p(r.q_value)})."
        )
    if r.rr <= g["rr_flag"]:
        return f"On {a}'s spike days, {sold} ({r.dip:.0%} lower), but a dip this size is within chance (p {_fmt_p(r.p_value)})."
    if r.rr < 1:
        return f"On {a}'s spike days, {sold}: close to normal (relative risk {r.rr:.2f}, p {_fmt_p(r.p_value)})."
    return f"On {a}'s spike days, {sold}: no dip (relative risk {r.rr:.2f})."


def cannibalization_tests(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame,
    spikes: pd.DataFrame | None = None,
    f: Filters | None = None,
    *,
    top_k: int = _C["top_k"],
    min_spike_days: int = _C["min_spike_days"],
    min_other_days: int = _C["min_other_days"],
    min_purchases: int = _C["min_purchases"],
    min_expected: float = _C["min_expected"],
    min_spike_views: int = _C["min_spike_views"],
    rr_flag: float = _C["rr_flag"],
    q_flag: float = _C["q_flag"],
    p_possible: float = _C["p_possible"],
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Does product B's conversion dip on product A's exposure-spike days?

    One row per ordered pair (A, B) among each category's ``top_k`` products by in-scope
    views, using the spike days in ``spikes`` (``exposure_spikes`` with the same ``f``; computed
    with defaults when None). Under H0, ``O_s | n ~ Binomial(n, pi0)``.

    Status:
    - "Insufficient data" when a guard fails: spike days, other days, purchases, expected
      spike-day purchases, or B's spike-day views;
    - "Likely cannibalized" when ``rr <= rr_flag`` and ``q <= q_flag``;
    - "Possible" when ``rr <= rr_flag`` and ``p <= p_possible``;
    - otherwise "No evidence".

    Rows are sorted with flagged pairs first.
    """
    f = Filters() if f is None else f
    d = _scoped(item_daily, product_dim, f, ("views", "purchases"))
    if d.empty:
        return _empty(_TEST_SCHEMA)
    if spikes is None:
        spikes = exposure_spikes(item_daily, product_dim, f)
    days = _spike_days(spikes)
    keys = ["category_group", "product_key", "event_date"]
    daily = d.groupby(keys, as_index=False, observed=True)[["views", "purchases"]].sum()
    parts = [
        part
        for category, g in daily.groupby("category_group", sort=True, observed=True)
        if (part := _pair_counts(str(category), g, days, top_k)) is not None
    ]
    if not parts:
        return _empty(_TEST_SCHEMA)
    t = pd.concat(parts, ignore_index=True)
    names = _dim(product_dim).set_index("product_key")["item_name"]
    t["item_name_a"] = t["product_a"].map(names).fillna(t["product_a"])
    t["item_name_b"] = t["product_b"].map(names).fillna(t["product_b"])

    o_s, o_n = t["O_s"].to_numpy(np.int64), t["O_n"].to_numpy(np.int64)
    e_s, e_n = t["E_s"].to_numpy(float), t["E_n"].to_numpy(float)
    n = o_s + o_n
    with np.errstate(divide="ignore", invalid="ignore"):
        pi0 = e_s / (e_s + e_n)
    t["n"], t["pi0"] = n, pi0

    guards = [
        ("spike_days", t["n_spike_days"].to_numpy() < min_spike_days),
        ("other_days", t["n_other_days"].to_numpy() < min_other_days),
        ("purchases", n < min_purchases),
        ("no_views", ~((e_s > 0) & (e_n > 0))),
        ("expected", ~(n * np.nan_to_num(pi0) >= min_expected)),
        ("spike_views", t["spike_views_b"].to_numpy() < min_spike_views),
    ]
    code = np.full(len(t), "", dtype=object)
    for name, failed in guards:
        code = np.where((code == "") & failed, name, code)
    tested = code == ""

    rr, rr_low, rr_high, p, q, fisher = (np.full(len(t), np.nan) for _ in range(6))
    if tested.any():
        k, m, nt = o_s[tested], o_n[tested], n[tested]
        scale = e_n[tested] / e_s[tested]
        rr[tested] = np.where(m > 0, k / np.maximum(m, 1) * scale, np.inf)
        alpha = 1 - ci_level
        lo = np.where(k > 0, stats.beta.ppf(alpha / 2, np.maximum(k, 1), m + 1), 0.0)
        hi = np.where(m > 0, stats.beta.ppf(1 - alpha / 2, k + 1, np.maximum(m, 1)), 1.0)
        rr_low[tested], rr_high[tested] = _odds(lo) * scale, _odds(hi) * scale
        p[tested] = stats.binom.cdf(k, nt, pi0[tested])  # == binomtest(k, n, pi0, alternative="less")
        q[tested] = stats.false_discovery_control(p[tested], method="bh")
        # Fisher exact "less" on [[O_s, O_n], [peer_s, peer_n]], as a hypergeometric CDF.
        peer_s, peer_n = t["peer_s"].to_numpy(np.int64)[tested], t["peer_n"].to_numpy(np.int64)[tested]
        fisher[tested] = stats.hypergeom.cdf(k, nt + peer_s + peer_n, nt, k + peer_s)
    with np.errstate(invalid="ignore"):
        hit = rr <= rr_flag
        status = np.where(
            ~tested,
            INSUFFICIENT,
            np.where(hit & (q <= q_flag), LIKELY, np.where(hit & (p <= p_possible), POSSIBLE, NO_EVIDENCE)),
        )
    t["rr"], t["rr_low"], t["rr_high"], t["dip"] = rr, rr_low, rr_high, 1 - rr
    t["p_value"], t["q_value"], t["fisher_p"] = p, q, fisher
    t["tested"], t["status"] = tested, status

    thresholds = {
        "min_spike_days": min_spike_days,
        "min_other_days": min_other_days,
        "min_purchases": min_purchases,
        "min_expected": min_expected,
        "min_spike_views": min_spike_views,
        "rr_flag": rr_flag,
    }
    n_tested = int(tested.sum())
    t["reason"] = [_pair_reason(r, c, n_tested, thresholds) for r, c in zip(t.itertuples(index=False), code)]
    t["_order"] = t["status"].map(_STATUS_ORDER)
    t = t.sort_values(
        ["_order", "q_value", "p_value", "category_group", "product_a", "product_b"], na_position="last", kind="stable"
    )
    return _conform(t, _TEST_SCHEMA)


def conflict_pairs(tests: pd.DataFrame | None) -> set[frozenset[str]]:
    """Unordered product pairs flagged "Likely cannibalized" (in either direction)."""
    if tests is None or len(tests) == 0 or "status" not in tests.columns:
        return set()
    hit = tests.loc[tests["status"] == LIKELY, ["product_a", "product_b"]]
    return {frozenset((str(a), str(b))) for a, b in hit.itertuples(index=False)}


# --------------------------------------------------------------------------- Phase 2b: trends


def load_keyword_map(source: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    """The keyword -> group map (group_type, group_value, keyword, notes) from a frame or CSV.

    A missing file or None gives an empty map. Strings are trimmed, ``group_type`` is
    lower-cased, and the first row wins for keywords repeated in any letter case.
    """
    cols = ["group_type", "group_value", "keyword", "notes"]
    if source is None:
        km = pd.DataFrame(columns=cols)
    elif isinstance(source, pd.DataFrame):
        km = source.copy()
    else:
        path = Path(source)
        km = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=cols)
    for col in cols:
        if col not in km.columns:
            km[col] = ""
    km = km.loc[:, cols].fillna("").astype(str)
    km = km.assign(**{c: km[c].str.strip() for c in cols})
    km["group_type"] = km["group_type"].str.lower()
    km = km[(km["keyword"] != "") & (km["group_value"] != "")]
    km = km[~km["keyword"].str.casefold().duplicated()]
    return km.reset_index(drop=True)


def _clean_trends(trends_daily: pd.DataFrame | None, keyword_map) -> pd.DataFrame:
    """One interest series per keyword: group_type, group_value, keyword, date, interest.

    The keyword map wins over the groups stored in ``trends_daily``. Partial (still-moving)
    points are dropped, and when several geos are present only "US" is kept.
    """
    cols = ["group_type", "group_value", "keyword", "date", "interest"]
    if trends_daily is None or len(trends_daily) == 0 or not {"date", "keyword", "interest"} <= set(trends_daily.columns):
        return pd.DataFrame(columns=cols)
    t = trends_daily
    if "is_partial" in t.columns:
        t = t[~t["is_partial"].fillna(False).astype(bool)]
    if "geo" in t.columns and t["geo"].nunique() > 1 and (t["geo"] == "US").any():
        t = t[t["geo"] == "US"]
    t = pd.DataFrame(
        {
            "group_type": t["group_type"].astype(str).str.strip().str.lower() if "group_type" in t.columns else "",
            "group_value": t["group_value"].astype(str).str.strip() if "group_value" in t.columns else "",
            "keyword": t["keyword"].astype(str).str.strip(),
            "date": _as_datetime(t["date"]),
            "interest": pd.to_numeric(t["interest"], errors="coerce"),
        }
    )
    km = load_keyword_map(keyword_map)
    if len(km):
        lookup = km.set_index(km["keyword"].str.casefold())
        folded = t["keyword"].str.casefold()
        hit = folded.isin(lookup.index)
        for col in ("group_type", "group_value", "keyword"):
            t.loc[hit, col] = folded[hit].map(lookup[col]).to_numpy()
    t = t.dropna(subset=["date", "interest"])
    t = t[(t["keyword"] != "") & (t["group_value"] != "") & t["group_type"].isin(tuple(_GROUP_COLUMN))]
    return t.groupby(cols[:4], as_index=False)["interest"].mean()


def _group_daily_views(item_daily: pd.DataFrame, product_dim: pd.DataFrame, f: Filters, groups: pd.DataFrame) -> pd.DataFrame:
    """In-scope views per (group_type, group) on every calendar day of the item data.

    The date range of ``f`` is ignored, because the windows come from the periods.
    """
    cols = ["group_type", "group", "event_date", "views"]
    d = _scoped(item_daily, product_dim, f, ("views",), use_dates=False)
    if d.empty or groups.empty:
        return pd.DataFrame(columns=cols)
    dates = _as_datetime(item_daily["event_date"])
    days = pd.date_range(dates.min(), dates.max(), freq="D").astype("datetime64[ns]")
    parts = []
    for group_type, group in groups[["group_type", "group"]].drop_duplicates().itertuples(index=False):
        rows = d[d[_GROUP_COLUMN[group_type]] == group]
        if rows.empty:
            continue
        per_day = rows.groupby("event_date")["views"].sum().reindex(days, fill_value=0)
        parts.append(pd.DataFrame({"group_type": group_type, "group": group, "event_date": days, "views": per_day.to_numpy()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _trend_inputs(trends_daily, item_daily, product_dim, keyword_map, f: Filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean trends (with groups spelled as in product_dim) and the matching daily site views."""
    tr = _clean_trends(trends_daily, keyword_map)
    if tr.empty or product_dim is None or len(product_dim) == 0 or item_daily is None or len(item_daily) == 0:
        return tr.iloc[0:0], pd.DataFrame()
    dim = _dim(product_dim)
    spelling = {gt: {str(v).casefold(): str(v) for v in dim[col].unique()} for gt, col in _GROUP_COLUMN.items()}
    tr["group"] = [spelling[gt].get(gv.casefold()) for gt, gv in zip(tr["group_type"], tr["group_value"])]
    tr = tr[tr["group"].notna()]
    site = _group_daily_views(item_daily, product_dim, f, tr)
    if site.empty:
        return tr.iloc[0:0], site
    tr = tr.merge(site[["group_type", "group"]].drop_duplicates(), on=["group_type", "group"])
    tr = tr[(tr["date"] >= site["event_date"].min()) & (tr["date"] <= site["event_date"].max())]
    return tr, site


def _period_end(dates: pd.Series, freq: str) -> pd.Series:
    """Label each date with the end of its period ("W-SUN": the week's Sunday)."""
    return dates.dt.to_period(freq).dt.end_time.dt.normalize().astype("datetime64[ns]")


def trend_index(
    trends_daily: pd.DataFrame | None,
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame,
    keyword_map: pd.DataFrame | str | Path | None,
    base: Period | str,
    f: Filters | None = None,
    *,
    freq: str = "W-SUN",
) -> pd.DataFrame:
    """Weekly Google interest and on-site views per (group, keyword), both indexed to base = 100.

    ``interest`` and ``site_views`` are the weekly means of daily values, so partial weeks are
    not biased low. Weeks cover the item data's date range. ``f`` sets devices and categories;
    its dates are ignored. Returns an empty frame (same columns) when there are no trends.
    """
    base = _as_period(base)
    f = Filters() if f is None else f
    tr, site = _trend_inputs(trends_daily, item_daily, product_dim, keyword_map, f)
    if tr.empty or site.empty:
        return _empty(_INDEX_SCHEMA)
    gkeys, kkeys = ["group_type", "group"], ["group_type", "group", "keyword"]
    base_int = tr[_within(tr["date"], base)].groupby(kkeys, as_index=False)["interest"].mean()
    base_views = site[_within(site["event_date"], base)].groupby(gkeys, as_index=False)["views"].mean()
    weekly_site = (
        site.assign(week=_period_end(site["event_date"], freq))
        .groupby([*gkeys, "week"], as_index=False)
        .agg(site_views=("views", "mean"), n_days=("views", "size"))
    )
    weekly_tr = (
        tr.assign(week=_period_end(tr["date"], freq))
        .groupby([*kkeys, "week"], as_index=False)
        .agg(interest=("interest", "mean"))
    )
    out = (
        tr[kkeys]
        .drop_duplicates()
        .merge(weekly_site, on=gkeys)
        .merge(weekly_tr, on=[*kkeys, "week"], how="left")
        .merge(base_int.rename(columns={"interest": "base_interest"}), on=kkeys, how="left")
        .merge(base_views.rename(columns={"views": "base_site_views"}), on=gkeys, how="left")
    )
    out["site_views"] = out["site_views"].astype(float)
    out["google_index"] = 100 * out["interest"] / out["base_interest"].where(out["base_interest"] > 0)
    out["site_views_index"] = 100 * out["site_views"] / out["base_site_views"].where(out["base_site_views"] > 0)
    return _conform(out.sort_values([*kkeys, "week"]), _INDEX_SCHEMA)


def _views_cutoff(products: pd.DataFrame | None, th_views: float | Thresholds | None) -> float:
    """The quadrant x-line T_v: as given, else the median views of eligible products."""
    if isinstance(th_views, Thresholds):
        return float(th_views.views)
    if th_views is not None:
        return float(th_views)
    if products is None or len(products) == 0 or "views" not in products.columns:
        return float("inf")
    pool = products[products["eligible"].fillna(False).astype(bool)] if "eligible" in products.columns else products
    views = pool["views"][pool["views"] > 0]
    return float(views.median()) if len(views) else float("inf")


def _promote_ahead(
    products: pd.DataFrame | None, group_type: str, group: str, th_views: float, limit: int | None
) -> tuple[list[str], list[str]]:
    """Group members converting above the group's prior but below the high-traffic line.

    Ranked by ``lift_per_1k``, then ``cr_adj``. Category groups use each product's
    ``prior_mean``; brand groups use the brand's pooled rate. Ineligible products are skipped.
    """
    col = _GROUP_COLUMN[group_type]
    if products is None or len(products) == 0 or not {"product_key", "cr_adj", "views", col} <= set(products.columns):
        return [], []
    members = products[products[col].astype(str).str.casefold() == group.casefold()]
    if members.empty:
        return [], []
    if group_type == "category" and "prior_mean" in members.columns:
        prior = members["prior_mean"].astype(float)
    else:
        views = float(members["views"].sum())
        prior = float(members["purchases"].sum()) / views if views > 0 and "purchases" in members.columns else np.nan
    pick = members[(members["cr_adj"] > prior) & (members["views"] < th_views)]
    if "eligible" in pick.columns:
        pick = pick[pick["eligible"].fillna(False).astype(bool)]
    order = [c for c in ("lift_per_1k", "cr_adj") if c in pick.columns]
    pick = pick.sort_values([*order, "product_key"], ascending=[*([False] * len(order)), True])
    if limit is not None:
        pick = pick.head(limit)
    names = pick["item_name"] if "item_name" in pick.columns else pick["product_key"]
    return pick["product_key"].astype(str).tolist(), names.astype(str).tolist()


def _alert_reason(r: Mapping, base: Period, focus: Period, g: Mapping[str, float]) -> str:
    """Plain-English explanation of one group's alert status."""
    group = r["group"]
    if np.isnan(r["interest_base"]):
        return f"No Google Trends data for {group} in the {base.label.lower()} window."
    if not r["reliable"]:
        return (
            f"Search interest for {group} is too low to read (baseline average {r['interest_base']:.1f} "
            f"of 100; at least {g['min_base_interest']:g} is needed)."
        )
    if np.isnan(r["M_ext"]):
        return f"No Google Trends data for {group} in the {focus.label.lower()} window."
    if np.isnan(r["M_site"]):
        return f"No on-site views for {group} in one of the windows."
    ext = f"Google interest {r['M_ext'] - 1:+.0%}"
    site = f"on-site views per day {r['M_site'] - 1:+.0%}"
    if r["flagged"]:
        return (
            f"{ext} in {focus.label} vs {base.label}, but {site}: demand outside the store is "
            f"outrunning our exposure (gap {r['gap']:.2f}x). Promote ahead of demand."
        )
    if r["views_base"] < g["min_base_views"]:
        return f"{ext}, but only {r['views_base']:,} baseline views on site (at least {g['min_base_views']:,} are needed)."
    if r["M_ext"] < g["ext_min"]:
        return f"{ext} in {focus.label} vs {base.label}: not trending (needs {g['ext_min'] - 1:+.0%})."
    return f"{ext} and {site}: our exposure is keeping pace (gap {r['gap']:.2f}x; needs {g['gap_min']:.2f}x)."


def trending_alerts(
    trends_daily: pd.DataFrame | None,
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame,
    keyword_map: pd.DataFrame | str | Path | None,
    base: Period | str,
    focus: Period | str,
    f: Filters | None = None,
    products_scored: pd.DataFrame | None = None,
    th_views: float | Thresholds | None = None,
    *,
    ext_min: float = _T["ext_min"],
    gap_min: float = _T["gap_min"],
    min_base_views: int = _T["min_base_views"],
    min_base_interest: float = _T["min_base_interest"],
    max_promote: int | None = None,
) -> pd.DataFrame:
    """"Trending outside our walls": Google momentum vs on-site view momentum per group.

    ``M_ext`` = mean interest in focus / mean interest in base, per keyword. A group's value
    is the mean over its reliable keywords (base mean >= ``min_base_interest``).
    ``M_site`` = (group views / day, focus) / (group views / day, base), and ``gap = M_ext / M_site``.

    A group is flagged when it is reliable, ``M_ext >= ext_min``, ``gap >= gap_min`` and
    baseline views >= ``min_base_views``. For flagged groups, ``promote_ahead`` lists products
    from ``products_scored`` (``types.PRODUCT_COLUMNS``) with ``cr_adj`` above the group prior and
    views below ``th_views`` (default: median views of eligible products). ``f`` sets devices
    and categories; its dates are ignored. With no trends data, returns an empty frame.
    """
    base, focus = _as_period(base), _as_period(focus)
    f = Filters() if f is None else f
    tr, site = _trend_inputs(trends_daily, item_daily, product_dim, keyword_map, f)
    if tr.empty or site.empty:
        return _empty(_ALERT_SCHEMA)
    gkeys, kkeys = ["group_type", "group"], ["group_type", "group", "keyword"]
    kw = tr[kkeys].drop_duplicates()
    for label, period in (("interest_base", base), ("interest_focus", focus)):
        means = tr[_within(tr["date"], period)].groupby(kkeys, as_index=False)["interest"].mean()
        kw = kw.merge(means.rename(columns={"interest": label}), on=kkeys, how="left")
    kw["m_ext"] = kw["interest_focus"] / kw["interest_base"].where(kw["interest_base"] > 0)
    kw["reliable"] = kw["interest_base"] >= min_base_interest
    site_b = site[_within(site["event_date"], base)].groupby(gkeys)["views"].agg(["sum", "size"])
    site_f = site[_within(site["event_date"], focus)].groupby(gkeys)["views"].agg(["sum", "size"])
    cutoff = _views_cutoff(products_scored, th_views)
    limits = {
        "ext_min": ext_min,
        "gap_min": gap_min,
        "min_base_views": min_base_views,
        "min_base_interest": min_base_interest,
    }

    rows = []
    for (group_type, group), k in kw.groupby(gkeys, sort=True):
        reliable = bool(k["reliable"].any())
        use = k[k["reliable"]] if reliable else k
        vb, db = (int(x) for x in site_b.loc[(group_type, group)]) if (group_type, group) in site_b.index else (0, 0)
        vf, df = (int(x) for x in site_f.loc[(group_type, group)]) if (group_type, group) in site_f.index else (0, 0)
        m_ext = float(use["m_ext"].mean())
        m_site = (vf / df) / (vb / db) if vb > 0 and db > 0 and df > 0 else np.nan
        if np.isnan(m_ext) or np.isnan(m_site):
            gap = np.nan
        else:
            gap = m_ext / m_site if m_site > 0 else np.inf
        flagged = bool(reliable and m_ext >= ext_min and gap >= gap_min and vb >= min_base_views)
        promote, names = _promote_ahead(products_scored, group_type, group, cutoff, max_promote) if flagged else ([], [])
        row = {
            "group_type": group_type,
            "group": group,
            "keywords": ", ".join(sorted(k["keyword"])),
            "n_keywords": len(k),
            "interest_base": float(use["interest_base"].mean()),
            "interest_focus": float(use["interest_focus"].mean()),
            "M_ext": m_ext,
            "views_base": vb,
            "views_focus": vf,
            "days_base": db,
            "days_focus": df,
            "M_site": m_site,
            "gap": gap,
            "reliable": reliable,
            "flagged": flagged,
            "promote_ahead": promote,
            "promote_names": names,
        }
        row["reason"] = _alert_reason(row, base, focus, limits)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["flagged", "gap", "group"], ascending=[False, False, True], na_position="last")
    return _conform(out, _ALERT_SCHEMA)


def lead_lag(index_df: pd.DataFrame | None, max_lag: int = 3, *, min_days: int = 4) -> pd.DataFrame:
    """Correlation of weekly % changes: Google interest ``lag`` weeks earlier vs on-site views.

    This is descriptive only (about 13 weeks of data). Weeks with fewer than ``min_days`` days of
    on-site data, such as a stub first week, are dropped. A correlation needs at least 3 week pairs.
    """
    need = {"group_type", "group", "keyword", "week"}
    if index_df is None or len(index_df) == 0 or not need <= set(index_df.columns):
        return _empty(_LEAD_LAG_SCHEMA)
    g_col = "interest" if "interest" in index_df.columns else "google_index"
    s_col = "site_views" if "site_views" in index_df.columns else "site_views_index"
    rows = []
    for (group_type, group, keyword), s in index_df.groupby(["group_type", "group", "keyword"], sort=True):
        s = s.sort_values("week")
        if "n_days" in s.columns:
            s = s[s["n_days"] >= min_days]
        google = s[g_col].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        onsite = s[s_col].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        for lag in range(max_lag + 1):
            lead = google.shift(lag)
            ok = lead.notna() & onsite.notna()
            n = int(ok.sum())
            corr = float(np.corrcoef(lead[ok], onsite[ok])[0, 1]) if n >= 3 and lead[ok].std() > 0 and onsite[ok].std() > 0 else np.nan
            rows.append({"group_type": group_type, "group": group, "keyword": keyword, "lag": lag, "corr": corr, "n_weeks": n})
    return _conform(pd.DataFrame(rows), _LEAD_LAG_SCHEMA) if rows else _empty(_LEAD_LAG_SCHEMA)

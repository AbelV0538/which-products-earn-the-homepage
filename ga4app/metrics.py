"""Product metrics: scope filtering, aggregation, empirical-Bayes shrinkage, quadrants,
action scores, holiday shift, quadrant movers, device lean, campaign scores and the slot plan.

Pure pandas / numpy / scipy (no Streamlit), so every function runs on hand-made frames.
The formulas follow ``docs/DESIGN.md`` section 5:

* ``p_bar`` (the site average) is ``sum(P) / sum(V)`` over the products in scope with V > 0.
  Products without views never enter a rate and are never eligible.
* A product's prior mean is its category's pooled rate in scope; categories with fewer than
  ``MIN_CATEGORY_PRODUCTS`` viewed products use ``p_bar`` instead.
* The posterior is ``Beta(P + m*mu, max(V - P, 0) + m*(1 - mu))``, i.e. it uses
  ``V_eff = max(V, P)``. ``m`` is estimated once (full season, all devices) by method of
  moments and held fixed across filters so scores stay comparable.

Functions that compare fixed windows (``period_comparison``, ``quadrant_movers``,
``campaign_scores``) ignore the date range of the ``Filters`` they receive and apply only its
devices / categories (/ min views). ``device_matrix`` ignores the device filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .config import Period
from .types import PRODUCT_COLUMNS, Filters, Thresholds

__all__ = [
    "ALL_PRODUCTS",
    "LEANS",
    "MOVER_STATUSES",
    "PRODUCT_COLUMNS",
    "Filters",
    "Period",
    "Thresholds",
    "campaign_scores",
    "classify",
    "compute_products",
    "daily_totals",
    "device_lean",
    "device_matrix",
    "device_summary",
    "estimate_prior_strength",
    "feature_candidates",
    "filter_daily",
    "period_comparison",
    "pooled_rate",
    "product_table",
    "quadrant_movers",
    "score_actions",
    "season_prior_strength",
    "shrink",
    "site_conversion",
    "slot_plan",
    "swap_gain",
    "thresholds",
    "transition_counts",
    "wilson_ci",
]

WINNERS, HIDDEN_GEMS, WASTED, LOW_PRIORITY = "Winners", "Hidden gems", "Wasted exposure", "Low priority"
ALL_PRODUCTS = "All products"  # group label of the site-wide rows of period_comparison(by=None)
MIN_CATEGORY_PRODUCTS = 20  # smaller categories borrow the overall pooled rate as prior mean
MIN_TEST_PURCHASES = 10  # two-proportion test only when both windows have this many purchases
MOVER_Z = 1.96
MOVER_MIN_REL_CHANGE = 0.25
LEANS = ("Mobile-first", "Desktop-first", "Both", "Not enough data")
MOVER_STATUSES = ("Rising", "Fading", "Stable", "Insufficient data")

# Additive item_daily columns summed per product (whichever are present).
_SUM_COLUMNS = (
    "views",
    "view_rows",
    "view_events",
    "primary_views",
    "add_to_carts",
    "checkouts",
    "purchases",
    "purchase_events_raw",
    "purchases_missing_txn",
    "units",
    "promo_views",
    "promo_clicks",
    "revenue_usd",
)
# Always present on a product table (zero-filled when the input lacks them).
_BASE_SUMS = ("views", "purchases", "units", "revenue_usd", "promo_views")
_RULE_QUANTILES = {"median": 0.5, "p50": 0.5, "p75": 0.75, "p90": 0.9}
_ONE_DAY = pd.Timedelta(days=1)

_PERIOD_COLUMNS = (
    "group",
    "period",
    "period_label",
    "views",
    "purchases",
    "cr",
    "ci_low",
    "ci_high",
    "lift",
    "p_value",
    "low_sample",
    "lift_vs_rest",
    "p_value_vs_rest",
    "is_base",
    "days",
    "views_per_day",
    "purchases_per_day",
    "revenue_usd",
)
_SLOT_COLUMNS = (
    "campaign",
    "campaign_key",
    "slot",
    "product_key",
    "item_name",
    "category_group",
    "brand",
    "score",
    "score_per_1k",
    "vs_avg",
    "window_avg",
    "views_w",
    "purchases_w",
    "season_cr_adj",
    "uplift",
)


# --------------------------------------------------------------------------- helpers


def _dates(s: pd.Series) -> pd.Series:
    """``s`` as datetime64 (snapshots load that way; raw parquet gives ``datetime.date``)."""
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s)


def _in_window(dates: pd.Series, start: date, end: date) -> np.ndarray:
    """Inclusive date mask; also keeps timestamps later on the end day."""
    d = _dates(dates)
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize() + _ONE_DAY
    return ((d >= lo) & (d < hi)).to_numpy(dtype=bool)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    """A count column as float, zeros when absent or missing."""
    if col not in df.columns:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)


def _text(s: pd.Series, default: str) -> pd.Series:
    return s.astype("str").fillna(default)


def _label_or_other(value: Any) -> str:
    return "Other" if value is None or pd.isna(value) else str(value)


def _pooled(views: np.ndarray, purchases: np.ndarray) -> float:
    """``sum(P) / sum(V)`` over entries with V > 0 (0.0 when nothing was viewed)."""
    seen = views > 0
    total = views[seen].sum()
    return float(purchases[seen].sum() / total) if total > 0 else 0.0


def _check_m(m: float) -> float:
    m = float(m)
    if not np.isfinite(m) or m < 0:
        raise ValueError(f"prior strength m must be a finite number >= 0, got {m!r}")
    return m


def _group_prior(groups: np.ndarray, views: np.ndarray, purchases: np.ndarray, overall: float) -> np.ndarray:
    """Per-row pooled rate of the row's group, or ``overall`` for small / unviewed groups."""
    if not len(groups):
        return np.zeros(0)
    seen = views > 0
    g = (
        pd.DataFrame({"g": groups[seen], "v": views[seen], "p": purchases[seen]})
        .groupby("g")
        .agg(n=("v", "size"), v=("v", "sum"), p=("p", "sum"))
    )
    rate = (g["p"] / g["v"]).where((g["n"] >= MIN_CATEGORY_PRODUCTS) & (g["v"] > 0))
    mu = pd.Series(groups).map(rate).to_numpy(dtype=float)
    return np.where(np.isfinite(mu), mu, overall)


def _categories(products: pd.DataFrame) -> np.ndarray:
    if "category_group" not in products.columns:
        return np.full(len(products), "Other", dtype=object)
    return _text(products["category_group"], "Other").to_numpy(dtype=object)


def _beta_interval(a: np.ndarray, b: np.ndarray, interval: float) -> tuple[np.ndarray, np.ndarray]:
    """Central ``interval`` of Beta(a, b); degenerate at 0 / 1 when a or b is zero."""
    q = (1.0 - float(interval)) / 2.0
    lo = np.full(a.shape, np.nan)
    hi = np.full(a.shape, np.nan)
    ok = (a > 0) & (b > 0)
    if ok.any():
        lo[ok] = stats.beta.ppf(q, a[ok], b[ok])
        hi[ok] = stats.beta.ppf(1.0 - q, a[ok], b[ok])
    at_zero = (a <= 0) & (b > 0)
    lo[at_zero] = hi[at_zero] = 0.0
    at_one = (b <= 0) & (a > 0)
    lo[at_one] = hi[at_one] = 1.0
    return lo, hi


def _dim_lookup(product_dim: pd.DataFrame | None, column: str) -> dict[str, Any]:
    if product_dim is None or column not in product_dim.columns or "product_key" not in product_dim.columns:
        return {}
    dim = product_dim.drop_duplicates("product_key")
    return dict(zip(dim["product_key"].astype(str), dim[column]))


def _labels_of(keys: pd.Series, product_dim: pd.DataFrame | None, column: str = "category_group") -> pd.Series:
    """Map product keys to a product_dim column ("Other" when unknown)."""
    lookup = _dim_lookup(product_dim, column)
    return _text(keys.astype(str).map(lookup), "Other")


def _attach_dim(frame: pd.DataFrame, product_dim: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join the product_dim columns; unknown products get their key as name and "Other"."""
    out = frame
    if product_dim is not None and "product_key" in product_dim.columns:
        cols = [c for c in product_dim.columns if c == "product_key" or c not in out.columns]
        dim = product_dim.drop_duplicates("product_key")[cols].copy()
        dim["product_key"] = dim["product_key"].astype(str)
        out = out.merge(dim, on="product_key", how="left")
    else:
        out = out.copy()
    if "item_name" not in out.columns:
        out["item_name"] = out["product_key"]
    out["item_name"] = out["item_name"].astype("str").fillna(out["product_key"])
    for col in ("category_group", "brand"):
        out[col] = _text(out[col], "Other") if col in out.columns else "Other"
    return out


def _as_period(p: Any) -> Period:
    """A ``Period`` from a Period-like object, a period key or a campaign name."""
    if isinstance(p, str):
        if p in config.PERIODS:
            return config.PERIODS[p]
        if p in config.CAMPAIGNS:
            return config.PERIODS[config.CAMPAIGNS[p]]
        raise ValueError(f"unknown period {p!r}")
    if hasattr(p, "start") and hasattr(p, "end"):
        return p
    raise TypeError(f"not a period: {p!r}")


def _as_periods(periods: Any) -> list[Period]:
    if isinstance(periods, (str, Period)):
        periods = [periods]
    elif isinstance(periods, Mapping):
        periods = list(periods.values())
    return [_as_period(p) for p in periods]


def _period_days(p: Period) -> int:
    return (p.end - p.start).days + 1


def _two_prop_p(k1: np.ndarray, n1: np.ndarray, k2: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """Two-sided p-value of the pooled two-proportion z-test."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = (k1 + k2) / (n1 + n2)
        se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
        z = (k2 / n2 - k1 / n1) / se
        return np.where(np.isfinite(z), 2.0 * stats.norm.sf(np.abs(z)), np.nan)


def _log_rate_var(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Delta-method variance of ``log(k / n)`` for a binomial proportion."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 - k / n) / k


def _ordered(products: pd.DataFrame) -> pd.DataFrame:
    """PRODUCT_COLUMNS first, extras after."""
    extras = [c for c in products.columns if c not in PRODUCT_COLUMNS]
    return products[[*PRODUCT_COLUMNS, *extras]]


# --------------------------------------------------------------------------- scope + aggregation


def filter_daily(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    f: Filters | None = None,
    *,
    ignore_dates: bool = False,
    ignore_devices: bool = False,
) -> pd.DataFrame:
    """Rows of ``item_daily`` inside scope ``f``.

    Dates are inclusive; empty ``f.devices`` / ``f.categories`` mean all. Categories filter on
    ``product_dim.category_group`` (products missing from the dim count as "Other").
    """
    f = f if f is not None else Filters()
    mask = np.ones(len(item_daily), dtype=bool)
    if not ignore_dates:
        mask &= _in_window(item_daily["event_date"], f.start, f.end)
    if f.devices and not ignore_devices:
        mask &= item_daily["device_category"].isin(list(f.devices)).to_numpy(dtype=bool)
    if f.categories:
        wanted = set(f.categories)
        lookup = _dim_lookup(product_dim, "category_group")
        keys = item_daily["product_key"].astype("str")
        keep = [k for k in pd.unique(keys) if _label_or_other(lookup.get(k)) in wanted]
        mask &= keys.isin(keep).to_numpy(dtype=bool)
    return item_daily.loc[mask].reset_index(drop=True)


def product_table(daily_f: pd.DataFrame, product_dim: pd.DataFrame | None) -> pd.DataFrame:
    """One row per product in ``daily_f``: summed counts plus the product_dim columns."""
    sums = [c for c in _SUM_COLUMNS if c in daily_f.columns]
    out = daily_f.groupby("product_key", observed=True, sort=True)[sums].sum().reset_index()
    out["product_key"] = out["product_key"].astype("str")
    for col in _BASE_SUMS:
        if col not in out.columns:
            out[col] = 0.0 if col == "revenue_usd" else 0
    return _attach_dim(out, product_dim)


def pooled_rate(products: pd.DataFrame) -> float:
    """The site average ``p_bar = sum(P) / sum(V)`` over products with V > 0."""
    return _pooled(_num(products, "views"), _num(products, "purchases"))


# --------------------------------------------------------------------------- empirical Bayes


def prior_strength_raw(products_full: pd.DataFrame, *, min_views: int = config.SHRINK_MIN_VIEWS) -> float:
    """The unclamped method-of-moments ``m`` behind ``estimate_prior_strength``.

    NaN when nothing can be estimated (fewer than two products, or a degenerate pooled rate);
    ``inf`` when ``tau2 <= 0`` (no heterogeneity beyond binomial noise).
    """
    v = _num(products_full, "views")
    p = _num(products_full, "purchases")
    used = (v >= max(min_views, 1)) & (v > 0)
    if used.sum() < 2:
        return float("nan")
    v_eff = np.maximum(v[used], p[used])
    p = p[used]
    mu_bar = float(p.sum() / v_eff.sum())
    if not 0.0 < mu_bar < 1.0:
        return float("nan")
    mu_c = _group_prior(_categories(products_full)[used], v_eff, p, mu_bar)
    r = p / v_eff
    tau2 = (np.sum(v_eff * (r - mu_c) ** 2) - np.sum(mu_c * (1.0 - mu_c))) / v_eff.sum()
    if not np.isfinite(tau2):
        return float("nan")
    if tau2 <= 0:
        return float("inf")
    return float(mu_bar * (1.0 - mu_bar) / tau2 - 1.0)


def estimate_prior_strength(
    products_full: pd.DataFrame,
    *,
    min_views: int = config.SHRINK_MIN_VIEWS,
    bounds: tuple[float, float] = config.SHRINK_M_BOUNDS,
) -> float:
    """Method-of-moments prior strength ``m`` from a full-season, all-device product table.

    ``tau2 = [sum V_i (r_i - mu_c)^2 - sum mu_c (1 - mu_c)] / sum V_i`` over products with
    ``V >= min_views``; ``m = mu (1 - mu) / tau2 - 1`` clamped to ``bounds``, and the upper bound
    when ``tau2 <= 0`` (no heterogeneity beyond binomial noise) or nothing can be estimated.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    raw = prior_strength_raw(products_full, min_views=min_views)
    if not np.isfinite(raw):
        return hi
    return float(np.clip(raw, lo, hi))


def season_prior_strength(item_daily: pd.DataFrame, product_dim: pd.DataFrame | None) -> float:
    """``estimate_prior_strength`` over the whole snapshot (full season, all devices and categories)."""
    return estimate_prior_strength(product_table(item_daily, product_dim))


def season_prior_strength_raw(item_daily: pd.DataFrame, product_dim: pd.DataFrame | None) -> float:
    """``prior_strength_raw`` over the whole snapshot (the value before the ``SHRINK_M_BOUNDS`` clamp)."""
    return prior_strength_raw(product_table(item_daily, product_dim))


def shrink(products: pd.DataFrame, m: float, *, interval: float = config.CONF_INTERVAL) -> pd.DataFrame:
    """Add ``cr_raw``, ``prior_mean``, ``cr_adj``, ``ci_low``/``ci_high``, ``confidence`` and
    ``purchases_exceed_views`` (plus the posterior parameters ``post_a``/``post_b``).

    ``confidence`` is "above" / "below" when the ``interval`` posterior interval lies entirely
    above / below the pooled rate ``p_bar`` of ``products``, else "uncertain".
    """
    m = _check_m(m)
    out = products.copy()
    v = _num(out, "views")
    p = _num(out, "purchases")
    p_bar = _pooled(v, p)
    # A category whose purchases exceed its views would give a negative Beta b.
    mu = np.clip(_group_prior(_categories(out), v, p, p_bar), 0.0, 1.0)
    v_eff = np.maximum(v, p)
    a = p + m * mu
    b = (v_eff - p) + m * (1.0 - mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr_raw = np.where(v > 0, p / v, np.nan)
        cr_adj = np.where(a + b > 0, a / (a + b), np.nan)
    lo, hi = _beta_interval(a, b, interval)
    confidence = np.where(lo > p_bar, "above", np.where(hi < p_bar, "below", "uncertain"))
    out["cr_raw"] = cr_raw
    out["prior_mean"] = mu
    out["cr_adj"] = cr_adj
    out["ci_low"] = lo
    out["ci_high"] = hi
    # No data at all: the interval is just the prior, which says nothing about this product.
    out["confidence"] = np.where(v_eff > 0, confidence, "uncertain")
    out["purchases_exceed_views"] = p > v
    out["post_a"] = a
    out["post_b"] = b
    return out


# --------------------------------------------------------------------------- quadrants + actions


def _views_cutoff(v_eligible: np.ndarray, rule: str | float) -> tuple[float | None, str]:
    if isinstance(rule, (int, float, np.number)) and not isinstance(rule, bool):
        return float(rule), "custom"
    key = str(rule).strip().lower()
    if key in _RULE_QUANTILES:
        q = _RULE_QUANTILES[key]
    elif key.startswith("p") and key[1:].isdigit() and 0 < int(key[1:]) < 100:
        q = int(key[1:]) / 100.0
    else:
        try:
            return float(key), "custom"
        except ValueError:
            raise ValueError(f"unknown views_rule {rule!r}; use median, p75, p90 or a number") from None
    if not len(v_eligible):
        return None, key
    return float(np.quantile(v_eligible, q)), key


def thresholds(products: pd.DataFrame, min_views: int, views_rule: str | float = "median") -> Thresholds:
    """Quadrant split lines.

    ``views`` is the ``views_rule`` cutoff (median / p75 / p90 / a custom number) over ELIGIBLE
    products (``V >= min_views`` and ``V > 0``); ``conversion`` is ``p_bar`` over ALL products in
    scope with V > 0. With no eligible product the views line falls back to ``max(min_views, 1)``.
    """
    v = _num(products, "views")
    p = _num(products, "purchases")
    eligible = (v >= min_views) & (v > 0)
    cut, rule = _views_cutoff(v[eligible], views_rule)
    if cut is None:
        cut = float(max(min_views, 1))
    return Thresholds(views=float(cut), conversion=_pooled(v, p), views_rule=rule)


def classify(
    products: pd.DataFrame,
    th: Thresholds,
    *,
    y: str = "cr_raw",
    min_views: int | None = None,
) -> pd.DataFrame:
    """Add ``eligible`` and ``quadrant`` (one of ``config.QUADRANTS``; "" when not eligible).

    ``y`` is the plotted metric ("cr_raw" or "cr_adj"). Eligibility is ``V >= min_views`` and
    ``V > 0``; with ``min_views=None`` an existing ``eligible`` column is reused, else
    ``config.DEFAULT_MIN_VIEWS`` applies.
    """
    if y not in ("cr_raw", "cr_adj"):
        raise ValueError(f"y must be 'cr_raw' or 'cr_adj', got {y!r}")
    if y not in products.columns:
        raise KeyError(f"products has no {y!r} column; run shrink() first")
    out = products.copy()
    v = _num(out, "views")
    if min_views is None and "eligible" in out.columns:
        eligible = out["eligible"].fillna(False).to_numpy(dtype=bool) & (v > 0)
    else:
        mv = config.DEFAULT_MIN_VIEWS if min_views is None else min_views
        eligible = (v >= mv) & (v > 0)
    yv = pd.to_numeric(out[y], errors="coerce").to_numpy(dtype=float)
    high = v >= th.views
    up = yv >= th.conversion
    out["eligible"] = eligible
    out["quadrant"] = np.select(
        [eligible & high & up, eligible & high & ~up, eligible & ~high & up, eligible & ~high & ~up],
        [WINNERS, WASTED, HIDDEN_GEMS, LOW_PRIORITY],
        default="",
    )
    return out


def score_actions(products: pd.DataFrame, th: Thresholds, *, top_n_codes: int = config.TOP_N_CODES) -> pd.DataFrame:
    """Add ``forgone_purchases``, ``lift_per_1k``, ``rpv``, ``views_share``, ``action_rank`` and ``code``.

    * ``forgone_purchases = V * max(p_bar - cr_adj, 0)`` (purchases lost vs an average product);
    * ``lift_per_1k = 1000 * max(cr_adj - p_bar, 0)`` (extra purchases per 1,000 extra views);
    * ranks: Winners by ``cr_adj``, Hidden gems by ``lift_per_1k`` (confidence "below" excluded),
      Wasted exposure by ``forgone_purchases``; the top ``top_n_codes`` get W#/G#/X# codes.
    """
    out = products if "quadrant" in products.columns else classify(products, th)
    out = out.copy()
    if "cr_adj" not in out.columns:
        raise KeyError("products has no 'cr_adj' column; run shrink() first")
    v = _num(out, "views")
    revenue = _num(out, "revenue_usd")
    cr_adj = pd.to_numeric(out["cr_adj"], errors="coerce").to_numpy(dtype=float)
    p_bar = float(th.conversion)
    seen = v > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out["forgone_purchases"] = np.where(seen, v * np.maximum(p_bar - cr_adj, 0.0), 0.0)
        out["lift_per_1k"] = np.where(seen, 1000.0 * np.maximum(cr_adj - p_bar, 0.0), 0.0)
        out["rpv"] = np.where(seen, revenue / v, np.nan)
    total = v.sum()
    out["views_share"] = v / total if total > 0 else np.zeros(len(out))

    rank = np.full(len(out), np.nan)
    code = np.full(len(out), "", dtype=object)
    quadrant = out["quadrant"].astype("str").fillna("").to_numpy(dtype=object)
    confidence = (
        out["confidence"].astype("str").fillna("").to_numpy(dtype=object)
        if "confidence" in out.columns
        else np.full(len(out), "", dtype=object)
    )
    order_by = {
        WINNERS: ["cr_adj", "views"],
        HIDDEN_GEMS: ["lift_per_1k", "cr_adj", "views"],
        WASTED: ["forgone_purchases", "views"],
    }
    pos = np.arange(len(out))
    for quad, prefix in config.QUADRANT_CODE_PREFIX.items():
        sel = quadrant == quad
        if quad == HIDDEN_GEMS:
            sel &= confidence != "below"
        if not sel.any():
            continue
        cols = order_by.get(quad, ["cr_adj", "views"])
        part = out.iloc[pos[sel]][[*cols, "product_key"]].assign(_pos=pos[sel])
        part = part.sort_values([*cols, "product_key"], ascending=[*([False] * len(cols)), True], kind="mergesort")
        ranked = part["_pos"].to_numpy()
        rank[ranked] = np.arange(1, len(ranked) + 1)
        top = ranked[:top_n_codes]
        code[top] = [f"{prefix}{i}" for i in range(1, len(top) + 1)]
    out["action_rank"] = rank
    out["code"] = code
    return out


def compute_products(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    f: Filters | None,
    m: float,
    *,
    views_rule: str | float = "median",
    y: str = "cr_raw",
) -> tuple[pd.DataFrame, Thresholds]:
    """The one-call pipeline the pages use: filter -> product table -> shrink -> thresholds ->
    classify -> action scores. Returns every ``types.PRODUCT_COLUMNS`` column (first, in order)
    plus the product_dim columns, sorted by views (descending)."""
    f = f if f is not None else Filters()
    daily = filter_daily(item_daily, product_dim, f)
    products = shrink(product_table(daily, product_dim), m)
    th = thresholds(products, f.min_views, views_rule)
    products = classify(products, th, y=y, min_views=f.min_views)
    products = score_actions(products, th)
    products = products.sort_values(["views", "product_key"], ascending=[False, True], kind="mergesort")
    return _ordered(products.reset_index(drop=True)), th


def feature_candidates(products: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The verdict's "Feature" list: top ``n`` by ``cr_adj`` among Hidden gems and Winners with
    confidence "above", topped up with "uncertain" ones when fewer than ``n`` qualify."""
    pool = products[
        products["quadrant"].isin([HIDDEN_GEMS, WINNERS]) & products["confidence"].isin(["above", "uncertain"])
    ]
    pool = pool.assign(_tier=(pool["confidence"] != "above").astype(int))
    pool = pool.sort_values(
        ["_tier", "cr_adj", "views", "product_key"], ascending=[True, False, False, True], kind="mergesort"
    )
    return pool.head(n).drop(columns="_tier").reset_index(drop=True)


def swap_gain(products: pd.DataFrame, n: int = config.TOP_N_CODES) -> pd.DataFrame:
    """Pair the i-th ranked Hidden gem with the i-th ranked Wasted-exposure product.

    ``swap_gain_per_1k = 1000 * (cr_adj_G - cr_adj_X)``: extra purchases per 1,000 views moved
    from the wasted product to the gem. Columns: ``pair, gem_code, gem_key, gem_name, gem_cr_adj,
    wasted_code, wasted_key, wasted_name, wasted_cr_adj, swap_gain_per_1k``.
    """
    cols = ["pair", "gem_code", "gem_key", "gem_name", "gem_cr_adj",
            "wasted_code", "wasted_key", "wasted_name", "wasted_cr_adj", "swap_gain_per_1k"]

    def ranked(quad: str) -> pd.DataFrame:
        sel = products[(products["quadrant"] == quad) & products["action_rank"].notna()]
        return sel.sort_values("action_rank", kind="mergesort").head(n).reset_index(drop=True)

    gems, wasted = ranked(HIDDEN_GEMS), ranked(WASTED)
    k = min(len(gems), len(wasted))
    if k == 0:
        return pd.DataFrame(columns=cols)
    g, x = gems.iloc[:k], wasted.iloc[:k]
    name_col = "item_name" if "item_name" in products.columns else "product_key"
    out = pd.DataFrame(
        {
            "pair": np.arange(1, k + 1),
            "gem_code": g["code"].to_numpy(dtype=object),
            "gem_key": g["product_key"].to_numpy(dtype=object),
            "gem_name": g[name_col].to_numpy(dtype=object),
            "gem_cr_adj": g["cr_adj"].to_numpy(dtype=float),
            "wasted_code": x["code"].to_numpy(dtype=object),
            "wasted_key": x["product_key"].to_numpy(dtype=object),
            "wasted_name": x[name_col].to_numpy(dtype=object),
            "wasted_cr_adj": x["cr_adj"].to_numpy(dtype=float),
        }
    )
    out["swap_gain_per_1k"] = 1000.0 * (out["gem_cr_adj"] - out["wasted_cr_adj"])
    return out


# --------------------------------------------------------------------------- intervals + windows


def wilson_ci(k: Any, n: Any, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for ``k`` successes in ``n`` trials (NaN where ``n <= 0``)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.clip(k / n, 0.0, 1.0)
        z2 = z * z
        denom = 1.0 + z2 / n
        centre = (p + z2 / (2.0 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
        bad = ~(n > 0)
        low = np.where(bad, np.nan, np.clip(centre - half, 0.0, 1.0))
        high = np.where(bad, np.nan, np.clip(centre + half, 0.0, 1.0))
    return low, high


def _group_labels(daily: pd.DataFrame, product_dim: pd.DataFrame | None, by: str | None) -> pd.Series:
    if by is None:
        return pd.Series(ALL_PRODUCTS, index=daily.index, dtype="str")
    if by in daily.columns:
        return _text(daily[by], "Other")
    return _labels_of(daily["product_key"], product_dim, by)


def period_comparison(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    periods: Sequence[Period] | Mapping[str, Period],
    f: Filters | None = None,
    *,
    by: str | None = "category_group",
) -> pd.DataFrame:
    """Conversion per group and window (long frame), compared with the FIRST window.

    Columns: ``group, period, period_label, views, purchases, cr, ci_low, ci_high`` (95% Wilson),
    ``lift`` (``cr / cr_first - 1``; NaN on the first window's rows), ``p_value`` (two-proportion
    z-test, only when both windows have >= 10 purchases), ``low_sample``, ``lift_vs_rest`` and
    ``p_value_vs_rest``, plus ``is_base, days, views_per_day, purchases_per_day, revenue_usd``.

    ``lift_vs_rest = RR_group / RR_rest - 1`` compares the group's rate ratio with that of every
    other group in scope pooled, which separates a category's own holiday effect from the
    store-wide surge; its p-value is a Wald test on the log ratio of the two rate ratios (all
    four purchase counts >= 10, else NaN). The date range of ``f`` is ignored (fixed windows);
    ``by=None`` gives site-wide rows labelled "All products" (no rest, so NaN). Like ``p_bar``,
    products with no views in a window are left out of that window's rate, and groups with no
    views in any window are dropped.
    """
    f = f if f is not None else Filters()
    plist = _as_periods(periods)
    if not plist:
        return pd.DataFrame(columns=list(_PERIOD_COLUMNS))
    daily = filter_daily(item_daily, product_dim, f, ignore_dates=True)
    labels = _group_labels(daily, product_dim, by)
    base_frame = pd.DataFrame(
        {
            "group": labels.to_numpy(dtype=object),
            "product_key": daily["product_key"].astype("str").to_numpy(dtype=object),
            "views": _num(daily, "views"),
            "purchases": _num(daily, "purchases"),
            "revenue_usd": _num(daily, "revenue_usd"),
        }
    )
    groups = _group_order(base_frame, by)
    frames = []
    for i, per in enumerate(plist):
        sub = base_frame.loc[_in_window(daily["event_date"], per.start, per.end)]
        per_product = sub.groupby(["group", "product_key"])[["views", "purchases", "revenue_usd"]].sum()
        per_product = per_product[per_product["views"] > 0]
        agg = per_product.groupby(level="group").sum().reindex(groups, fill_value=0.0)
        agg = agg.rename_axis("group").reset_index()
        agg.insert(1, "period", per.key)
        agg.insert(2, "period_label", getattr(per, "label", per.key))
        agg["is_base"] = i == 0
        agg["days"] = _period_days(per)
        agg["_i"] = i
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True)
    out = out[out.groupby("group")["views"].transform("sum").to_numpy(dtype=float) > 0].reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=list(_PERIOD_COLUMNS))
    v = out["views"].to_numpy(dtype=float)
    p = out["purchases"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cr"] = np.where(v > 0, p / v, np.nan)
    out["ci_low"], out["ci_high"] = wilson_ci(p, v)

    first = frames[0].set_index("group")
    base_v = out["group"].map(first["views"]).to_numpy(dtype=float)
    base_p = out["group"].map(first["purchases"]).to_numpy(dtype=float)
    is_base = out["is_base"].to_numpy(dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        base_cr = np.where(base_v > 0, base_p / base_v, np.nan)
        lift = out["cr"].to_numpy(dtype=float) / base_cr - 1.0
    out["lift"] = np.where(is_base | ~np.isfinite(lift), np.nan, lift)
    testable = ~is_base & (p >= MIN_TEST_PURCHASES) & (base_p >= MIN_TEST_PURCHASES)
    out["p_value"] = np.where(testable, _two_prop_p(base_p, base_v, p, v), np.nan)
    out["low_sample"] = (p < MIN_TEST_PURCHASES) | (~is_base & (base_p < MIN_TEST_PURCHASES))

    # The same comparison for "every other group in scope", pooled.
    totals = out.groupby("_i")[["views", "purchases"]].transform("sum")
    rest_v = totals["views"].to_numpy(dtype=float) - v
    rest_p = totals["purchases"].to_numpy(dtype=float) - p
    at_base = out["_i"].to_numpy() == 0
    groups_base = out.loc[at_base, "group"]
    rest_bv = out["group"].map(pd.Series(rest_v[at_base], index=groups_base)).to_numpy(dtype=float)
    rest_bp = out["group"].map(pd.Series(rest_p[at_base], index=groups_base)).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rr = np.log((p / v) / (base_p / base_v)) - np.log((rest_p / rest_v) / (rest_bp / rest_bv))
        se = np.sqrt(_log_rate_var(p, v) + _log_rate_var(base_p, base_v) + _log_rate_var(rest_p, rest_v) + _log_rate_var(rest_bp, rest_bv))
        z_rest = log_rr / se
    out["lift_vs_rest"] = np.where(is_base | ~np.isfinite(log_rr), np.nan, np.expm1(log_rr))
    rest_ok = testable & (rest_p >= MIN_TEST_PURCHASES) & (rest_bp >= MIN_TEST_PURCHASES) & np.isfinite(z_rest)
    out["p_value_vs_rest"] = np.where(rest_ok, 2.0 * stats.norm.sf(np.abs(np.nan_to_num(z_rest))), np.nan)
    days = out["days"].to_numpy(dtype=float)
    out["views_per_day"] = v / days
    out["purchases_per_day"] = p / days
    for col in ("views", "purchases"):
        out[col] = out[col].round().astype("int64")
    return out[list(_PERIOD_COLUMNS)]


def _group_order(frame: pd.DataFrame, by: str | None) -> list[str]:
    present = list(dict.fromkeys(frame["group"]))
    if by == "category_group":
        known = [g for g in config.CATEGORY_GROUPS if g in present]
        return known + sorted(g for g in present if g not in known)
    if by is None:
        return [ALL_PRODUCTS]
    totals = frame.groupby("group")["views"].sum().sort_values(ascending=False, kind="mergesort")
    return [str(g) for g in totals.index]


def site_conversion(
    site_daily: pd.DataFrame,
    periods: Sequence[Period] | Mapping[str, Period],
    *,
    devices: Sequence[str] = (),
) -> pd.DataFrame:
    """Site-level ``purchase / view_item`` events per window, from ``site_daily``.

    Columns: ``period, period_label, start, end, days, views, purchases, cr, ci_low, ci_high``.
    """
    plist = _as_periods(periods)
    sd = site_daily
    if devices:
        sd = sd[sd["device_category"].isin(list(devices))]
    names = sd["event_name"].astype("str").to_numpy(dtype=object)
    events = _num(sd, "events")
    rows = []
    for per in plist:
        in_w = _in_window(sd["event_date"], per.start, per.end)
        rows.append(
            {
                "period": per.key,
                "period_label": getattr(per, "label", per.key),
                "start": per.start,
                "end": per.end,
                "days": _period_days(per),
                "views": int(events[in_w & (names == "view_item")].sum()),
                "purchases": int(events[in_w & (names == "purchase")].sum()),
            }
        )
    out = pd.DataFrame(rows, columns=["period", "period_label", "start", "end", "days", "views", "purchases"])
    v = out["views"].to_numpy(dtype=float)
    p = out["purchases"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cr"] = np.where(v > 0, p / v, np.nan)
    out["ci_low"], out["ci_high"] = wilson_ci(p, v)
    return out


def daily_totals(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    f: Filters | None = None,
    *,
    keys: Iterable[str] | None = None,
    ignore_dates: bool = False,
) -> pd.DataFrame:
    """Views, purchases, revenue and ``cr`` per day in scope (optionally for some products only).

    Every day of the scope's range appears (zeros on days without rows), so lines have no gaps.
    """
    f = f if f is not None else Filters()
    daily = filter_daily(item_daily, product_dim, f, ignore_dates=ignore_dates)
    if keys is not None:
        daily = daily[daily["product_key"].astype("str").isin([str(k) for k in keys])]
    days = _dates(daily["event_date"]).dt.normalize().to_numpy()
    g = (
        pd.DataFrame(
            {
                "event_date": days,
                "views": _num(daily, "views"),
                "purchases": _num(daily, "purchases"),
                "revenue_usd": _num(daily, "revenue_usd"),
            }
        )
        .groupby("event_date")
        .sum()
    )
    all_days = _dates(item_daily["event_date"]).dt.normalize() if len(item_daily) else pd.Series(dtype="datetime64[ns]")
    if len(all_days):
        lo, hi = all_days.min(), all_days.max()
        if not ignore_dates:
            lo, hi = max(lo, pd.Timestamp(f.start)), min(hi, pd.Timestamp(f.end))
        g = g.reindex(pd.date_range(lo, hi, freq="D", name="event_date"), fill_value=0.0)
    out = g.reset_index()
    for col in ("views", "purchases"):
        out[col] = out[col].round().astype("int64")
    v = out["views"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cr"] = np.where(v > 0, out["purchases"].to_numpy(dtype=float) / v, np.nan)
    return out


# --------------------------------------------------------------------------- quadrant movers


def quadrant_movers(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    base: Period | str,
    focus: Period | str,
    f: Filters | None,
    m: float,
    *,
    y: str = "cr_raw",
    views_rule: str | float = "median",
) -> pd.DataFrame:
    """Per product: quadrant and ``cr_adj`` in ``base`` vs ``focus``, each window classified with
    its own thresholds and category priors (same ``m``).

    ``z = (logit cr_adj_f - logit cr_adj_b) / sqrt(1/a_b + 1/b_b + 1/a_f + 1/b_f)``; status is
    "Rising" / "Fading" when ``|z| >= 1.96`` and ``|rel_change| >= 25%`` in the same direction,
    "Insufficient data" when the product is not eligible in both windows (or z is undefined),
    else "Stable".
    """
    f = f if f is not None else Filters()
    pb, pf = _as_period(base), _as_period(focus)
    tb, _ = compute_products(item_daily, product_dim, replace(f, start=pb.start, end=pb.end), m, views_rule=views_rule, y=y)
    tf, _ = compute_products(item_daily, product_dim, replace(f, start=pf.start, end=pf.end), m, views_rule=views_rule, y=y)
    cols = ["product_key", "views", "purchases", "cr_raw", "cr_adj", "quadrant", "post_a", "post_b"]
    mv = tb[cols].merge(tf[cols], on="product_key", how="outer", suffixes=("_base", "_focus"))
    names = pd.concat([tb, tf])[["product_key", "item_name", "category_group", "brand"]].drop_duplicates("product_key")
    mv = mv.merge(names, on="product_key", how="left")
    for col in ("views_base", "views_focus", "purchases_base", "purchases_focus"):
        mv[col] = mv[col].fillna(0).astype("int64")
    for col in ("quadrant_base", "quadrant_focus"):
        mv[col] = mv[col].astype("str").fillna("")

    a_b, b_b = mv["post_a_base"].to_numpy(dtype=float), mv["post_b_base"].to_numpy(dtype=float)
    a_f, b_f = mv["post_a_focus"].to_numpy(dtype=float), mv["post_b_focus"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (np.log(a_f) - np.log(b_f)) - (np.log(a_b) - np.log(b_b))
        z = delta / np.sqrt(1.0 / a_b + 1.0 / b_b + 1.0 / a_f + 1.0 / b_f)
        rel = mv["cr_adj_focus"].to_numpy(dtype=float) / mv["cr_adj_base"].to_numpy(dtype=float) - 1.0
    z = np.where(np.isfinite(z), z, np.nan)
    rel = np.where(np.isfinite(rel), rel, np.nan)
    both = (mv["quadrant_base"] != "").to_numpy() & (mv["quadrant_focus"] != "").to_numpy()
    mv["rel_change"] = rel
    mv["z"] = z
    mv["status"] = np.select(
        [
            ~both | ~np.isfinite(z),
            (z >= MOVER_Z) & (rel >= MOVER_MIN_REL_CHANGE),
            (z <= -MOVER_Z) & (rel <= -MOVER_MIN_REL_CHANGE),
        ],
        ["Insufficient data", "Rising", "Fading"],
        default="Stable",
    )
    mv["moved"] = both & (mv["quadrant_base"] != mv["quadrant_focus"]).to_numpy()
    mv["transition"] = np.where(both, mv["quadrant_base"] + " → " + mv["quadrant_focus"], "")
    mv["base"], mv["focus"] = pb.key, pf.key
    mv = mv.drop(columns=["post_a_base", "post_b_base", "post_a_focus", "post_b_focus"])
    mv = mv.sort_values(["z", "product_key"], ascending=[False, True], na_position="last", kind="mergesort")
    first = ["product_key", "item_name", "category_group", "brand", "quadrant_base", "quadrant_focus",
             "cr_adj_base", "cr_adj_focus", "rel_change", "z", "status"]
    return mv[[*first, *[c for c in mv.columns if c not in first]]].reset_index(drop=True)


def transition_counts(movers: pd.DataFrame) -> pd.DataFrame:
    """4x4 counts of products by (base quadrant, focus quadrant), in ``config.QUADRANTS`` order."""
    quads = list(config.QUADRANTS)
    out = pd.DataFrame(
        0, index=pd.Index(quads, name="quadrant_base"), columns=pd.Index(quads, name="quadrant_focus"), dtype="int64"
    )
    both = movers[movers["quadrant_base"].isin(quads) & movers["quadrant_focus"].isin(quads)]
    for (qb, qf), n in both.groupby(["quadrant_base", "quadrant_focus"]).size().items():
        out.loc[qb, qf] = int(n)
    return out


# --------------------------------------------------------------------------- devices


def device_matrix(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    keys: Sequence[str],
    f: Filters | None,
    m: float,
) -> pd.DataFrame:
    """Long frame, one row per (product_key, device) for ``keys`` x every device.

    Columns: ``product_key, device`` (alias ``device_category``), ``views, purchases, cr_raw,
    cr_adj_d, device_rate`` (``p_bar_d``), ``index = cr_adj_d / p_bar_d``, ``log2_index`` (clipped
    to +-1, for the diverging colour), ``cr_adj`` (all-device), ``p_bar``, plus name / category.
    ``cr_adj_d = (P + m*mu_id) / (max(V, P) + m)`` with ``mu_id = cr_adj * p_bar_d / p_bar``.
    The device filter of ``f`` is ignored; its dates and categories apply.
    """
    f = f if f is not None else Filters()
    m = _check_m(m)
    daily = filter_daily(item_daily, product_dim, f, ignore_devices=True)
    products = shrink(product_table(daily, product_dim), m)
    p_bar = pooled_rate(products)
    adj = dict(zip(products["product_key"], products["cr_adj"]))

    cells = pd.DataFrame(
        {
            "product_key": daily["product_key"].astype("str").to_numpy(dtype=object),
            "device": _text(daily["device_category"], "other").to_numpy(dtype=object),
            "views": _num(daily, "views"),
            "purchases": _num(daily, "purchases"),
        }
    )
    cells = cells.groupby(["product_key", "device"], as_index=False)[["views", "purchases"]].sum()
    seen = cells[cells["views"] > 0].groupby("device")[["views", "purchases"]].sum()
    rate_by_device = (seen["purchases"] / seen["views"]).to_dict()
    devices = list(config.DEVICES) + sorted(set(cells["device"]) - set(config.DEVICES))

    keys = list(dict.fromkeys(str(k) for k in keys))
    grid = pd.MultiIndex.from_product([keys, devices], names=["product_key", "device"]).to_frame(index=False)
    grid = grid.merge(cells, on=["product_key", "device"], how="left")
    v = grid["views"].fillna(0).to_numpy(dtype=float)
    p = grid["purchases"].fillna(0).to_numpy(dtype=float)
    rate_d = grid["device"].map(rate_by_device).to_numpy(dtype=float)
    cr_i = grid["product_key"].map(adj).to_numpy(dtype=float)
    cr_i = np.where(np.isfinite(cr_i), cr_i, p_bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = cr_i * rate_d / p_bar if p_bar > 0 else rate_d
        cr_adj_d = (p + m * mu) / (np.maximum(v, p) + m)
        index = np.where(rate_d > 0, cr_adj_d / rate_d, np.nan)
        cr_raw = np.where(v > 0, p / v, np.nan)
        log2_index = np.clip(np.log2(index), -1.0, 1.0)
    grid["views"] = v.astype("int64")
    grid["purchases"] = p.astype("int64")
    grid["cr_raw"] = cr_raw
    grid["cr_adj_d"] = cr_adj_d
    grid["device_rate"] = rate_d
    grid["index"] = index
    grid["log2_index"] = log2_index
    grid["cr_adj"] = cr_i
    grid["p_bar"] = p_bar
    grid["device_category"] = grid["device"]
    grid = _attach_dim(grid, product_dim)
    first = ["product_key", "item_name", "category_group", "device", "views", "purchases", "cr_raw", "cr_adj_d",
             "device_rate", "index", "log2_index", "cr_adj", "p_bar", "device_category"]
    return grid[[*first, *[c for c in grid.columns if c not in first and c != "brand"], "brand"]]


def device_lean(
    matrix: pd.DataFrame,
    *,
    rr_hi: float = config.DEVICE_LEAN["rr_hi"],
    rr_lo: float = config.DEVICE_LEAN["rr_lo"],
    alpha: float = config.DEVICE_LEAN["alpha"],
    min_purchases: int = config.DEVICE_LEAN["min_purchases"],
) -> pd.DataFrame:
    """Mobile-vs-desktop lean per product from a ``device_matrix`` (tablet is left out).

    ``E_d = V_d * p_bar_d``; ``rr = (O_m/E_m) / (O_d/E_d)`` (+0.5 on both counts when one is 0);
    ``p_value`` = two-sided ``binomtest(O_m, O_m + O_d, E_m / (E_m + E_d))``. ``lean`` is
    "Mobile-first" (rr >= rr_hi and p < alpha), "Desktop-first" (rr <= rr_lo and p < alpha),
    "Not enough data" (O_m + O_d < min_purchases, or a device without views), else "Both".
    """
    dev_col = "device" if "device" in matrix.columns else "device_category"
    keys = list(dict.fromkeys(matrix["product_key"].astype("str")))
    device = matrix[dev_col].astype("str").to_numpy(dtype=object)
    by_dev: dict[str, pd.DataFrame] = {}
    for d in ("mobile", "desktop"):
        sub = matrix.loc[device == d]
        by_dev[d] = sub.assign(_k=sub["product_key"].astype("str")).drop_duplicates("_k").set_index("_k")

    def cell(dev: str, key: str, col: str) -> float:
        frame = by_dev[dev]
        if key not in frame.index or col not in frame.columns:
            return 0.0
        val = float(frame.at[key, col])
        return val if np.isfinite(val) else 0.0

    rows = []
    for key in keys:
        vm, om, rm = cell("mobile", key, "views"), cell("mobile", key, "purchases"), cell("mobile", key, "device_rate")
        vd, od, rd = cell("desktop", key, "views"), cell("desktop", key, "purchases"), cell("desktop", key, "device_rate")
        em, ed = vm * rm, vd * rd
        n = om + od
        rr = p_value = np.nan
        if em > 0 and ed > 0:
            if n > 0:
                cm, cd = (om, od) if om > 0 and od > 0 else (om + 0.5, od + 0.5)
                rr = (cm / em) / (cd / ed)
                p_value = float(stats.binomtest(int(round(om)), int(round(n)), em / (em + ed)).pvalue)
        if n < min_purchases or not (em > 0 and ed > 0):
            lean = "Not enough data"
        elif rr >= rr_hi and p_value < alpha:
            lean = "Mobile-first"
        elif rr <= rr_lo and p_value < alpha:
            lean = "Desktop-first"
        else:
            lean = "Both"
        rows.append(
            {
                "product_key": key,
                "views_mobile": int(round(vm)),
                "views_desktop": int(round(vd)),
                "purchases_mobile": int(round(om)),
                "purchases_desktop": int(round(od)),
                "expected_mobile": em,
                "expected_desktop": ed,
                "rr": rr,
                "p_value": p_value,
                "lean": lean,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=["product_key", "views_mobile", "views_desktop", "purchases_mobile", "purchases_desktop",
                 "expected_mobile", "expected_desktop", "rr", "p_value", "lean"],
    )
    for col in ("item_name", "category_group"):
        if col in matrix.columns:
            lookup = dict(zip(matrix["product_key"].astype("str"), matrix[col]))
            out.insert(1 if col == "item_name" else 2, col, out["product_key"].map(lookup))
    return out


def device_summary(item_daily: pd.DataFrame, product_dim: pd.DataFrame | None, f: Filters | None = None) -> pd.DataFrame:
    """Per device in scope (device filter ignored): ``views, purchases, cr`` (pooled over
    product-device cells with V > 0), ``views_share``, ``purchases_share``."""
    f = f if f is not None else Filters()
    daily = filter_daily(item_daily, product_dim, f, ignore_devices=True)
    cells = pd.DataFrame(
        {
            "product_key": daily["product_key"].astype("str").to_numpy(dtype=object),
            "device": _text(daily["device_category"], "other").to_numpy(dtype=object),
            "views": _num(daily, "views"),
            "purchases": _num(daily, "purchases"),
        }
    ).groupby(["product_key", "device"], as_index=False)[["views", "purchases"]].sum()
    cells = cells[cells["views"] > 0]
    devices = list(config.DEVICES) + sorted(set(cells["device"]) - set(config.DEVICES))
    out = cells.groupby("device")[["views", "purchases"]].sum().reindex(devices, fill_value=0.0)
    out = out.rename_axis("device").reset_index()
    v = out["views"].to_numpy(dtype=float)
    p = out["purchases"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cr"] = np.where(v > 0, p / v, np.nan)
        out["views_share"] = v / v.sum() if v.sum() > 0 else 0.0
        out["purchases_share"] = p / p.sum() if p.sum() > 0 else 0.0
    out["views"] = out["views"].astype("int64")
    out["purchases"] = out["purchases"].astype("int64")
    return out


# --------------------------------------------------------------------------- campaigns + slot plan


def _group_rates(products: pd.DataFrame) -> pd.DataFrame:
    """Per category: summed views / purchases over products with V > 0, and the pooled rate."""
    seen = products[_num(products, "views") > 0]
    g = (
        pd.DataFrame({"g": _categories(seen), "views": _num(seen, "views"), "purchases": _num(seen, "purchases")})
        .groupby("g")[["views", "purchases"]]
        .sum()
    )
    g["cr"] = np.where(g["views"] > 0, g["purchases"] / g["views"].where(g["views"] > 0, 1.0), np.nan)
    return g


def campaign_scores(
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame | None,
    campaign: Period | str,
    f: Filters | None,
    m: float,
) -> pd.DataFrame:
    """Expected conversion per product in a campaign window, borrowing strength from the season.

    ``s_i`` = full-season ``cr_adj``; ``u_cw = cr_cw / cr_c,season`` per category (the site-wide
    uplift when the category has < ``config.CATEGORY_UPLIFT_MIN_VIEWS`` window views);
    ``mu_iw = s_i * u_cw``; ``score = (P_w + m*mu_iw) / (max(V_w, P_w) + m)``.
    Columns include ``score, score_per_1k, vs_avg`` (``score / window_avg - 1``), ``window_avg``,
    ``views_w, purchases_w, cr_raw_w, season_cr_adj, uplift, uplift_source, prior_w, forgone_w``,
    ``eligible`` (season views >= ``f.min_views``) and ``rank`` (among eligible; 1 = best).
    The date range of ``f`` is ignored; its devices and categories apply. Products without
    season views are left out.
    """
    f = f if f is not None else Filters()
    m = _check_m(m)
    per = _as_period(campaign)
    season_daily = filter_daily(item_daily, product_dim, f, ignore_dates=True)
    season = shrink(product_table(season_daily, product_dim), m)
    season = season[season["views"] > 0].reset_index(drop=True)
    window = product_table(season_daily.loc[_in_window(season_daily["event_date"], per.start, per.end)], product_dim)

    cat_s, cat_w = _group_rates(season), _group_rates(window)
    site_s, site_w = pooled_rate(season), pooled_rate(window)
    window_views = float(_num(window, "views").sum())
    site_uplift = site_w / site_s if site_s > 0 and window_views > 0 else 1.0
    uplift: dict[str, float] = {}
    source: dict[str, str] = {}
    for cat in cat_s.index:
        cr_season = float(cat_s.at[cat, "cr"])
        v_w = float(cat_w.at[cat, "views"]) if cat in cat_w.index else 0.0
        if v_w >= config.CATEGORY_UPLIFT_MIN_VIEWS and cr_season > 0:
            uplift[cat], source[cat] = float(cat_w.at[cat, "cr"]) / cr_season, "category"
        else:
            uplift[cat], source[cat] = site_uplift, "site"

    out = season[["product_key", "item_name", "category_group", "brand", "views", "purchases", "cr_adj"]].rename(
        columns={"views": "views_season", "purchases": "purchases_season", "cr_adj": "season_cr_adj"}
    )
    w = window.set_index("product_key")
    out["views_w"] = out["product_key"].map(w["views"]).fillna(0).astype("int64")
    out["purchases_w"] = out["product_key"].map(w["purchases"]).fillna(0).astype("int64")
    cats = out["category_group"].astype("str")
    out["uplift"] = cats.map(uplift).fillna(site_uplift).to_numpy(dtype=float)
    out["uplift_source"] = cats.map(source).fillna("site")
    v_w = out["views_w"].to_numpy(dtype=float)
    p_w = out["purchases_w"].to_numpy(dtype=float)
    prior = out["season_cr_adj"].to_numpy(dtype=float) * out["uplift"].to_numpy(dtype=float)
    score = (p_w + m * prior) / (np.maximum(v_w, p_w) + m) if m > 0 else np.where(v_w > 0, p_w / np.maximum(v_w, p_w), prior)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["cr_raw_w"] = np.where(v_w > 0, p_w / v_w, np.nan)
    out["prior_w"] = prior
    out["score"] = score
    out["score_per_1k"] = 1000.0 * score
    out["window_avg"] = site_w if window_views > 0 else np.nan
    out["vs_avg"] = score / site_w - 1.0 if window_views > 0 and site_w > 0 else np.nan
    out["forgone_w"] = v_w * np.maximum(site_w - score, 0.0) if window_views > 0 else 0.0
    out["eligible"] = out["views_season"].to_numpy(dtype=float) >= max(f.min_views, 1)
    out["campaign_key"] = per.key
    out["campaign_label"] = getattr(per, "label", per.key)
    out = out.sort_values(["score", "views_w", "product_key"], ascending=[False, False, True], kind="mergesort")
    out["rank"] = np.nan
    elig = out["eligible"].to_numpy(dtype=bool)
    out.loc[elig, "rank"] = np.arange(1, int(elig.sum()) + 1, dtype=float)
    return out.reset_index(drop=True)


def slot_plan(
    scores: Mapping[str, pd.DataFrame],
    n_slots: Mapping[str, int] | int,
    *,
    conflicts: Iterable[frozenset[str]] = frozenset(),
    exclude: Iterable[str] = frozenset(),
    allow_repeats: bool = False,
) -> pd.DataFrame:
    """Greedy homepage plan: per campaign (``config.CAMPAIGNS`` order first), take the best
    ``score_per_1k`` products, skipping ``exclude``d keys, any product whose ``conflicts``
    partner (cannibalization pair) is already picked for that campaign and, unless
    ``allow_repeats``, products already picked for an earlier campaign. Only rows with
    ``eligible`` True are candidates when that column exists. A campaign missing from
    ``n_slots`` gets ``config.DEFAULT_SLOTS``.
    """
    partners: dict[str, set[str]] = {}
    for pair in conflicts:
        members = [str(k) for k in pair]
        if len(members) != 2:
            continue
        a, b = members
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    excluded = {str(k) for k in exclude}
    order = [c for c in config.CAMPAIGNS if c in scores] + [c for c in scores if c not in config.CAMPAIGNS]
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for campaign in order:
        frame = scores[campaign]
        n = int(n_slots) if isinstance(n_slots, (int, np.integer)) else int(n_slots.get(campaign, config.DEFAULT_SLOTS))
        if frame is None or not len(frame) or n <= 0:
            continue
        cand = frame
        if "eligible" in cand.columns:
            cand = cand[cand["eligible"].fillna(False).astype(bool)]
        sort_cols = [c for c in ("score_per_1k", "views_w") if c in cand.columns]
        cand = cand.sort_values([*sort_cols, "product_key"], ascending=[*([False] * len(sort_cols)), True], kind="mergesort")
        picked: list[str] = []
        for rec in cand.to_dict("records"):
            key = str(rec["product_key"])
            if key in excluded or key in picked or (not allow_repeats and key in used):
                continue
            if partners.get(key, set()) & set(picked):
                continue
            picked.append(key)
            rows.append(
                {
                    "campaign": campaign,
                    "campaign_key": rec.get("campaign_key") or config.CAMPAIGNS.get(campaign, ""),
                    "slot": len(picked),
                    "product_key": key,
                    **{c: rec.get(c, np.nan) for c in _SLOT_COLUMNS[4:]},
                }
            )
            if len(picked) >= n:
                break
        used.update(picked)
    out = pd.DataFrame(rows, columns=list(_SLOT_COLUMNS))
    out["slot"] = out["slot"].astype("int64")
    return out

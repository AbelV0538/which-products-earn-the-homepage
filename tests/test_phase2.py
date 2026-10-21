"""Phase 2 analytics on small hand-made frames: spikes, pair tests, BH, trends and lead/lag."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ga4app import config, phase2
from ga4app.config import Period
from ga4app.types import Filters

START = date(2020, 11, 2)


# --------------------------------------------------------------------------- builders


def _frames(
    views: dict[str, list[int] | np.ndarray],
    purchases: dict[str, list[int] | np.ndarray] | None = None,
    *,
    category: str | dict[str, str] = "Apparel",
    start: date = START,
    promo: dict[str, list[int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """item_daily (one desktop row per product-day) and product_dim from per-day lists."""
    rows = []
    for key, v in views.items():
        p = (purchases or {}).get(key, np.zeros(len(v), dtype=int))
        pr = (promo or {}).get(key, np.zeros(len(v), dtype=int))
        for i, (vi, pi, qi) in enumerate(zip(v, p, pr)):
            rows.append(
                {
                    "event_date": pd.Timestamp(start) + pd.Timedelta(days=i),
                    "device_category": "desktop",
                    "product_key": key,
                    "views": int(vi),
                    "purchases": int(pi),
                    "promo_views": int(qi),
                }
            )
    item = pd.DataFrame(rows)
    cats = category if isinstance(category, dict) else {k: category for k in views}
    dim = pd.DataFrame(
        {
            "product_key": list(views),
            "item_name": [k.title() for k in views],
            "category_group": [cats[k] for k in views],
            "brand": "Google",
        }
    )
    return item, dim


def _spikes(key: str, day_idx: list[int], *, start: date = START) -> pd.DataFrame:
    """A hand-made spike frame: ``key`` spikes on the given day offsets."""
    return pd.DataFrame(
        {
            "product_key": key,
            "event_date": [pd.Timestamp(start) + pd.Timedelta(days=i) for i in day_idx],
            "is_spike": True,
        }
    )


def _flat_category(days: int = 40, *, b_rate: float = 0.02, b_views: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deterministic category: every product converts at 2% (B at ``b_rate``) every day."""
    views = {"a": [100] * days, "b": [b_views] * days, "p1": [500] * days, "p2": [500] * days, "p3": [500] * days}
    purchases = {k: [round(v[0] * 0.02)] * days for k, v in views.items()}
    purchases["b"] = [round(b_views * b_rate)] * days
    return _frames(views, purchases)


def _pair(tests: pd.DataFrame, a: str, b: str) -> pd.Series:
    row = tests[(tests["product_a"] == a) & (tests["product_b"] == b)]
    assert len(row) == 1
    return row.iloc[0]


# --------------------------------------------------------------------------- exposure spikes


def test_spike_detection_on_hand_made_series():
    n = 30
    a = np.full(n, 30)
    a[[5, 12, 20]] = 90  # share 90/362 ~ 0.25 >= 2 x 0.099
    a[25] = 60  # share 60/332 ~ 0.18: elevated, but under 2x
    d = np.full(n, 2)
    d[8] = 12  # 6x its share, but only 12 views (< min_daily)
    item, dim = _frames({"a": a, "b": [120] * n, "c": [150] * n, "d": d})
    out = phase2.exposure_spikes(item, dim, Filters())

    assert tuple(out.columns) == phase2.SPIKE_COLUMNS
    assert len(out) == 4 * n
    spike_days = out.loc[out["is_spike"], ["product_key", "event_date"]]
    assert set(spike_days["product_key"]) == {"a"}
    expected = {pd.Timestamp(START) + pd.Timedelta(days=i) for i in (5, 12, 20)}
    assert set(spike_days["event_date"]) == expected
    # Shares partition each category-day.
    assert np.allclose(out.groupby("event_date")["share"].sum(), 1.0)
    row_a = out[out["product_key"] == "a"].iloc[0]
    assert row_a["median_share"] == pytest.approx(30 / 302)
    assert row_a["category_views"] == 302
    assert (out["signal"] == "views").all()
    # The season-wide median (window=None) gives the same spikes on a level series.
    season = phase2.exposure_spikes(item, dim, Filters(), window=None)
    assert season["is_spike"].sum() == 3


def test_spike_ratio_and_min_daily_are_respected():
    n = 30
    a = np.full(n, 30)
    a[[5, 12, 20]] = 90
    a[25] = 60
    item, dim = _frames({"a": a, "b": [120] * n, "c": [150] * n})
    assert phase2.exposure_spikes(item, dim, ratio=1.5)["is_spike"].sum() == 4  # day 25 joins
    assert phase2.exposure_spikes(item, dim, ratio=3.0)["is_spike"].sum() == 0
    assert phase2.exposure_spikes(item, dim, min_daily=100)["is_spike"].sum() == 0


def test_category_wide_surge_is_not_a_spike():
    n = 30
    mult = np.ones(n, dtype=int)
    mult[[7, 8, 9]] = 3  # Black-Friday-style traffic surge for the whole category
    item, dim = _frames({"a": 40 * mult, "b": 120 * mult, "c": 150 * mult})
    assert not phase2.exposure_spikes(item, dim)["is_spike"].any()


def test_level_shift_is_not_a_run_of_spikes():
    """A product whose share doubles for two weeks is a plateau, not 15 featured days."""
    n = 40
    a = np.full(n, 40)
    a[25:] = 100
    item, dim = _frames({"a": a, "b": [150] * n, "c": [150] * n})
    assert phase2.exposure_spikes(item, dim, window=None)["is_spike"].sum() == 15
    assert phase2.exposure_spikes(item, dim)["is_spike"].sum() <= 1


def test_promo_signal_with_and_without_promotions():
    n = 20
    views = {"a": [50] * n, "b": [100] * n}
    item, dim = _frames(views)
    assert (phase2.exposure_spikes(item, dim, signal="promo")["signal"] == "views").all()  # real data: no promos

    promo_a = [5] * n
    promo_a[4] = 40
    item, dim = _frames(views, promo={"a": promo_a, "b": [10] * n})
    out = phase2.exposure_spikes(item, dim, signal="promo", min_daily=30)
    assert (out["signal"] == "promo").all()
    spike = out[out["is_spike"]]
    assert list(spike["product_key"]) == ["a"] and list(spike["exposure"]) == [40]
    with pytest.raises(ValueError):
        phase2.exposure_spikes(item, dim, signal="banner")


def test_filters_scope_spikes():
    n = 30
    a = np.full(n, 30)
    a[[5, 12]] = 90
    item, dim = _frames({"a": a, "b": [120] * n, "c": [150] * n})
    out = phase2.exposure_spikes(item, dim, Filters(start=date(2020, 11, 10), end=date(2020, 11, 20)))
    assert out["event_date"].min() >= pd.Timestamp("2020-11-10")
    assert out["event_date"].max() <= pd.Timestamp("2020-11-20")
    assert set(out.loc[out["is_spike"], "event_date"]) == {pd.Timestamp(START) + pd.Timedelta(days=12)}
    assert phase2.exposure_spikes(item, dim, Filters(devices=("mobile",))).empty
    assert phase2.exposure_spikes(item, dim, Filters(categories=("Bags",))).empty


# --------------------------------------------------------------------------- pair tests


def test_no_dip_when_conversion_is_unchanged():
    item, dim = _flat_category()
    t = phase2.cannibalization_tests(item, dim, _spikes("a", [3, 9, 15, 21, 27]))
    assert tuple(t.columns) == phase2.TEST_COLUMNS
    assert len(t) == 5 * 4  # every ordered pair of the category's 5 products
    r = _pair(t, "a", "b")
    assert bool(r["tested"]) and r["status"] == phase2.NO_EVIDENCE
    # Local baseline: non-spike days within +/-7 days of a spike -> days 0..34 minus 5 spikes.
    assert (r["n_spike_days"], r["n_other_days"]) == (5, 30)
    assert (r["O_s"], r["O_n"]) == (40, 240)
    assert r["E_s"] == pytest.approx(40.0) and r["E_n"] == pytest.approx(240.0)
    assert r["rr"] == pytest.approx(1.0)
    assert r["rr_low"] < 1 < r["rr_high"]
    # Pairs whose A never spiked are guarded out.
    assert (t.loc[t["product_a"] != "a", "status"] == phase2.INSUFFICIENT).all()


def test_dip_statistics_match_scipy():
    item, dim = _flat_category()
    spike_idx = [3, 9, 15, 21, 27]
    dates = {pd.Timestamp(START) + pd.Timedelta(days=i) for i in spike_idx}
    on = item["event_date"].isin(dates) & (item["product_key"] == "b")
    item.loc[on, "purchases"] = 2  # B sells 2 instead of 8 on A's spike days
    t = phase2.cannibalization_tests(item, dim, _spikes("a", spike_idx))
    r = _pair(t, "a", "b")

    k, m = 10, 240  # O_s, O_n (30 local non-spike days)
    e_s, e_n = 40.0, 240.0  # 400 views x the peers' 2% rate
    assert (r["O_s"], r["O_n"]) == (k, m)
    assert r["E_s"] == pytest.approx(e_s) and r["E_n"] == pytest.approx(e_n)
    assert r["rr"] == pytest.approx((k / e_s) / (m / e_n))
    assert r["pi0"] == pytest.approx(e_s / (e_s + e_n))
    assert r["p_value"] == pytest.approx(stats.binomtest(k, k + m, e_s / (e_s + e_n), alternative="less").pvalue)
    ci = stats.binomtest(k, k + m).proportion_ci(confidence_level=0.95, method="exact")
    to_rr = lambda pi: pi / (1 - pi) * e_n / e_s  # noqa: E731
    assert r["rr_low"] == pytest.approx(to_rr(ci.low)) and r["rr_high"] == pytest.approx(to_rr(ci.high))
    # Peers (p1-p3) sell 30 a day: 150 on spike days, 900 on the 30 local other days.
    fisher = stats.fisher_exact([[k, m], [150, 900]], alternative="less").pvalue
    assert r["fisher_p"] == pytest.approx(fisher)
    assert r["status"] == phase2.LIKELY and r["rr"] <= config.CANNIBAL["rr_flag"]
    assert "B sold 10" in r["reason"] and "spike days" in r["reason"]
    assert t.iloc[0]["status"] == phase2.LIKELY  # flagged pairs sort first
    assert phase2.conflict_pairs(t) == {frozenset(("a", "b"))}


def test_local_baseline_ignores_b_seasonality():
    """B converts 3x early in the season, far from A's late spikes: not a dip near the spikes."""
    item, dim = _flat_category(days=60)
    early = item["event_date"] < pd.Timestamp(START) + pd.Timedelta(days=30)
    item.loc[early & (item["product_key"] == "b"), "purchases"] = 24
    spike_idx = [45, 50, 55]
    local = _pair(phase2.cannibalization_tests(item, dim, _spikes("a", spike_idx)), "a", "b")
    assert local["n_other_days"] == 60 - 38 - 3  # days 38..59 minus the spikes
    assert local["rr"] == pytest.approx(1.0) and local["status"] == phase2.NO_EVIDENCE
    assert "nearby days" in local["reason"]
    # A season-wide baseline would read B's early peak as a dip on the spike days.
    wide = _pair(phase2.cannibalization_tests(item, dim, _spikes("a", spike_idx), local_days=100), "a", "b")
    assert wide["rr"] < config.CANNIBAL["rr_flag"]
    narrow = _pair(phase2.cannibalization_tests(item, dim, _spikes("a", spike_idx), local_days=1), "a", "b")
    assert narrow["status"] == phase2.INSUFFICIENT and "within 1 day of" in narrow["reason"]


def test_category_shock_on_spike_days_is_absorbed():
    """Every product converts 2x on A's spike days: B's rate relative to its peers is unchanged."""
    item, dim = _flat_category()
    spike_idx = [3, 9, 15, 21, 27]
    dates = {pd.Timestamp(START) + pd.Timedelta(days=i) for i in spike_idx}
    on = item["event_date"].isin(dates)
    item.loc[on, "purchases"] *= 2
    r =_pair(phase2.cannibalization_tests(item, dim, _spikes("a", spike_idx)), "a", "b")
    assert r["rr"] == pytest.approx(1.0)
    assert r["status"] == phase2.NO_EVIDENCE


@pytest.mark.parametrize(
    ("kwargs", "spike_idx", "phrase"),
    [
        ({}, [3, 9], "spike day"),
        ({"min_other_days": 50}, [3, 9, 15], "non-spike day"),
        ({"min_purchases": 10_000}, [3, 9, 15], "purchase"),
        ({"min_expected": 1e6}, [3, 9, 15], "expected"),
        ({"min_spike_views": 10**7}, [3, 9, 15], "views on"),
    ],
)
def test_insufficient_data_guards(kwargs, spike_idx, phrase):
    item, dim = _flat_category()
    t = phase2.cannibalization_tests(item, dim, _spikes("a", spike_idx), **kwargs)
    r = _pair(t, "a", "b")
    assert r["status"] == phase2.INSUFFICIENT and not bool(r["tested"])
    assert np.isnan(r["rr"]) and np.isnan(r["p_value"]) and np.isnan(r["q_value"])
    assert phrase in r["reason"]


def test_default_guards_on_a_thin_peer():
    """B with ~0.4 purchases a day and 5 spike days fails the default purchase guards."""
    item, dim = _flat_category(b_views=20, b_rate=0.02)
    r = _pair(phase2.cannibalization_tests(item, dim, _spikes("a", [3, 9, 15, 21, 27])), "a", "b")
    assert r["status"] == phase2.INSUFFICIENT
    assert r["n"] < config.CANNIBAL["min_purchases"]


def test_top_k_limits_the_pairs():
    item, dim = _flat_category()
    t = phase2.cannibalization_tests(item, dim, _spikes("a", [3, 9, 15]), top_k=3)
    assert set(t["product_a"]) | set(t["product_b"]) == {"p1", "p2", "p3"}  # the 3 most viewed
    assert len(t) == 6


def test_bh_q_values_match_scipy():
    rng = np.random.default_rng(3)
    n = 45
    views, purchases, cats = {}, {}, {}
    for c, cat in enumerate(("Apparel", "Bags", "Drinkware")):
        for j in range(5):
            key = f"{cat[:3].lower()}{j}"
            v = rng.poisson(300 + 150 * j, n)
            views[key], purchases[key], cats[key] = v, rng.binomial(v, 0.015 + 0.003 * c), cat
    item, dim = _frames(views, purchases, category=cats)
    spikes = pd.concat(
        [_spikes(k, sorted(rng.choice(n, size=6, replace=False).tolist())) for k in views], ignore_index=True
    )
    t = phase2.cannibalization_tests(item, dim, spikes, min_spike_views=0)
    tested = t[t["tested"]]
    assert len(tested) >= 30
    assert np.allclose(tested["q_value"], stats.false_discovery_control(tested["p_value"].to_numpy(), method="bh"))
    assert t.loc[~t["tested"], "q_value"].isna().all()
    assert set(t["status"]) <= set(phase2.STATUSES)


def _simulate(rng: np.random.Generator, runs: int, *, b_dip: float) -> tuple[int, int]:
    """(tested runs, runs with p < 0.05) for A -> B in small random categories.

    A's views quadruple on 6 random days of 45; every product converts at 1%, except that B
    converts at ``b_dip`` x 1% on A's spike days. Spikes are detected, not given.
    """
    n_days, hits, tested = 45, 0, 0
    lam = {"a": 60, "b": 300, "p1": 400, "p2": 400, "p3": 400, "p4": 400}
    for _ in range(runs):
        spike_idx = rng.choice(n_days, size=6, replace=False)
        views, purchases = {}, {}
        for key, base in lam.items():
            mu = np.full(n_days, float(base)) * rng.lognormal(0.0, 0.1, n_days)
            rate = np.full(n_days, 0.01)
            if key == "a":
                mu[spike_idx] *= 4
            if key == "b":
                rate[spike_idx] *= b_dip
            v = rng.poisson(mu)
            views[key], purchases[key] = v, rng.binomial(v, rate)
        item, dim = _frames(views, purchases)
        r = _pair(phase2.cannibalization_tests(item, dim, phase2.exposure_spikes(item, dim)), "a", "b")
        if r["tested"]:
            tested += 1
            hits += bool(r["p_value"] < 0.05)
    return tested, hits


def test_null_simulation_false_positive_rate():
    """200 small datasets with no effect: p < 0.05 for the A -> B pair in about 5% of them."""
    tested, hits = _simulate(np.random.default_rng(2026), 200, b_dip=1.0)
    assert tested >= 150
    assert hits / tested <= 0.10


def test_simulated_dip_is_usually_detected():
    tested, hits = _simulate(np.random.default_rng(11), 40, b_dip=0.4)
    assert tested >= 30
    assert hits / tested >= 0.6


def test_empty_inputs_return_typed_empty_frames():
    item, dim = _flat_category()
    empty_item = item.iloc[0:0]
    for out, cols in (
        (phase2.exposure_spikes(empty_item, dim), phase2.SPIKE_COLUMNS),
        (phase2.cannibalization_tests(empty_item, dim, None), phase2.TEST_COLUMNS),
        (phase2.cannibalization_tests(item, dim, item.iloc[0:0]), phase2.TEST_COLUMNS),
    ):
        assert tuple(out.columns) == cols
    no_spikes = phase2.cannibalization_tests(item, dim, pd.DataFrame())
    assert (no_spikes["status"] == phase2.INSUFFICIENT).all()
    assert phase2.conflict_pairs(None) == set() and phase2.conflict_pairs(no_spikes) == set()


# --------------------------------------------------------------------------- trends


BASE, FOCUS = config.PERIODS["baseline"], config.PERIODS["december"]
SEASON = pd.date_range("2020-11-01", "2020-12-31", freq="D")


def _trend_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flat on-site views for Bags and Office; Google interest +60% in December for 'backpack'."""
    n = len(SEASON)
    start = SEASON[0].date()
    views = {"bag1": [80] * n, "bag2": [40] * n, "bag3": [30] * n, "note1": [100] * n, "note2": [60] * n}
    cats = {"bag1": "Bags", "bag2": "Bags", "bag3": "Bags", "note1": "Office & Stationery", "note2": "Office & Stationery"}
    item, dim = _frames(views, category=cats, start=start)
    dec = (SEASON >= pd.Timestamp(FOCUS.start)) & (SEASON <= pd.Timestamp(FOCUS.end))
    rows = []
    for kw, group, base_level, rise in (
        ("backpack", "Bags", 50.0, 1.6),
        ("notebook", "Office & Stationery", 60.0, 1.0),
        ("tote bag", "Bags", 2.0, 3.0),  # far too low to read
    ):
        interest = np.where(dec, base_level * rise, base_level)
        rows.append(pd.DataFrame({"date": SEASON.date, "keyword": kw, "group_type": "category", "group_value": group,
                                  "geo": "US", "interest": interest, "is_partial": False}))
    trends = pd.concat(rows, ignore_index=True)
    km = pd.DataFrame(
        {
            "group_type": ["category", "category", "category"],
            "group_value": ["Bags", "Office & Stationery", "Bags"],
            "keyword": ["backpack", "notebook", "tote bag"],
            "notes": ["", "", ""],
        }
    )
    return trends, item, dim, km


def _scored() -> pd.DataFrame:
    """A hand-made products frame with the columns trending_alerts reads."""
    return pd.DataFrame(
        {
            "product_key": ["bag1", "bag2", "bag3", "note1", "note2"],
            "item_name": ["Bag One", "Bag Two", "Bag Three", "Note One", "Note Two"],
            "category_group": ["Bags", "Bags", "Bags", "Office & Stationery", "Office & Stationery"],
            "brand": "Google",
            "views": [4880, 2440, 1830, 6100, 3660],
            "purchases": [20, 30, 20, 30, 40],
            "prior_mean": [0.007, 0.007, 0.007, 0.007, 0.007],
            "cr_adj": [0.009, 0.010, 0.0095, 0.006, 0.010],
            "lift_per_1k": [3.0, 4.0, 3.5, 0.0, 4.0],
            "eligible": True,
        }
    )


def test_trending_alert_flags_the_rising_group():
    trends, item, dim, km = _trend_frames()
    al = phase2.trending_alerts(trends, item, dim, km, BASE, FOCUS, Filters(), _scored(), 4000.0)
    assert tuple(al.columns) == phase2.ALERT_COLUMNS
    bags = al.set_index("group").loc["Bags"]
    office = al.set_index("group").loc["Office & Stationery"]
    assert bags["M_ext"] == pytest.approx(1.6)  # the unreliable 'tote bag' row is left out
    assert bags["interest_base"] == pytest.approx(50.0)
    assert bags["M_site"] == pytest.approx(1.0) and bags["gap"] == pytest.approx(1.6)
    assert bags["views_base"] == 150 * 22 and bags["days_base"] == 22 and bags["days_focus"] == 17
    assert bool(bags["reliable"]) and bool(bags["flagged"])
    # bag1 is above the high-traffic line; bag2 and bag3 rank by lift per 1k views.
    assert bags["promote_ahead"] == ["bag2", "bag3"] and bags["promote_names"] == ["Bag Two", "Bag Three"]
    assert not bool(office["flagged"]) and office["promote_ahead"] == []
    assert office["M_ext"] == pytest.approx(1.0)
    assert bags["m_ext_rel"] == pytest.approx(1.6) and office["m_ext_rel"] == pytest.approx(1 / 1.6)
    assert al.iloc[0]["group"] == "Bags"  # flagged first
    assert "Promote ahead" in bags["reason"]


def test_trending_alert_guards():
    trends, item, dim, km = _trend_frames()
    al = phase2.trending_alerts(trends, item, dim, km, "baseline", "december", None, _scored(), 4000.0, min_base_views=10**6)
    assert not al["flagged"].any()
    assert "baseline views" in al.set_index("group").loc["Bags", "reason"]

    only_low = trends[trends["keyword"] == "tote bag"]
    al = phase2.trending_alerts(only_low, item, dim, km, "baseline", "december", None, _scored(), 4000.0)
    row = al.set_index("group").loc["Bags"]
    assert not bool(row["reliable"]) and not bool(row["flagged"])
    assert "too low" in row["reason"]


def test_trending_alert_needs_to_beat_the_holiday_tide():
    """Every category's interest rises 60%: Bags passes the old rules but only matches the tide."""
    trends, item, dim, km = _trend_frames()
    dec = pd.to_datetime(trends["date"]).between(pd.Timestamp(FOCUS.start), pd.Timestamp(FOCUS.end))
    trends.loc[dec & (trends["keyword"] == "notebook"), "interest"] = 60.0 * 1.6
    al = phase2.trending_alerts(trends, item, dim, km, BASE, FOCUS, Filters(), _scored(), 4000.0).set_index("group")
    bags = al.loc["Bags"]
    assert bags["M_ext"] == pytest.approx(1.6) and bags["gap"] == pytest.approx(1.6)
    assert bags["m_ext_rel"] == pytest.approx(1.0)
    assert not al["flagged"].any()
    assert "holiday tide" in bags["reason"]
    # With no other category to compare against, the benchmark is skipped (NaN), not failed.
    solo = phase2.trending_alerts(trends[trends["keyword"] != "notebook"], item, dim, km, BASE, FOCUS, Filters(),
                                  _scored(), 4000.0).set_index("group")
    assert np.isnan(solo.loc["Bags", "m_ext_rel"]) and bool(solo.loc["Bags", "flagged"])


def test_trend_index_is_100_in_the_base_period():
    trends, item, dim, km = _trend_frames()
    idx = phase2.trend_index(trends, item, dim, km, BASE, Filters())
    assert tuple(idx.columns) == phase2.TREND_INDEX_COLUMNS
    assert set(idx["keyword"]) == {"backpack", "notebook", "tote bag"}
    assert (idx["week"].dt.dayofweek == 6).all()  # W-SUN labels are Sundays
    bp = idx[idx["keyword"] == "backpack"].set_index("week")
    assert bp.loc[pd.Timestamp("2020-11-15"), "google_index"] == pytest.approx(100.0)
    assert bp.loc[pd.Timestamp("2020-12-13"), "google_index"] == pytest.approx(160.0)
    assert np.allclose(bp["site_views_index"], 100.0)
    assert bp.loc[pd.Timestamp("2020-11-01"), "n_days"] == 1  # Nov 1, 2020 was a Sunday


def test_trend_partial_points_and_keyword_map_spelling():
    trends, item, dim, km = _trend_frames()
    trends = trends.assign(group_value="ignored", keyword=trends["keyword"].str.upper())
    trends.loc[trends["date"] == date(2020, 12, 31), "is_partial"] = True
    idx = phase2.trend_index(trends, item, dim, km, BASE)
    assert set(idx["group"]) == {"Bags", "Office & Stationery"}  # the map wins over stored groups
    assert set(idx["keyword"]) == {"backpack", "notebook", "tote bag"}


def test_empty_trends_path_never_raises():
    _, item, dim, km = _trend_frames()
    for trends in (None, pd.DataFrame(), pd.DataFrame(columns=["date", "keyword", "interest"])):
        idx = phase2.trend_index(trends, item, dim, km, BASE)
        al = phase2.trending_alerts(trends, item, dim, km, BASE, FOCUS, Filters(), _scored(), 4000.0)
        assert idx.empty and tuple(idx.columns) == phase2.TREND_INDEX_COLUMNS
        assert al.empty and tuple(al.columns) == phase2.ALERT_COLUMNS
    ll = phase2.lead_lag(phase2.trend_index(None, item, dim, km, BASE))
    assert ll.empty and tuple(ll.columns) == phase2.LEAD_LAG_COLUMNS
    trends, *_ = _trend_frames()
    assert phase2.trend_index(trends, item.iloc[0:0], dim, km, BASE).empty
    assert phase2.trending_alerts(trends, item, dim.iloc[0:0], None, BASE, FOCUS).empty


def test_lead_lag_finds_a_one_week_lead():
    rng = np.random.default_rng(5)
    weeks = pd.date_range("2020-11-08", periods=13, freq="W-SUN")
    google = 50 * np.cumprod(1 + rng.normal(0, 0.1, len(weeks)))
    growth = np.r_[1.0, google[1:] / google[:-1]]
    site = 1000 * np.cumprod(np.r_[1.0, growth[:-1]])  # site's % change = Google's, one week later
    idx = pd.DataFrame(
        {"group_type": "category", "group": "Bags", "keyword": "backpack", "week": weeks, "n_days": 7,
         "interest": google, "site_views": site}
    )
    ll = phase2.lead_lag(idx, max_lag=3)
    assert tuple(ll.columns) == phase2.LEAD_LAG_COLUMNS
    assert list(ll["lag"]) == [0, 1, 2, 3]
    assert ll.set_index("lag").loc[1, "corr"] == pytest.approx(1.0)
    assert ll.set_index("lag").loc[1, "n_weeks"] == 11
    assert abs(ll.set_index("lag").loc[0, "corr"]) < 0.9


def test_load_keyword_map(tmp_path):
    path = tmp_path / "kw.csv"
    path.write_text(
        "group_type,group_value,keyword,notes\nCategory , Bags ,backpack,x\ncategory,Bags,Backpack,dup\n"
        "category,,tote bag,no group\nbrand,YouTube,youtube merch,\n",
        encoding="utf-8",
    )
    km = phase2.load_keyword_map(path)
    assert list(km.columns) == ["group_type", "group_value", "keyword", "notes"]
    assert km[["group_type", "group_value", "keyword"]].values.tolist() == [
        ["category", "Bags", "backpack"],
        ["brand", "YouTube", "youtube merch"],
    ]
    assert phase2.load_keyword_map(tmp_path / "missing.csv").empty
    assert phase2.load_keyword_map(None).empty


def test_periods_accept_keys_or_objects():
    trends, item, dim, km = _trend_frames()
    by_key = phase2.trending_alerts(trends, item, dim, km, "baseline", "december")
    by_obj = phase2.trending_alerts(trends, item, dim, km, BASE, Period("dec", "Dec", FOCUS.start, FOCUS.end))
    assert np.allclose(by_key["gap"], by_obj["gap"])
    with pytest.raises(ValueError):
        phase2.trend_index(trends, item, dim, km, "not_a_period")

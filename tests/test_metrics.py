"""Metric formulas on small hand-made frames, plus a smoke test on the real snapshots."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ga4app import config, contracts, metrics
from ga4app.metrics import Filters, Thresholds
from ga4app.types import PRODUCT_COLUMNS

D = pd.Timestamp


def _products(rows: list[tuple], cols: tuple[str, ...] = ("product_key", "category_group", "views", "purchases")) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(cols))


def _daily(rows: list[tuple]) -> pd.DataFrame:
    """item_daily-shaped rows: (date, device, product_key, views, purchases[, revenue])."""
    recs = []
    for r in rows:
        day, dev, key, v, p = r[:5]
        rev = r[5] if len(r) > 5 else 10.0 * p
        recs.append(
            {"event_date": D(day), "device_category": dev, "product_key": key, "views": v, "purchases": p,
             "units": p, "promo_views": 0, "revenue_usd": float(rev)}
        )
    return pd.DataFrame(recs)


def _dim(cats: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_key": list(cats),
            "item_name": [k.upper() for k in cats],
            "category_group": list(cats.values()),
            "brand": "Google",
        }
    )


# --------------------------------------------------------------------------- pooled rate + Wilson


def test_pooled_rate_ignores_products_without_views():
    prods = _products([("a", "Apparel", 100, 1), ("b", "Apparel", 300, 2), ("c", "Apparel", 0, 5)])
    assert metrics.pooled_rate(prods) == pytest.approx(3 / 400)
    assert metrics.thresholds(prods, min_views=0).conversion == pytest.approx(3 / 400)
    assert metrics.pooled_rate(prods.iloc[:0]) == 0.0


def test_wilson_known_values():
    lo, hi = metrics.wilson_ci(5, 100)
    assert float(lo) == pytest.approx(0.0215, abs=5e-4)
    assert float(hi) == pytest.approx(0.1118, abs=5e-4)
    lo, hi = metrics.wilson_ci([0, 5, 3], [10, 100, 0])
    assert lo[0] == 0.0 and 0 < hi[0] < 1
    assert np.isnan(lo[2]) and np.isnan(hi[2])


# --------------------------------------------------------------------------- empirical Bayes


def _beta_binomial(m_true: float, seed: int, n_per_cat: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cat, mu in (("Apparel", 0.004), ("Drinkware", 0.012)):
        views = np.round(np.exp(rng.normal(np.log(8000), 0.8, n_per_cat))).astype(int) + 100
        theta = rng.beta(m_true * mu, m_true * (1 - mu), n_per_cat)
        purchases = rng.binomial(views, theta)
        rows += [(f"{cat}-{i}", cat, int(v), int(p)) for i, (v, p) in enumerate(zip(views, purchases))]
    return _products(rows)


@pytest.mark.parametrize("m_true", [400.0, 3000.0])
def test_method_of_moments_recovers_m(m_true):
    m_hat = metrics.estimate_prior_strength(_beta_binomial(m_true, seed=11))
    assert m_true / 2 <= m_hat <= m_true * 2
    if m_true == 400.0:
        assert m_hat == pytest.approx(m_true, rel=0.30)


def test_prior_strength_bounds():
    lo, hi = config.SHRINK_M_BOUNDS
    # No spread beyond the category means at all: tau^2 <= 0 -> upper bound.
    exact = _products([(f"p{i}", "Apparel", 1000 * (i + 1), 10 * (i + 1)) for i in range(30)])
    assert metrics.estimate_prior_strength(exact) == hi
    # Wild spread -> clamped at the lower bound.
    wild = _products([(f"p{i}", "Apparel", 1000, 0 if i % 2 else 400) for i in range(30)])
    assert metrics.estimate_prior_strength(wild) == lo
    # Products under min_views are ignored; with fewer than 2 left there is nothing to estimate.
    assert metrics.estimate_prior_strength(_products([("a", "Apparel", 50, 1), ("b", "Apparel", 5000, 30)])) == hi


def test_shrinkage_is_monotone_in_purchases():
    rows = [(f"p{k}", "Apparel", 500, k) for k in range(0, 41, 4)]
    out = metrics.shrink(_products(rows), m=300.0)
    for col in ("cr_adj", "ci_low", "ci_high"):
        assert np.all(np.diff(out[col].to_numpy()) > 0), col
    # Every posterior mean sits between the raw rate and the prior mean.
    raw, prior, adj = out["cr_raw"].to_numpy(), out["prior_mean"].to_numpy(), out["cr_adj"].to_numpy()
    assert np.all((adj - np.minimum(raw, prior) >= -1e-12) & (np.maximum(raw, prior) - adj >= -1e-12))
    # A stronger prior pulls harder towards the prior mean.
    strong = metrics.shrink(_products(rows), m=5000.0)
    assert np.all(np.abs(strong["cr_adj"] - prior) <= np.abs(out["cr_adj"] - prior) + 1e-12)


def test_prior_mean_uses_category_rate_only_for_large_categories():
    rows = [(f"a{i}", "Apparel", 1000, 10) for i in range(20)] + [(f"d{i}", "Drinkware", 1000, 40) for i in range(5)]
    out = metrics.shrink(_products(rows), m=200.0)
    p_bar = (20 * 10 + 5 * 40) / 25_000
    assert out.loc[out["category_group"] == "Apparel", "prior_mean"].eq(0.01).all()
    assert out.loc[out["category_group"] == "Drinkware", "prior_mean"].to_numpy() == pytest.approx([p_bar] * 5)


def test_confidence_calls():
    rows = [("hi", "Apparel", 20_000, 400), ("lo", "Apparel", 20_000, 5), ("few", "Apparel", 50, 0)]
    rows += [(f"f{i}", "Apparel", 2000, 10) for i in range(5)]
    out = metrics.shrink(_products(rows), m=200.0).set_index("product_key")
    assert out.at["hi", "confidence"] == "above"
    assert out.at["lo", "confidence"] == "below"
    assert out.at["few", "confidence"] == "uncertain"
    assert (out["ci_low"] <= out["cr_adj"]).all() and (out["cr_adj"] <= out["ci_high"]).all()


def test_purchases_exceed_views_flag_and_clip():
    rows = [("gift", "Apparel", 10, 30), ("ghost", "Apparel", 0, 3), ("ok", "Apparel", 5000, 25)]
    m = 200.0
    out = metrics.shrink(_products(rows), m=m).set_index("product_key")
    assert out["purchases_exceed_views"].to_dict() == {"gift": True, "ghost": True, "ok": False}
    mu = out.at["gift", "prior_mean"]
    # V_eff = max(V, P): no negative failures enter the posterior.
    assert out.at["gift", "post_b"] == pytest.approx(m * (1 - mu))
    assert out.at["gift", "cr_adj"] == pytest.approx((30 + m * mu) / (30 + m))
    for key in ("gift", "ghost"):
        assert 0.0 <= out.at[key, "ci_low"] <= out.at[key, "cr_adj"] <= out.at[key, "ci_high"] <= 1.0
    assert np.isnan(out.at["ghost", "cr_raw"])
    # The unviewed product's purchases stay out of p_bar.
    assert metrics.pooled_rate(out.reset_index()) == pytest.approx(55 / 5010)


# --------------------------------------------------------------------------- thresholds + quadrants


def test_thresholds_rules():
    prods = _products([(f"p{i}", "Apparel", v, 1) for i, v in enumerate([0, 50, 100, 200, 300, 400, 1000])])
    th = metrics.thresholds(prods, min_views=100)
    assert th.views == 300.0 and th.views_rule == "median"  # median of 100..1000 only
    assert th.conversion == pytest.approx(6 / 2050)  # all viewed products, eligible or not
    assert metrics.thresholds(prods, 100, "p75").views == pytest.approx(np.quantile([100, 200, 300, 400, 1000], 0.75))
    assert metrics.thresholds(prods, 100, "p90").views == pytest.approx(np.quantile([100, 200, 300, 400, 1000], 0.9))
    custom = metrics.thresholds(prods, 100, 750.0)
    assert custom.views == 750.0 and custom.views_rule == "custom"
    assert metrics.thresholds(prods, 5000).views == 5000.0  # nothing eligible: fall back to the floor
    with pytest.raises(ValueError):
        metrics.thresholds(prods, 100, "mode")


def test_quadrant_assignment_on_hand_made_points():
    th = Thresholds(views=1000.0, conversion=0.01)
    prods = pd.DataFrame(
        {
            "product_key": ["win", "waste", "gem", "low", "edge", "small", "none"],
            "views": [2000, 2000, 500, 500, 1000, 50, 0],
            "cr_raw": [0.02, 0.005, 0.02, 0.005, 0.01, 0.5, np.nan],
            "cr_adj": [0.005, 0.02, 0.005, 0.02, 0.01, 0.5, np.nan],
        }
    )
    out = metrics.classify(prods, th, min_views=100).set_index("product_key")
    assert out["quadrant"].to_dict() == {
        "win": "Winners",
        "waste": "Wasted exposure",
        "gem": "Hidden gems",
        "low": "Low priority",
        "edge": "Winners",  # V == T_v counts as high traffic, y == T_c as converting
        "small": "",
        "none": "",
    }
    assert out["eligible"].tolist() == [True, True, True, True, True, False, False]
    adj = metrics.classify(prods, th, y="cr_adj", min_views=100).set_index("product_key")
    assert adj.at["win", "quadrant"] == "Wasted exposure" and adj.at["low", "quadrant"] == "Hidden gems"
    with pytest.raises(ValueError):
        metrics.classify(prods, th, y="views")


def test_forgone_lift_and_action_codes():
    th = Thresholds(views=1000.0, conversion=0.01)
    prods = pd.DataFrame(
        {
            "product_key": ["w1", "w2", "x1", "x2", "g1", "g2", "g_below", "l1"],
            "quadrant": ["Winners", "Winners", "Wasted exposure", "Wasted exposure",
                         "Hidden gems", "Hidden gems", "Hidden gems", "Low priority"],
            "views": [4000, 3000, 10_000, 5000, 400, 600, 300, 200],
            "cr_adj": [0.020, 0.030, 0.004, 0.002, 0.018, 0.015, 0.012, 0.001],
            "confidence": ["above", "above", "below", "below", "above", "uncertain", "below", "below"],
            "revenue_usd": [800.0, 900.0, 400.0, 100.0, 60.0, 60.0, 30.0, 1.0],
        }
    )
    out = metrics.score_actions(prods, th, top_n_codes=1).set_index("product_key")
    v, cr = out["views"], out["cr_adj"]
    assert out["forgone_purchases"].to_numpy() == pytest.approx((v * np.maximum(0.01 - cr, 0)).to_numpy())
    assert out["lift_per_1k"].to_numpy() == pytest.approx((1000 * np.maximum(cr - 0.01, 0)).to_numpy())
    assert out.at["x1", "forgone_purchases"] == pytest.approx(60.0)
    assert out.at["g1", "lift_per_1k"] == pytest.approx(8.0)
    assert out["rpv"].to_numpy() == pytest.approx((out["revenue_usd"] / v).to_numpy())
    assert out["views_share"].sum() == pytest.approx(1.0)
    # Winners by cr_adj, gems by lift (confidence "below" excluded), wasted by forgone purchases.
    assert out.loc[["w2", "w1", "x1", "x2", "g1", "g2"], "action_rank"].tolist() == [1, 2, 1, 2, 1, 2]
    assert np.isnan(out.at["g_below", "action_rank"]) and np.isnan(out.at["l1", "action_rank"])
    assert out["code"].to_dict() == {"w1": "", "w2": "W1", "x1": "X1", "x2": "", "g1": "G1", "g2": "",
                                     "g_below": "", "l1": ""}
    swaps = metrics.swap_gain(out.reset_index(), n=2)
    assert swaps[["gem_key", "wasted_key"]].values.tolist() == [["g1", "x1"], ["g2", "x2"]]
    assert swaps["swap_gain_per_1k"].tolist() == pytest.approx([14.0, 13.0])


# --------------------------------------------------------------------------- scope filter + pipeline


def _scope_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = _daily(
        [
            ("2020-10-31", "desktop", "a", 100, 1),
            ("2020-11-01", "desktop", "a", 100, 1),
            ("2020-11-01", "mobile", "a", 50, 2),
            ("2020-11-22", "desktop", "b", 70, 0),
            ("2020-11-23", "mobile", "b", 30, 3),
            ("2020-11-22", "tablet", "c", 5, 1),
        ]
    )
    return daily, _dim({"a": "Apparel", "b": "Drinkware", "c": "Bags"})


def test_filter_daily_inclusive_edges_and_scope():
    daily, dim = _scope_daily()
    f = Filters(start=date(2020, 11, 1), end=date(2020, 11, 22))
    got = metrics.filter_daily(daily, dim, f)
    assert sorted(got["event_date"].dt.day.tolist()) == [1, 1, 22, 22]
    # A timestamp later on the end day still counts as that day.
    late = daily.assign(event_date=daily["event_date"] + pd.Timedelta(hours=15))
    assert len(metrics.filter_daily(late, dim, f)) == 4
    assert set(metrics.filter_daily(daily, dim, Filters(devices=("mobile",)))["product_key"]) == {"a", "b"}
    assert set(metrics.filter_daily(daily, dim, Filters(categories=("Drinkware", "Bags")))["product_key"]) == {"b", "c"}
    assert len(metrics.filter_daily(daily, dim, Filters(devices=("mobile",)), ignore_devices=True)) == len(daily) - 1
    assert len(metrics.filter_daily(daily, dim, f, ignore_dates=True)) == len(daily)
    # Plain datetime.date objects (raw parquet) work too.
    raw = daily.assign(event_date=[d.date() for d in daily["event_date"]])
    assert len(metrics.filter_daily(raw, dim, f)) == 4


def test_device_sums_equal_all_device_total():
    daily, dim = _scope_daily()
    f = Filters(start=date(2020, 11, 1), end=date(2020, 11, 30), devices=("mobile",))
    totals = metrics.product_table(metrics.filter_daily(daily, dim, f, ignore_devices=True), dim).set_index("product_key")
    matrix = metrics.device_matrix(daily, dim, ["a", "b", "c"], f, m=100.0)
    assert len(matrix) == 3 * len(config.DEVICES)
    sums = matrix.groupby("product_key")[["views", "purchases"]].sum()
    pd.testing.assert_frame_equal(sums, totals[["views", "purchases"]].astype("int64"), check_names=False, check_index_type=False)
    rate = matrix.drop_duplicates("device").set_index("device")["device_rate"]
    assert rate["mobile"] == pytest.approx(5 / 80) and rate["desktop"] == pytest.approx(1 / 170)
    assert (matrix["index"] - matrix["cr_adj_d"] / matrix["device_rate"]).abs().max() < 1e-12
    summary = metrics.device_summary(daily, dim, f).set_index("device")
    assert summary["views"].sum() == totals["views"].sum()


def test_compute_products_columns_and_zero_view_products():
    daily = _daily(
        [("2020-11-02", "desktop", f"p{i}", 200 + 100 * i, i % 4) for i in range(8)]
        + [("2020-11-02", "mobile", "ghost", 0, 2), ("2020-11-03", "mobile", "unknown", 300, 1)]
    )
    dim = _dim({f"p{i}": "Apparel" for i in range(8)} | {"ghost": "Drinkware"})
    prods, th = metrics.compute_products(daily, dim, Filters(min_views=100), m=150.0)
    assert list(prods.columns[: len(PRODUCT_COLUMNS)]) == list(PRODUCT_COLUMNS)
    assert prods["views"].is_monotonic_decreasing
    ghost = prods.set_index("product_key").loc["ghost"]
    assert not ghost["eligible"] and ghost["quadrant"] == "" and np.isnan(ghost["cr_raw"])
    assert ghost["purchases_exceed_views"] and ghost["code"] == ""
    unknown = prods.set_index("product_key").loc["unknown"]
    assert unknown["category_group"] == "Other" and unknown["item_name"] == "unknown"
    assert th.conversion == pytest.approx(metrics.pooled_rate(prods))
    assert set(prods.loc[prods["eligible"], "quadrant"]) <= set(config.QUADRANTS)
    # An empty scope returns an empty, fully-typed frame instead of raising.
    empty, _ = metrics.compute_products(daily, dim, Filters(start=date(2021, 1, 1), end=date(2021, 1, 2)), m=150.0)
    assert empty.empty and set(PRODUCT_COLUMNS) <= set(empty.columns)


# --------------------------------------------------------------------------- windows


def _two_window_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i in range(20):  # Apparel triples in the focus window; Drinkware stays flat
        rows += [("2020-11-10", "desktop", f"a{i}", 1000, 5), ("2020-11-25", "desktop", f"a{i}", 1000, 15)]
        rows += [("2020-11-10", "desktop", f"d{i}", 1000, 5), ("2020-11-25", "desktop", f"d{i}", 1000, 5)]
    rows += [("2020-11-10", "mobile", "b0", 400, 1), ("2020-11-25", "mobile", "b0", 400, 2)]
    dim = _dim({f"a{i}": "Apparel" for i in range(20)} | {f"d{i}": "Drinkware" for i in range(20)} | {"b0": "Bags"})
    return _daily(rows), dim


def test_period_comparison_lift_tests_and_site_rows():
    daily, dim = _two_window_daily()
    base, focus = config.PERIODS["baseline"], config.PERIODS["cyber_week"]
    pc = metrics.period_comparison(daily, dim, [base, focus], Filters()).set_index(["group", "period"])
    assert list(dict.fromkeys(pc.index.get_level_values(0))) == ["Apparel", "Drinkware", "Bags"]
    assert pc.at[("Apparel", "cyber_week"), "lift"] == pytest.approx(2.0)
    assert pc.at[("Apparel", "cyber_week"), "p_value"] < 1e-6
    assert pc.at[("Drinkware", "cyber_week"), "lift"] == pytest.approx(0.0)
    assert pc.at[("Drinkware", "cyber_week"), "p_value"] == pytest.approx(1.0)
    assert np.isnan(pc.at[("Apparel", "baseline"), "lift"]) and np.isnan(pc.at[("Apparel", "baseline"), "p_value"])
    # Bags has < 10 purchases: no test, flagged as low sample.
    assert np.isnan(pc.at[("Bags", "cyber_week"), "p_value"]) and pc.at[("Bags", "cyber_week"), "low_sample"]
    # Against the rest of the store, Apparel still rose and Drinkware fell behind.
    assert pc.at[("Apparel", "cyber_week"), "lift_vs_rest"] > 0.5
    assert pc.at[("Drinkware", "cyber_week"), "lift_vs_rest"] < 0
    lo, hi = pc.at[("Apparel", "cyber_week"), "ci_low"], pc.at[("Apparel", "cyber_week"), "ci_high"]
    assert lo < pc.at[("Apparel", "cyber_week"), "cr"] < hi
    site = metrics.period_comparison(daily, dim, [base, focus], Filters(), by=None)
    assert site["group"].unique().tolist() == [metrics.ALL_PRODUCTS]
    assert site["views"].tolist() == [40_400, 40_400] and site["purchases"].tolist() == [201, 402]
    assert site["lift_vs_rest"].isna().all()


def test_site_conversion_hand_made():
    site = pd.DataFrame(
        {
            "event_date": [D("2020-11-01"), D("2020-11-01"), D("2020-11-22"), D("2020-11-23"), D("2020-11-23")],
            "device_category": ["desktop", "desktop", "mobile", "desktop", "desktop"],
            "event_name": ["view_item", "purchase", "purchase", "view_item", "purchase"],
            "events": [1000, 10, 2, 500, 9],
            "users": 1,
            "sessions": 1,
        }
    )
    out = metrics.site_conversion(site, [config.PERIODS["baseline"], config.PERIODS["cyber_week"]]).set_index("period")
    assert out.at["baseline", "cr"] == pytest.approx(0.012) and out.at["cyber_week", "cr"] == pytest.approx(0.018)
    assert metrics.site_conversion(site, ["baseline"], devices=("mobile",)).at[0, "purchases"] == 2


def test_quadrant_movers_and_transition_counts():
    rows = []
    for i in range(10):
        v = 1000 + 200 * i
        rows += [("2020-11-10", "desktop", f"p{i}", v, 5), ("2020-11-25", "desktop", f"p{i}", v, 5)]
    rows += [("2020-11-10", "desktop", "riser", 3000, 3), ("2020-11-25", "desktop", "riser", 3000, 60)]
    rows += [("2020-11-10", "desktop", "gone", 3000, 10)]
    daily = _daily(rows)
    dim = _dim({f"p{i}": "Apparel" for i in range(10)} | {"riser": "Apparel", "gone": "Apparel"})
    mv = metrics.quadrant_movers(daily, dim, "baseline", "cyber_week", Filters(min_views=100), m=200.0)
    status = mv.set_index("product_key")["status"]
    assert status["riser"] == "Rising" and status["gone"] == "Insufficient data"
    assert set(status) <= set(metrics.MOVER_STATUSES)
    assert {"quadrant_base", "quadrant_focus", "cr_adj_base", "cr_adj_focus", "rel_change", "z"} <= set(mv.columns)
    counts = metrics.transition_counts(mv)
    assert counts.shape == (4, 4) and list(counts.index) == list(config.QUADRANTS)
    assert counts.to_numpy().sum() == int(((mv["quadrant_base"] != "") & (mv["quadrant_focus"] != "")).sum())


# --------------------------------------------------------------------------- devices


def test_device_lean_calls():
    rows = []
    for key, (om, od) in {"mob": (100, 40), "desk": (40, 100), "both": (50, 52), "thin": (3, 2)}.items():
        rows += [(key, "mobile", 5000, om, 0.01), (key, "desktop", 5000, od, 0.01), (key, "tablet", 100, 90, 0.01)]
    matrix = pd.DataFrame(rows, columns=["product_key", "device", "views", "purchases", "device_rate"])
    lean = metrics.device_lean(matrix).set_index("product_key")
    assert lean["lean"].to_dict() == {"mob": "Mobile-first", "desk": "Desktop-first", "both": "Both", "thin": "Not enough data"}
    assert lean.at["mob", "rr"] == pytest.approx(2.5)
    assert lean.at["mob", "p_value"] < 0.001 and lean.at["both", "p_value"] > 0.5
    assert lean.at["mob", "purchases_mobile"] == 100  # tablet never enters the call


# --------------------------------------------------------------------------- campaigns + slot plan


def test_campaign_scores_borrow_strength_and_site_uplift():
    rows = []
    for i in range(25):
        rows += [("2020-11-05", "desktop", f"a{i}", 2000, 10), ("2020-11-27", "desktop", f"a{i}", 400, 4)]
        rows += [("2020-11-05", "desktop", f"d{i}", 2000, 10), ("2020-11-27", "desktop", f"d{i}", 40, 0)]
    rows += [("2020-11-05", "desktop", "star", 2000, 60)]  # no Black Friday views at all
    dim = _dim({f"a{i}": "Apparel" for i in range(25)} | {f"d{i}": "Drinkware" for i in range(25)} | {"star": "Drinkware"})
    m = 300.0
    cs = metrics.campaign_scores(_daily(rows), dim, config.PERIODS["black_friday"], Filters(min_views=100), m)
    cs = cs.set_index("product_key")
    assert cs.at["a0", "uplift_source"] == "category"  # 10,000 window views
    assert cs.at["d0", "uplift_source"] == "site"  # 1,000 < CATEGORY_UPLIFT_MIN_VIEWS
    season_a, window_a = 350 / 60_000, 100 / 10_000
    assert cs.at["a0", "uplift"] == pytest.approx(window_a / season_a)
    # No window data: the score is the season rate times the uplift, not zero.
    star = cs.loc["star"]
    assert star["views_w"] == 0 and star["score"] == pytest.approx(star["season_cr_adj"] * star["uplift"])
    assert star["vs_avg"] > 0
    # With window data the score is a views-weighted blend of window rate and prior.
    for key in ("a0", "d0"):
        r = cs.loc[key]
        w = r["views_w"] / (r["views_w"] + m)
        assert r["score"] == pytest.approx(w * r["cr_raw_w"] + (1 - w) * r["prior_w"])
    assert cs["score_per_1k"].to_numpy() == pytest.approx(1000 * cs["score"].to_numpy())
    assert {"score", "score_per_1k", "vs_avg", "views_w", "purchases_w"} <= set(cs.columns)


def _scores(keys: list[str], *, ineligible: tuple[str, ...] = ()) -> pd.DataFrame:
    n = len(keys)
    return pd.DataFrame(
        {
            "product_key": keys,
            "item_name": [k.upper() for k in keys],
            "category_group": "Apparel",
            "score_per_1k": np.linspace(20, 10, n),
            "vs_avg": np.linspace(1, 0.5, n),
            "views_w": 100,
            "eligible": [k not in ineligible for k in keys],
        }
    )


def test_slot_plan_conflicts_exclude_and_repeats():
    keys = ["a", "b", "c", "d", "e"]
    # Mapping order is deliberately not config.CAMPAIGNS order.
    scores = {c: _scores(keys) for c in reversed(list(config.CAMPAIGNS))}
    slots = {c: 2 for c in config.CAMPAIGNS}
    bf, cm, dec = list(config.CAMPAIGNS)

    def picks(plan: pd.DataFrame) -> dict[str, list[str]]:
        return {c: plan.loc[plan["campaign"] == c, "product_key"].tolist() for c in config.CAMPAIGNS}

    plan = metrics.slot_plan(scores, slots)
    assert picks(plan) == {bf: ["a", "b"], cm: ["c", "d"], dec: ["e"]}
    assert plan["campaign"].tolist()[0] == bf and plan["slot"].tolist() == [1, 2, 1, 2, 1]
    assert {"campaign", "slot", "product_key", "item_name", "category_group", "score_per_1k", "vs_avg"} <= set(plan.columns)
    assert picks(metrics.slot_plan(scores, slots, allow_repeats=True)) == {bf: ["a", "b"], cm: ["a", "b"], dec: ["a", "b"]}
    conflict = metrics.slot_plan(scores, slots, conflicts={frozenset({"a", "b"})}, allow_repeats=True)
    assert picks(conflict)[bf] == ["a", "c"]
    assert picks(metrics.slot_plan(scores, slots, exclude={"a", "c"}))[bf] == ["b", "d"]
    gated = metrics.slot_plan({bf: _scores(keys, ineligible=("a",))}, {bf: 2})
    assert gated["product_key"].tolist() == ["b", "c"]
    assert metrics.slot_plan({}, slots).empty


# --------------------------------------------------------------------------- real snapshots (smoke)


@pytest.fixture(scope="module")
def real() -> dict:
    data_dir = config.DEFAULT_DATA_DIR
    if not (data_dir / "manifest.json").exists():
        pytest.skip("real snapshots not built (data/manifest.json missing)")
    return {name: contracts.read_snapshot(data_dir, name) for name in ("item_daily", "product_dim", "site_daily")}


def test_real_full_season_pipeline(real):
    item, dim = real["item_daily"], real["product_dim"]
    m = metrics.season_prior_strength(item, dim)
    lo, hi = config.SHRINK_M_BOUNDS
    assert lo <= m <= hi
    prods, th = metrics.compute_products(item, dim, Filters(), m)
    assert set(PRODUCT_COLUMNS) <= set(prods.columns) and len(prods) >= 300
    assert set(config.QUADRANTS) <= set(prods["quadrant"])
    assert 0.003 < th.conversion < 0.009
    assert not prods.loc[prods["views"] == 0, "eligible"].any()
    # Products seen often but never bought are the clearest Wasted exposure.
    never = prods[(prods["purchases"] == 0) & (prods["views"] >= th.views)]
    assert len(never) >= 1 and (never["quadrant"] == "Wasted exposure").all()
    top = never.iloc[0]
    assert top["forgone_purchases"] == pytest.approx(top["views"] * (th.conversion - top["cr_adj"]))


def test_real_site_conversion_matches_pipeline(real):
    keys = ("baseline", "cyber_week", "december", "full")
    out = metrics.site_conversion(real["site_daily"], [config.PERIODS[k] for k in keys])
    assert (out["cr"] * 100).round(2).tolist() == [1.18, 1.85, 1.70, 1.47]

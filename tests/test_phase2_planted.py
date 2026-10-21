"""Phase 2 against the effects planted by ``ga4app.synthetic`` (cannibal pair, trending group)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ga4app import config, phase2
from ga4app.types import Filters

synthetic = pytest.importorskip("ga4app.synthetic")


@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out: Path = tmp_path_factory.mktemp("syn")
    truth = synthetic.make_synthetic_snapshots(out)
    return {
        "truth": truth,
        "item": pd.read_parquet(out / "item_daily.parquet"),
        "dim": pd.read_parquet(out / "product_dim.parquet"),
        "trends": pd.read_parquet(out / "trends_daily.parquet"),
    }


@pytest.fixture(scope="module")
def tests_df(syn: dict) -> pd.DataFrame:
    spikes = phase2.exposure_spikes(syn["item"], syn["dim"], Filters())
    return phase2.cannibalization_tests(syn["item"], syn["dim"], spikes, Filters())


def _scored(item: pd.DataFrame, dim: pd.DataFrame, *, m: float = 1000.0, min_views: int = 200) -> pd.DataFrame:
    """A minimal stand-in for the metrics product frame (full season, all devices)."""
    p = item.groupby("product_key", as_index=False)[["views", "purchases"]].sum()
    p = p.merge(dim[["product_key", "item_name", "category_group", "brand"]], on="product_key")
    cat = p.groupby("category_group")[["purchases", "views"]].transform("sum")
    p["prior_mean"] = cat["purchases"] / cat["views"]
    p["cr_adj"] = (p["purchases"] + m * p["prior_mean"]) / (p["views"] + m)
    p_bar = p["purchases"].sum() / p["views"].sum()
    p["lift_per_1k"] = 1000 * np.maximum(p["cr_adj"] - p_bar, 0)
    p["eligible"] = p["views"] >= min_views
    return p


def _row(tests: pd.DataFrame, a: str, b: str) -> pd.Series:
    row = tests[(tests["product_a"] == a) & (tests["product_b"] == b)]
    assert len(row) == 1, f"pair {a} -> {b} was not tested"
    return row.iloc[0]


def test_planted_spike_days_are_found(syn: dict):
    spikes = phase2.exposure_spikes(syn["item"], syn["dim"])
    for pair in ("cannibal_pair", "null_pair"):
        truth = syn["truth"][pair]
        found = spikes.loc[spikes["is_spike"] & (spikes["product_key"] == truth["a"]), "event_date"]
        assert set(found.dt.strftime("%Y-%m-%d")) >= set(truth["spike_days"])
        assert len(found) <= len(truth["spike_days"]) + 2


def test_planted_cannibal_pair_is_flagged(syn: dict, tests_df: pd.DataFrame):
    cp = syn["truth"]["cannibal_pair"]
    r = _row(tests_df, cp["a"], cp["b"])
    assert r["category_group"] == cp["category_group"]
    assert r["status"] == phase2.LIKELY
    assert r["rr"] < 0.8 and r["q_value"] < 0.1
    assert r["rr_low"] <= cp["b_dip"] <= r["rr_high"]  # the planted 40% rate is inside the 95% interval
    assert r["fisher_p"] < 0.05
    assert frozenset((cp["a"], cp["b"])) in phase2.conflict_pairs(tests_df)


def test_null_pair_is_not_flagged(syn: dict, tests_df: pd.DataFrame):
    npair = syn["truth"]["null_pair"]
    r = _row(tests_df, npair["a"], npair["b"])
    assert bool(r["tested"])
    assert r["status"] not in (phase2.LIKELY, phase2.POSSIBLE)
    assert frozenset((npair["a"], npair["b"])) not in phase2.conflict_pairs(tests_df)


def test_category_filter_scopes_the_pairs(syn: dict):
    f = Filters(categories=("Drinkware",))
    t = phase2.cannibalization_tests(syn["item"], syn["dim"], phase2.exposure_spikes(syn["item"], syn["dim"], f), f)
    assert set(t["category_group"]) == {"Drinkware"}
    assert len(t) == config.CANNIBAL["top_k"] * (config.CANNIBAL["top_k"] - 1)


def test_planted_trending_group_is_flagged(syn: dict):
    truth, item, dim = syn["truth"], syn["item"], syn["dim"]
    products = _scored(item, dim)
    th_views = float(products.loc[products["eligible"], "views"].median())
    al = phase2.trending_alerts(
        syn["trends"], item, dim, truth["trends_keywords_csv"], "baseline", "december", Filters(), products, th_views
    ).set_index("group")
    trending, control = al.loc[truth["trending_group"]], al.loc[truth["control_group"]]
    assert bool(trending["flagged"]) and bool(trending["reliable"])
    assert trending["M_ext"] == pytest.approx(1.6, abs=0.12)
    assert trending["M_site"] == pytest.approx(1.0, abs=0.08)
    assert not bool(control["flagged"])
    assert trending["m_ext_rel"] >= config.TRENDING["gap_min"]  # beats the other categories' tide
    assert al["flagged"].sum() == 1  # the seasonal groups rise less than their on-site views

    promote = trending["promote_ahead"]
    assert promote, "the flagged group should have products to promote"
    picked = products.set_index("product_key").loc[promote]
    assert (picked["category_group"] == truth["trending_group"]).all()
    assert (picked["views"] < th_views).all() and (picked["cr_adj"] > picked["prior_mean"]).all()
    assert list(picked["lift_per_1k"]) == sorted(picked["lift_per_1k"], reverse=True)
    gems = set(truth["hidden_gems"]) & set(dim.loc[dim["category_group"] == truth["trending_group"], "product_key"])
    assert gems <= set(promote)


def test_trend_index_and_lead_lag_on_synthetic(syn: dict):
    truth = syn["truth"]
    idx = phase2.trend_index(syn["trends"], syn["item"], syn["dim"], truth["trends_keywords_csv"], "baseline")
    assert set(idx["group"]) == set(pd.read_csv(truth["trends_keywords_csv"])["group_value"])
    tg = idx[(idx["group"] == truth["trending_group"]) & (idx["week"] == pd.Timestamp("2020-12-13"))].iloc[0]
    assert tg["google_index"] == pytest.approx(160, abs=20)
    assert tg["site_views_index"] == pytest.approx(100, abs=15)
    ll = phase2.lead_lag(idx)
    assert set(ll["lag"]) == {0, 1, 2, 3}
    assert ll["corr"].dropna().between(-1, 1).all()


def test_no_trends_on_synthetic_never_raises(syn: dict):
    truth, item, dim = syn["truth"], syn["item"], syn["dim"]
    km = truth["trends_keywords_csv"]
    assert phase2.trending_alerts(None, item, dim, km, "baseline", "december", Filters(), _scored(item, dim)).empty
    assert phase2.trend_index(syn["trends"].iloc[0:0], item, dim, km, "baseline").empty

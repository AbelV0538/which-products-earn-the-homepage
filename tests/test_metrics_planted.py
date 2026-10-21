"""Metrics recover the effects planted by ``ga4app.synthetic`` (seed 7, 240 products)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

synthetic = pytest.importorskip("ga4app.synthetic")

from ga4app import config, contracts, metrics  # noqa: E402
from ga4app.metrics import Filters  # noqa: E402

BASELINE, CYBER_WEEK = config.PERIODS["baseline"], config.PERIODS["cyber_week"]


@pytest.fixture(scope="module")
def syn(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("syn")
    truth = synthetic.make_synthetic_snapshots(out)
    item = contracts.read_snapshot(out, "item_daily")
    dim = contracts.read_snapshot(out, "product_dim")
    m = metrics.season_prior_strength(item, dim)
    products, th = metrics.compute_products(item, dim, Filters(), m)
    return {"truth": truth, "item": item, "dim": dim, "m": m, "products": products, "th": th}


def test_prior_strength_recovers_planted_spread(syn):
    # Pinned gems / wasted products add spread on top of the Beta(1500) draw, so m comes out lower.
    assert synthetic.RATE_SPREAD_M / 2 <= syn["m"] <= synthetic.RATE_SPREAD_M * 2


@pytest.mark.parametrize("y", ["cr_raw", "cr_adj"])
def test_planted_gems_and_wasted_land_in_their_quadrants(syn, y):
    truth = syn["truth"]
    products, _ = metrics.compute_products(syn["item"], syn["dim"], Filters(), syn["m"], y=y)
    quad = products.set_index("product_key")["quadrant"]
    assert (quad.loc[truth["hidden_gems"]] == "Hidden gems").all()
    assert (quad.loc[truth["wasted_exposure"]] == "Wasted exposure").all()


def test_planted_products_dominate_the_action_lists(syn):
    truth, products = syn["truth"], syn["products"]
    by_key = products.set_index("product_key")
    assert (by_key.loc[truth["wasted_exposure"], "confidence"] == "below").all()
    assert (by_key.loc[truth["wasted_exposure"], "forgone_purchases"] > 0).all()
    codes = products.loc[products["code"] != "", ["code", "product_key"]].set_index("code")["product_key"]
    gem_codes = [c for c in codes.index if c.startswith("G")]
    assert sum(codes[c] in truth["hidden_gems"] for c in gem_codes) >= 3
    assert sum(codes[c] in truth["wasted_exposure"] for c in codes.index if c.startswith("X")) >= 2


def test_purchases_exceed_views_product_is_flagged_not_eligible(syn):
    by_key = syn["products"].set_index("product_key")
    for key in syn["truth"]["purchases_exceed_views"]:
        row = by_key.loc[key]
        assert row["purchases_exceed_views"] and not row["eligible"] and row["quadrant"] == ""
        assert 0 < row["cr_adj"] < 1


def test_planted_device_leans(syn):
    truth = syn["truth"]
    keys = [truth["mobile_first"], truth["desktop_first"]]
    lean = metrics.device_lean(metrics.device_matrix(syn["item"], syn["dim"], keys, Filters(), syn["m"]))
    lean = lean.set_index("product_key")["lean"]
    assert lean[truth["mobile_first"]] == "Mobile-first"
    assert lean[truth["desktop_first"]] == "Desktop-first"


def test_device_lean_rarely_fires_without_a_planted_lean(syn):
    planted = {syn["truth"]["mobile_first"], syn["truth"]["desktop_first"]}
    products = syn["products"]
    top = products[~products["product_key"].isin(planted)].nlargest(60, "views")["product_key"].tolist()
    lean = metrics.device_lean(metrics.device_matrix(syn["item"], syn["dim"], top, Filters(), syn["m"]))
    false_calls = lean["lean"].isin(["Mobile-first", "Desktop-first"]).sum()
    assert false_calls <= 0.15 * len(top)


def test_apparel_cyber_week_lift_is_significant_and_drinkware_is_not(syn):
    pc = metrics.period_comparison(syn["item"], syn["dim"], [BASELINE, CYBER_WEEK], Filters())
    focus = pc[pc["period"] == "cyber_week"].set_index("group")
    # Apparel: planted 1.8x on top of the store-wide surge.
    assert focus.at["Apparel", "p_value_vs_rest"] < 1e-4
    assert 0.4 < focus.at["Apparel", "lift_vs_rest"] < 1.3
    assert focus.at["Apparel", "lift"] > focus.at["Drinkware", "lift"]
    # Drinkware: planted 1.0x, i.e. it only rides the surge. Compare it with the other
    # un-boosted groups (with Apparel in "the rest", the rest is boosted).
    others = tuple(g for g in config.CATEGORY_GROUPS if g != "Apparel")
    pc2 = metrics.period_comparison(syn["item"], syn["dim"], [BASELINE, CYBER_WEEK], Filters(categories=others))
    drink = pc2[(pc2["period"] == "cyber_week") & (pc2["group"] == "Drinkware")].iloc[0]
    assert drink["p_value_vs_rest"] > 0.05
    assert abs(drink["lift_vs_rest"]) < 0.3


def test_apparel_products_rise_into_cyber_week(syn):
    mv = metrics.quadrant_movers(syn["item"], syn["dim"], BASELINE, CYBER_WEEK, Filters(), syn["m"])
    decided = mv[mv["status"] != "Insufficient data"]
    rising = decided.groupby("category_group")["status"].apply(lambda s: float((s == "Rising").mean()))
    assert rising["Apparel"] > 0.5 and rising["Apparel"] > 2 * rising["Drinkware"]
    counts = metrics.transition_counts(mv)
    assert counts.to_numpy().sum() == len(mv[(mv["quadrant_base"] != "") & (mv["quadrant_focus"] != "")])


def test_campaign_score_borrows_strength(syn):
    truth, m = syn["truth"], syn["m"]
    cs = metrics.campaign_scores(syn["item"], syn["dim"], config.PERIODS["cyber_monday"], Filters(), m)
    by_key = cs.set_index("product_key")
    # Two-day window: the planted gems barely sell, yet their season rate keeps them above average.
    gems = by_key.loc[truth["hidden_gems"]]
    assert gems["purchases_w"].sum() <= 3
    assert (gems["vs_avg"] > 0).sum() >= 4
    # Every score is a blend of the window rate and the season-based prior...
    seen = cs[cs["views_w"] > 0]
    lo = np.minimum(seen["cr_raw_w"], seen["prior_w"]) - 1e-12
    hi = np.maximum(seen["cr_raw_w"], seen["prior_w"]) + 1e-12
    assert ((seen["score"] >= lo) & (seen["score"] <= hi)).all()
    unseen = cs[cs["views_w"] == 0]
    assert np.allclose(unseen["score"], unseen["prior_w"])
    # ... which steadies it: far less spread than the raw window rates.
    elig = seen[seen["eligible"]]
    assert elig["score"].std() < 0.6 * elig["cr_raw_w"].std()


def test_campaign_uplift_follows_planted_holiday_effect(syn):
    cs = metrics.campaign_scores(syn["item"], syn["dim"], config.PERIODS["black_friday"], Filters(), syn["m"])
    uplift = cs.drop_duplicates("category_group").set_index("category_group")["uplift"]
    ratio = uplift["Apparel"] / uplift["Drinkware"]
    planted = syn["truth"]["holiday_uplift"]["Apparel"] / syn["truth"]["holiday_uplift"]["Drinkware"]
    assert planted / 1.6 < ratio < planted * 1.4


def test_slot_plan_keeps_the_cannibal_pair_apart(syn):
    pair = syn["truth"]["cannibal_pair"]
    scores = {
        name: metrics.campaign_scores(syn["item"], syn["dim"], key, Filters(), syn["m"])
        for name, key in config.CAMPAIGNS.items()
    }
    n = {name: 200 for name in config.CAMPAIGNS}
    conflict = frozenset({pair["a"], pair["b"]})
    plan = metrics.slot_plan(scores, n, conflicts={conflict}, allow_repeats=True)
    for _, picks in plan.groupby("campaign")["product_key"]:
        assert not conflict <= set(picks)
    free = metrics.slot_plan(scores, n, allow_repeats=True)
    assert any(conflict <= set(p) for _, p in free.groupby("campaign")["product_key"])
    assert isinstance(plan, pd.DataFrame) and plan["slot"].min() == 1

"""The synthetic snapshot set: realistic scale, planted effects visible in raw counts, determinism.

These are simple pooled ratios over the raw synthetic counts, deliberately independent of
``ga4app.metrics`` and ``ga4app.phase2``.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from ga4app import config, contracts
from ga4app.synthetic import MIN_PRODUCTS, make_synthetic_snapshots

TRUTH_KEYS = {
    "seed": int,
    "n_products": int,
    "hidden_gems": list,
    "wasted_exposure": list,
    "mobile_first": str,
    "desktop_first": str,
    "holiday_uplift": dict,
    "cannibal_pair": dict,
    "null_pair": dict,
    "trending_group": str,
    "control_group": str,
    "trends_keywords_csv": str,
}
PARQUET = ("item_daily", "product_dim", "promo_daily", "site_daily", "site_funnel", "trends_daily")
JSON = ("overview", "profile", "manifest", "trends_meta", "bq_proof")


@pytest.fixture(scope="module")
def frames(synthetic_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: contracts.read_snapshot(synthetic_dir, name) for name in PARQUET}


@pytest.fixture(scope="module")
def manifest(synthetic_dir: Path) -> dict:
    return contracts.read_json(synthetic_dir / "manifest.json")


@pytest.fixture(scope="module")
def per_product(frames) -> pd.DataFrame:
    return frames["item_daily"].groupby("product_key")[["views", "purchases"]].sum()


def _rate(df: pd.DataFrame) -> float:
    return float(df["purchases"].sum() / df["views"].sum())


def _site_rate(per_product: pd.DataFrame) -> float:
    return _rate(per_product)


def _in(dates: pd.Series, start: str, end: str) -> pd.Series:
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def _group_keys(frames, group: str) -> list[str]:
    dim = frames["product_dim"]
    return dim.loc[dim["category_group"] == group, "product_key"].tolist()


# --------------------------------------------------------------------------- contract shape


def test_truth_has_contract_keys(synthetic_truth: dict) -> None:
    for key, typ in TRUTH_KEYS.items():
        assert isinstance(synthetic_truth[key], typ), key
    assert synthetic_truth["seed"] == 7 and synthetic_truth["n_products"] == 240
    assert len(synthetic_truth["hidden_gems"]) == 6 and len(synthetic_truth["wasted_exposure"]) == 6
    assert synthetic_truth["holiday_uplift"] == {"Apparel": 1.8, "Drinkware": 1.0}
    assert set(synthetic_truth["cannibal_pair"]) == {"a", "b", "category_group", "spike_days", "b_dip"}
    assert synthetic_truth["cannibal_pair"]["b_dip"] == 0.4
    assert set(synthetic_truth["null_pair"]) == {"a", "b", "category_group", "spike_days"}
    assert synthetic_truth["trending_group"] != synthetic_truth["control_group"]
    roles = [
        *synthetic_truth["hidden_gems"],
        *synthetic_truth["wasted_exposure"],
        synthetic_truth["mobile_first"],
        synthetic_truth["desktop_first"],
        *(synthetic_truth[p][k] for p in ("cannibal_pair", "null_pair") for k in ("a", "b")),
    ]
    assert len(set(roles)) == len(roles), "planted roles must be disjoint"


def test_all_files_written_and_manifest_matches(synthetic_dir: Path, synthetic_truth: dict, manifest: dict) -> None:
    for name in PARQUET:
        assert (synthetic_dir / contracts.PARQUET_FILES[name]).exists(), name
    for name in JSON:
        assert (synthetic_dir / contracts.JSON_FILES[name]).exists(), name
    assert Path(synthetic_truth["trends_keywords_csv"]) == synthetic_dir / "trends_keywords.csv"
    assert manifest["source"] == "synthetic"
    assert manifest["schema_version"] == contracts.SCHEMA_VERSION
    assert manifest["view_metric"] == "view_events" and manifest["product_key"] == "name_key"
    assert manifest["synthetic_truth"] == json.loads(json.dumps(synthetic_truth))
    assert set(manifest["periods"]) == set(config.PERIODS)
    for name in PARQUET:
        entry = manifest["files"][name]
        assert entry["sha256"] == contracts.file_sha256(synthetic_dir / entry["file"])
        assert entry["rows"] > 0


def test_side_json_files(synthetic_dir: Path) -> None:
    overview = contracts.read_json(synthetic_dir / "overview.json")
    for key in (
        "event_count", "user_count", "day_count", "session_count", "date_min", "date_max", "matches_pitch",
        "events_by_device", "events_by_name", "top_countries", "tracking_gaps", "source_csv", "generated_at",
    ):  # fmt: skip
        assert key in overview, key
    assert overview["day_count"] == 92
    assert (overview["date_min"], overview["date_max"]) == ("2020-11-01", "2021-01-31")
    # add_to_cart is barely tracked before Nov 25, as in the real CSV.
    assert overview["tracking_gaps"]["add_to_cart"]["first_reliable_date"] == "2020-11-25"
    profile = contracts.read_json(synthetic_dir / "profile.json")
    assert profile["decisions"]["product_key"] == "name_key"
    assert profile["decisions"]["view_metric"] == "view_events"
    assert profile["decisions"]["featured_signal"] == "views_share"
    assert profile["decisions"]["notes"]
    meta = contracts.read_json(synthetic_dir / "trends_meta.json")
    assert meta["status"] == "ok" and meta["geo"] == "US" and len(meta["keywords"]) == 6
    proof = contracts.read_json(synthetic_dir / "bq_proof.json")
    assert proof["event_count"] == overview["event_count"] and proof["matches_pitch"] is False


# --------------------------------------------------------------------------- realism


def test_volumes_look_like_the_real_store(frames, per_product: pd.DataFrame) -> None:
    views = per_product["views"]
    assert 2_000 <= views.median() <= 3_100
    assert 12_000 <= views.quantile(0.9) <= 20_000
    assert 0.0048 <= _site_rate(per_product) <= 0.0062
    dev = frames["item_daily"].groupby("device_category")["views"].sum()
    share = dev / dev.sum()
    assert abs(share["desktop"] - 0.58) < 0.02 and abs(share["mobile"] - 0.40) < 0.02 and abs(share["tablet"] - 0.02) < 0.01
    dim = frames["product_dim"].set_index("product_key")
    other = views[dim.index[dim["category_group"] == "Other"]].sum() / views.sum()
    assert other <= 0.15
    item = frames["item_daily"]
    assert item["event_date"].nunique() == 92
    assert set(item["device_category"].unique()) == set(config.DEVICES)
    assert (item["revenue_usd"] >= 0).all()


def test_products_look_like_real_store_items(frames) -> None:
    dim, item = frames["product_dim"], frames["item_daily"]
    assert len(dim) == 240 and dim["product_key"].is_unique
    assert (dim["product_key"] == dim["item_name"].str.lower()).all()
    assert set(dim["brand"]) <= {"Google", "YouTube", "Android"}
    assert all(name.startswith(brand + " ") for name, brand in zip(dim["item_name"], dim["brand"]))
    assert set(dim["category_group"]) == set(config.CATEGORY_GROUPS)
    assert set(item["product_key"]) <= set(dim["product_key"])
    raw = dim["category_raw"].dropna()
    assert raw.str.startswith("Home/Apparel/").any() and (raw == "Home/Sale/").any()
    assert not dim["category_path"].str.startswith("Home").any()
    assert (dim.loc[dim["category_raw"] == "Home/Apparel/Men's / Unisex/", "category_path"] == "Apparel > Men's / Unisex").all()
    assert (dim["n_item_ids"] == dim["item_ids"].str.count(r"\|") + 1).all()
    unsold = dim["total_purchases"] == 0
    assert dim.loc[unsold, "median_price_usd"].isna().all()
    assert dim.loc[~unsold, "median_price_usd"].notna().all()


def test_site_tables_consistent_with_item_volumes(frames, synthetic_dir: Path) -> None:
    item, site, funnel = frames["item_daily"], frames["site_daily"], frames["site_funnel"]
    events = site.groupby("event_name")["events"].sum()
    assert item["views"].sum() / events["view_item"] == pytest.approx(11.4, rel=0.01)
    assert item["purchases"].sum() / events["purchase"] == pytest.approx(2.8, rel=0.02)
    assert events["page_view"] > events["scroll"] > events["view_item"] > events["purchase"]
    assert events["session_start"] == pytest.approx(0.92 * events["view_item"], rel=0.02)
    assert (site["users"] <= site["sessions"]).all() and (site["sessions"] <= site["events"]).all()
    assert len(funnel) == 92 * 3
    assert (funnel["view_sessions"] <= funnel["sessions"]).all()
    assert (funnel["view_purchase_sessions"] <= funnel["purchase_sessions"]).all()
    overview = contracts.read_json(synthetic_dir / "overview.json")
    assert overview["session_count"] == int(funnel["sessions"].sum())
    assert overview["event_count"] == int(events.sum())


def test_category_colors_follow_season_views(frames, manifest: dict) -> None:
    dim = frames["product_dim"].set_index("product_key")
    views = frames["item_daily"].groupby("product_key")["views"].sum()
    by_group = views.groupby(dim["category_group"]).sum().sort_values(ascending=False)
    colors = manifest["category_colors"]
    assert set(colors) == set(config.CATEGORY_GROUPS)
    top3 = list(by_group.index[:3])
    assert [colors[g] for g in top3] == list(config.SCATTER_CATEGORICAL)
    assert all(colors[g] == config.PALETTE["other"] for g in config.CATEGORY_GROUPS if g not in top3)


def test_purchases_exceed_views_product(per_product: pd.DataFrame, synthetic_truth: dict) -> None:
    for key in synthetic_truth["purchases_exceed_views"]:
        assert per_product.loc[key, "purchases"] > per_product.loc[key, "views"]


# --------------------------------------------------------------------------- planted effects


def test_hidden_gems_convert_far_above_the_site(per_product: pd.DataFrame, synthetic_truth: dict) -> None:
    gems = synthetic_truth["hidden_gems"]
    assert _rate(per_product.loc[gems]) >= 2.0 * _site_rate(per_product)
    rank = per_product["views"].rank(pct=True)
    assert (rank[gems] <= 0.40).all(), "hidden gems sit in the bottom 40% by views"
    assert (per_product.loc[gems, "views"] >= 400).all()


def test_wasted_exposure_converts_far_below_the_site(per_product: pd.DataFrame, synthetic_truth: dict) -> None:
    wasted = synthetic_truth["wasted_exposure"]
    assert _rate(per_product.loc[wasted]) <= 0.3 * _site_rate(per_product)
    rank = per_product["views"].rank(pct=True)
    assert (rank[wasted] > 0.85).all(), "wasted-exposure products sit in the top 15% by views"


@pytest.mark.parametrize(
    ("role", "favoured", "other"), [("mobile_first", "mobile", "desktop"), ("desktop_first", "desktop", "mobile")]
)
def test_device_first_products(frames, synthetic_truth: dict, role: str, favoured: str, other: str) -> None:
    item = frames["item_daily"]
    g = item[item["product_key"] == synthetic_truth[role]].groupby("device_category")[["views", "purchases"]].sum()
    rate = g["purchases"] / g["views"]
    assert rate[favoured] >= 2.0 * rate[other]
    assert g["purchases"].sum() >= 40


def test_holiday_uplift_apparel_vs_drinkware(frames, synthetic_truth: dict) -> None:
    item = frames["item_daily"]

    def uplift(group: str) -> float:
        d = item[item["product_key"].isin(_group_keys(frames, group))]
        cyber = _rate(d[_in(d["event_date"], "2020-11-23", "2020-11-30")])
        return cyber / _rate(d[_in(d["event_date"], "2020-11-01", "2020-11-22")])

    planted = synthetic_truth["holiday_uplift"]["Apparel"] / synthetic_truth["holiday_uplift"]["Drinkware"]
    assert uplift("Apparel") / uplift("Drinkware") >= 0.75 * planted
    assert uplift("Drinkware") > 1.2  # the site-wide Cyber Week seasonality applies to every group


def _spike_view(frames, pair: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    item = frames["item_daily"]
    spike = pd.to_datetime(pair["spike_days"])
    cat = item[item["product_key"].isin(_group_keys(frames, pair["category_group"]))]
    a_views = cat[cat["product_key"] == pair["a"]].groupby("event_date")["views"].sum()
    share = (a_views / cat.groupby("event_date")["views"].sum()).fillna(0)
    b = item[item["product_key"] == pair["b"]].groupby("event_date")[["views", "purchases"]].sum()
    return b, share, pd.Series(b.index.isin(spike), index=b.index)


@pytest.mark.parametrize("pair_name", ["cannibal_pair", "null_pair"])
def test_spike_pairs_have_share_spikes(frames, synthetic_truth: dict, pair_name: str) -> None:
    pair = synthetic_truth[pair_name]
    assert {pair["a"], pair["b"]} <= set(_group_keys(frames, pair["category_group"]))
    days = pd.to_datetime(pair["spike_days"])
    assert len(days) == 8 and days.is_unique
    assert not _in(pd.Series(days), "2020-11-23", "2020-11-30").any(), "spike days avoid Cyber Week"
    _, share, _ = _spike_view(frames, pair)
    assert (share[share.index.isin(days)] >= 2.0 * share.median()).all()


def test_cannibal_pair_b_dips_on_spike_days(frames, synthetic_truth: dict) -> None:
    b, _, on = _spike_view(frames, synthetic_truth["cannibal_pair"])
    assert _rate(b[on]) <= 0.6 * _rate(b[~on])


def test_null_pair_b_does_not_dip(frames, synthetic_truth: dict) -> None:
    b, _, on = _spike_view(frames, synthetic_truth["null_pair"])
    assert _rate(b[on]) >= 0.7 * _rate(b[~on])


def test_trending_and_control_groups(frames, synthetic_truth: dict, synthetic_dir: Path) -> None:
    trends, item = frames["trends_daily"], frames["item_daily"]
    assert trends["interest"].between(0, 100).all() and (trends["source"] == "synthetic").all()
    assert trends["date"].min() == pd.Timestamp("2020-10-01") and trends["date"].max() == pd.Timestamp("2021-01-31")
    km = pd.read_csv(synthetic_dir / "trends_keywords.csv", dtype=str, keep_default_na=False)
    assert list(km.columns) == ["group_type", "group_value", "keyword", "notes"]
    assert set(km["group_value"]) == set(config.CATEGORY_GROUPS) - {"Other"}
    assert set(km["keyword"]) == set(trends["keyword"])

    def ratio(dates: pd.Series, values: pd.Series) -> float:
        december = values[_in(dates, "2020-12-02", "2020-12-18")].mean()
        return float(december / values[_in(dates, "2020-11-01", "2020-11-22")].mean())

    def interest(group: str) -> float:
        t = trends[trends["group_value"] == group]
        return ratio(t["date"], t["interest"])

    def site_views(group: str) -> float:
        v = item[item["product_key"].isin(_group_keys(frames, group))].groupby("event_date")["views"].sum()
        return ratio(pd.Series(v.index, index=v.index), v)

    trending, control = synthetic_truth["trending_group"], synthetic_truth["control_group"]
    assert interest(trending) >= 1.4
    assert 0.85 <= site_views(trending) <= 1.15
    assert 0.85 <= interest(control) <= 1.15
    assert 0.85 <= site_views(control) <= 1.15


# --------------------------------------------------------------------------- determinism and CLI


def test_same_seed_gives_identical_files(tmp_path: Path, manifest: dict) -> None:
    t0 = time.perf_counter()
    make_synthetic_snapshots(tmp_path / "again", seed=7)
    elapsed = time.perf_counter() - t0
    again = contracts.read_json(tmp_path / "again" / "manifest.json")
    assert {k: v["sha256"] for k, v in again["files"].items()} == {k: v["sha256"] for k, v in manifest["files"].items()}
    assert elapsed < 10.0
    other = make_synthetic_snapshots(tmp_path / "other", seed=8)
    other_manifest = contracts.read_json(tmp_path / "other" / "manifest.json")
    assert other_manifest["files"]["item_daily"]["sha256"] != manifest["files"]["item_daily"]["sha256"]
    assert other["seed"] == 8


def test_too_few_products_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_synthetic_snapshots(tmp_path, n_products=MIN_PRODUCTS - 1)


def _cli():
    path = config.PROJECT_ROOT / "scripts" / "make_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_the_real_data_dir(capsys: pytest.CaptureFixture[str]) -> None:
    before = sorted(p.name for p in config.DEFAULT_DATA_DIR.iterdir()) if config.DEFAULT_DATA_DIR.exists() else []
    assert _cli().main(["--out", str(config.DEFAULT_DATA_DIR)]) == 2
    assert _cli().main(["--out", str(config.DEFAULT_DATA_DIR / "raw")]) == 2
    after = sorted(p.name for p in config.DEFAULT_DATA_DIR.iterdir()) if config.DEFAULT_DATA_DIR.exists() else []
    assert before == after
    assert "real data directory" in capsys.readouterr().out


def test_cli_writes_a_snapshot_set(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    assert _cli().main(["--out", str(out), "--seed", "3", "--n-products", "150"]) == 0
    manifest = contracts.read_json(out / "manifest.json")
    assert manifest["source"] == "synthetic" and manifest["synthetic_truth"]["n_products"] == 150

"""The snapshot builder and verifier on a tiny hand-made raw export."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ga4app import config, contracts

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_snapshots = _script("build_snapshots")
verify_snapshots = _script("verify_snapshots")

DATES = pd.to_datetime(["2020-11-01", "2020-11-02", "2020-11-03"])
DEVICES = ("desktop", "mobile")
# (item_id, item_name, item_category, views per row, purchases per row, price)
CATALOG = [
    ("GGOEGXXX0001", "Google Tee Green", "Home/Sale/", 400, 4, None),
    ("9190001", "Google  Tee  Green", "Apparel", 0, 2, 25.0),  # same product, purchase-side id
    ("GGOEGXXX0002", "Android Iconic 4&quot; Decal", "Home/Stationery/Stickers/", 300, 3, None),
    ("GGOEGXXX0003", 'Android Iconic 4" Decal', "Home/Stationery/Stickers/", 100, 1, 2.0),
    ("GGOEGXXX0004", "Google Laptop Sleeve", "Lifestyle/Bags/", 200, 1, 30.0),
    ("GGOEGXXX0005", "Google Seattle Campus Mug", "Campus Collection", 150, 3, 18.0),
    ("GGOEGXXX0006", "Google Mini Kick Ball", "Lifestyle/", 50, 0, None),
    ("GGOEGXXX0007", "Google Women's Striped L/S", "Home/Sale/", 250, 0, None),
    ("9195910", "Womens Google Striped LS", "Apparel", 0, 2, 16.0),  # legacy purchase name (alias)
    ("9181151", "Gift Card - $25.00", "Gift Cards", 0, 1, 25.0),  # purchased, never viewed
    ("GGOEGXXX0009", "(not set)", "", 5, 0, None),  # placeholder name
]
N_ROWS = len(DATES) * len(DEVICES)  # raw rows per catalog entry


def _daily_row(date, device, item_id, name, views, purchases) -> dict:
    return {
        "event_date": date,
        "device_category": device,
        "item_id": item_id,
        "item_name": name,
        "view_rows": 2 * views,
        "view_events": views,
        "primary_views": views // 3,
        "add_to_carts": views // 10,
        "checkouts": views // 20,
        "purchases": purchases,
        "purchase_events_raw": purchases + (1 if purchases else 0),
        "purchases_missing_txn": 0,
        "units": purchases,
        "promo_views": 0,
        "promo_clicks": 0,
        "revenue_usd": 20.0 * purchases,
    }


def write_raw(data_dir: Path, *, extra_device: tuple[str, int] = ("smart tv", 1)) -> None:
    """Write raw exports + profile.json; ``extra_device`` adds one row of an unknown device."""
    rows = [
        _daily_row(d, dev, i, n, v, p) for d in DATES for dev in DEVICES for i, n, _, v, p, _ in CATALOG
    ]
    device, views = extra_device
    rows.append(_daily_row(DATES[0], device, "GGOEGXXX0004", "Google Laptop Sleeve", views, 0))
    contracts.write_snapshot(pd.DataFrame(rows), "bq_item_daily", data_dir)

    attrs = []
    for item_id, name, category, views, purchases, price in CATALOG:
        attrs.append(
            {
                "item_id": item_id,
                "item_name": name,
                "item_category": category,
                "item_category2": None,
                "item_brand": None,
                "item_rows": 2 * views * N_ROWS + purchases * N_ROWS,
                "view_rows": 2 * views * N_ROWS,
                "purchase_rows": purchases * N_ROWS,
                "median_price_usd": price,
                "first_seen": DATES[0],
                "last_seen": DATES[-1],
            }
        )
    # A placeholder category with more view rows must not win over the real one.
    attrs.append({**attrs[0], "item_category": "(not set)", "view_rows": 10**6, "item_rows": 10**6, "median_price_usd": None})
    contracts.write_snapshot(pd.DataFrame(attrs), "bq_item_attrs", data_dir)

    promo = [
        ("view_promotion", "Reach New Heights", "Front Page Carousel", None, 50),
        ("select_promotion", "Reach New Heights", "Front Page Carousel", None, 5),
        ("view_promotion", None, "Front Page Carousel", None, 7),
        ("view_promotion", None, None, None, 11),
        ("view_promotion", None, "Front Page Carousel", "Fall Campaign 2020", 1),
    ]
    contracts.write_snapshot(
        pd.DataFrame(
            [
                {
                    "event_date": DATES[0],
                    "device_category": "desktop",
                    "event_name": event,
                    "promotion_id": None,
                    "promotion_name": promo_name,
                    "creative_name": creative,
                    "creative_slot": None,
                    "item_id": None,
                    "item_name": item_name,
                    "item_rows": n,
                    "events": n,
                }
                for event, promo_name, creative, item_name, n in promo
            ]
        ),
        "bq_promo_daily",
        data_dir,
    )
    contracts.write_json_atomic(
        {
            "decisions": {
                "product_key": "name_key",
                "view_metric": "view_events",
                "featured_signal": "views_share",
                "notes": ["tiny test export"],
            },
            "purchases": {"purchase_events": 5692},
        },
        data_dir / contracts.JSON_FILES["profile"],
    )


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    write_raw(tmp_path)
    return tmp_path


def _read(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    item = contracts.read_snapshot(data_dir, "item_daily")
    dim = contracts.read_snapshot(data_dir, "product_dim").set_index("product_key")
    manifest = contracts.read_json(data_dir / contracts.JSON_FILES["manifest"])
    return item, dim, manifest


def _raw_total(column: str, *, include=lambda name: name != "(not set)") -> int:
    """Raw-export total of a column over the catalog rows (the dropped smart-tv row excluded)."""
    per_row = {"view_events": lambda v, p: v, "view_rows": lambda v, p: 2 * v, "purchases": lambda v, p: p}
    return sum(per_row[column](v, p) * N_ROWS for _, n, _, v, p, _ in CATALOG if include(n))


# --------------------------------------------------------------------------- name_key (default)


def test_name_key_build(raw_dir: Path):
    assert build_snapshots.main(["--data-dir", str(raw_dir)]) == 0
    item, dim, manifest = _read(raw_dir)

    assert set(dim.index) == {
        "google tee green",
        'android iconic 4" decal',
        "google laptop sleeve",
        "google seattle campus mug",
        "google mini kick ball",
        "google women's striped l/s",
        "gift card - $25.00",
    }
    assert set(item["device_category"].astype(str)) == set(DEVICES)  # the 1-view smart tv is dropped
    assert (item["views"] == item["view_events"]).all()
    assert int(item["views"].sum()) == _raw_total("view_events")
    assert int(item["purchases"].sum()) == _raw_total("purchases")
    assert not item.duplicated(["event_date", "device_category", "product_key"]).any()

    tee = dim.loc["google tee green"]
    assert tee["item_name"] == "Google Tee Green"
    assert tee["item_ids"] == "9190001|GGOEGXXX0001" and tee["n_item_ids"] == 2 and tee["n_names"] == 2
    assert tee["category_raw"] == "Home/Sale/" and tee["category_path"] == "Sale"
    assert tee["category_group"] == "Apparel" and tee["brand"] == "Google"
    assert tee["median_price_usd"] == 25.0
    assert tee["total_views"] == 400 * N_ROWS and tee["total_purchases"] == 6 * N_ROWS

    decal = dim.loc['android iconic 4" decal']
    assert decal["item_name"] == 'Android Iconic 4" Decal' and decal["n_names"] == 2
    assert decal["category_path"] == "Stationery > Stickers" and decal["category_group"] == "Office & Stationery"
    assert decal["brand"] == "Android"

    assert dim.loc["google seattle campus mug", "category_group"] == "Drinkware"
    assert dim.loc["google laptop sleeve", "category_group"] == "Bags"
    assert dim.loc["google mini kick ball", "category_group"] == "Lifestyle & Gifts"

    striped = dim.loc["google women's striped l/s"]  # the legacy purchase name folds in
    assert striped["item_name"] == "Google Women's Striped L/S"
    assert striped["total_purchases"] == 2 * N_ROWS and striped["n_names"] == 2
    assert manifest["name_aliases"] == {"womens google striped ls": "google women's striped l/s"}

    gift = dim.loc["gift card - $25.00"]  # purchase-only products are kept
    assert gift["total_views"] == 0 and gift["total_purchases"] == N_ROWS
    assert gift["category_raw"] == "Gift Cards" and gift["category_group"] == "Lifestyle & Gifts"

    assert manifest["source"] == "bigquery+csv"
    assert manifest["schema_version"] == contracts.SCHEMA_VERSION
    assert manifest["product_key"] == "name_key" and manifest["view_metric"] == "view_events"
    assert manifest["decisions"]["notes"] == ["tiny test export"] and "overrides" not in manifest["decisions"]
    assert manifest["placeholders"] == list(config.PLACEHOLDERS)
    assert set(manifest["periods"]) == set(config.PERIODS)
    assert manifest["periods"]["cyber_week"]["start"] == "2020-11-23"
    assert manifest["devices"]["unknown"][0]["device"] == "smart tv"
    assert manifest["devices"]["unknown"][0]["action"] == "dropped"
    assert manifest["dropped_rows"]["no_product_key"] == N_ROWS
    for name in ("item_daily", "product_dim", "promo_daily"):
        entry = manifest["files"][name]
        assert entry["sha256"] == contracts.file_sha256(raw_dir / entry["file"])
        assert entry["built_from"]
    assert manifest["files"]["item_daily"]["rows"] == len(item)
    assert "site_daily" not in manifest["files"]  # only listed when present


def test_category_colors_freeze_three_hues(raw_dir: Path):
    build_snapshots.main(["--data-dir", str(raw_dir)])
    colors = _read(raw_dir)[2]["category_colors"]
    assert set(colors) == set(config.CATEGORY_GROUPS)
    hued = {g: c for g, c in colors.items() if c != config.PALETTE["other"]}
    assert len(hued) == 3
    # Apparel (650 views/row), Office & Stationery (400), Bags (200); Drinkware (150) is gray.
    assert hued == dict(zip(("Apparel", "Office & Stationery", "Bags"), config.SCATTER_CATEGORICAL))
    assert colors["Other"] == config.PALETTE["other"]


def test_promo_daily(raw_dir: Path):
    build_snapshots.main(["--data-dir", str(raw_dir)])
    promo = contracts.read_snapshot(raw_dir, "promo_daily")
    by_name = promo.groupby("promotion_name")[["promo_views", "promo_clicks"]].sum()
    assert by_name.loc["Reach New Heights"].tolist() == [50, 5]
    assert by_name.loc["Front Page Carousel"].tolist() == [8, 0]  # creative_name fallback
    assert by_name.loc["(unknown)"].tolist() == [11, 0]
    assert set(promo["product_key"].dropna()) == {"fall campaign 2020"}
    assert promo["creative_slot"].isna().all()


def test_rerun_is_idempotent(raw_dir: Path):
    build_snapshots.main(["--data-dir", str(raw_dir)])
    names = ["item_daily.parquet", "product_dim.parquet", "promo_daily.parquet", "manifest.json"]
    first = {n: (raw_dir / n).read_bytes() for n in names}
    build_snapshots.main(["--data-dir", str(raw_dir)])
    assert {n: (raw_dir / n).read_bytes() for n in names} == first


# --------------------------------------------------------------------------- overrides


def test_item_id_strategy(raw_dir: Path):
    assert build_snapshots.main(["--data-dir", str(raw_dir), "--product-key", "item_id"]) == 0
    item, dim, manifest = _read(raw_dir)
    assert set(dim.index) == {i for i, *_ in CATALOG}  # one product per id, placeholder names included
    assert dim.loc["GGOEGXXX0001", "item_name"] == "Google Tee Green"
    assert dim.loc["9190001", "total_views"] == 0 and dim.loc["9190001", "total_purchases"] == 2 * N_ROWS
    assert dim.loc["GGOEGXXX0002", "item_name"] == 'Android Iconic 4" Decal'
    assert (dim["n_item_ids"] == 1).all()
    assert dim["n_names"].to_dict() == {i: (0 if n == "(not set)" else 1) for i, n, *_ in CATALOG}
    assert dim.loc["GGOEGXXX0009", "item_name"] == "GGOEGXXX0009"  # no usable name: the id is shown
    assert dim.loc["GGOEGXXX0009", "category_group"] == "Other"
    assert int(item["views"].sum()) == _raw_total("view_events", include=lambda n: True)
    assert manifest["product_key"] == "item_id" and manifest["name_aliases"] == {}
    assert manifest["decisions"]["overrides"]["product_key"] == {"profile": "name_key", "used": "item_id"}


def test_view_metric_override(raw_dir: Path):
    assert build_snapshots.main(["--data-dir", str(raw_dir), "--view-metric", "view_rows"]) == 0
    item, dim, manifest = _read(raw_dir)
    assert (item["views"] == item["view_rows"]).all()
    assert int(item["views"].sum()) == _raw_total("view_rows")
    assert dim.loc["google tee green", "total_views"] == 800 * N_ROWS
    assert manifest["view_metric"] == "view_rows" and manifest["decisions"]["view_metric"] == "view_rows"
    assert manifest["decisions"]["overrides"]["view_metric"] == {"profile": "view_events", "used": "view_rows"}


def test_no_aliases_keeps_legacy_names_apart(raw_dir: Path):
    build_snapshots.main(["--data-dir", str(raw_dir), "--no-aliases"])
    _, dim, manifest = _read(raw_dir)
    assert dim.loc["womens google striped ls", "total_views"] == 0
    assert dim.loc["womens google striped ls", "total_purchases"] == 2 * N_ROWS
    assert dim.loc["google women's striped l/s", "total_purchases"] == 0
    assert manifest["name_aliases"] == {}
    assert manifest["summary"]["products_purchase_only"] == 2


def test_large_unknown_device_folds_into_other(tmp_path: Path):
    write_raw(tmp_path, extra_device=("smart tv", 5000))
    build_snapshots.main(["--data-dir", str(tmp_path)])
    item, _, manifest = _read(tmp_path)
    assert "other" in set(item["device_category"].astype(str))
    assert manifest["devices"]["unknown"][0]["action"] == "mapped to 'other'"
    checks = {c.name: c for c in verify_snapshots.run_checks(tmp_path)}
    assert checks["devices within config.DEVICES"].status == "FAIL"


def test_missing_raw_export_fails_cleanly(tmp_path: Path, capsys):
    assert build_snapshots.main(["--data-dir", str(tmp_path)]) == 1
    assert "not found" in capsys.readouterr().out


# --------------------------------------------------------------------------- helpers and verifier


def test_weighted_median():
    wm = build_snapshots.weighted_median
    assert wm(np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1])) == 20.0
    assert wm(np.array([10.0, 20.0, 30.0]), np.array([1, 1, 10])) == 30.0
    assert wm(np.array([10.0, 20.0]), np.array([1, 1])) == 10.0  # lower median
    assert wm(np.array([np.nan, 5.0]), np.array([9, 1])) == 5.0
    assert wm(np.array([5.0]), np.array([0])) is None


def test_verify_on_tiny_export(raw_dir: Path, capsys):
    build_snapshots.main(["--data-dir", str(raw_dir)])
    checks = {c.name: c for c in verify_snapshots.run_checks(raw_dir)}
    assert checks["product_dim covers item_daily"].status == "PASS"
    assert checks["devices within config.DEVICES"].status == "PASS"
    assert checks["item_daily view_events vs raw export"].status == "PASS"
    assert checks["manifest hashes match the files"].status == "PASS"
    assert checks["92 dates in item_daily"].status == "FAIL"  # 3 days only
    assert checks["snapshots load and validate"].status == "FAIL"  # no site_daily in the tiny export
    assert verify_snapshots.main(["--data-dir", str(raw_dir)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "PASS" in out


def test_real_snapshots_verify():
    """The committed real snapshots pass every HARD check (skipped before they are built)."""
    data_dir = config.DEFAULT_DATA_DIR
    if not (data_dir / contracts.JSON_FILES["manifest"]).exists():
        pytest.skip("real snapshots not built")
    failed = [f"{c.name}: {c.detail}" for c in verify_snapshots.run_checks(data_dir) if c.status == "FAIL"]
    assert not failed

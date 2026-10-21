"""Snapshot contracts: schema validation, typed round trips and the SQL placeholder lists."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ga4app import config, contracts
from ga4app.contracts import ContractError, validate_frame

_IN_LIST = re.compile(r"\bIN\s*\(\s*('(?:[^'\\]|\\.)*'(?:\s*,\s*'(?:[^'\\]|\\.)*')*)\s*\)", re.IGNORECASE)
_LITERAL = re.compile(r"'((?:[^'\\]|\\.)*)'")


def _item_daily(**overrides) -> pd.DataFrame:
    """A valid two-row item_daily frame; ``overrides`` replace whole columns."""
    df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-11-01", "2020-11-02"]),
            "device_category": ["desktop", "mobile"],
            "product_key": ["google zip hoodie f/c", "google zip hoodie f/c"],
        }
    )
    for col, spec in contracts.SCHEMAS["item_daily"].items():
        if col not in df.columns:
            df[col] = np.array([12.5, 0.0]) if spec == "float" else np.array([3, 1], dtype=np.int64)
    for col, values in overrides.items():
        df[col] = values
    return df


def _product_dim(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "product_key": ["google zip hoodie f/c", "android iconic 4\" decal"],
            "item_name": ["Google Zip Hoodie F/C", 'Android Iconic 4" Decal'],
            "item_ids": ["GGOEGAPX1|9195573", "GGOEAOFX2"],
            "n_item_ids": [2, 1],
            "n_names": [1, 2],
            "category_raw": ["Home/Apparel/Men's / Unisex/", None],
            "category_path": ["Apparel > Men's / Unisex", ""],
            "category_group": ["Apparel", "Office & Stationery"],
            "brand": ["Google", "Android"],
            "median_price_usd": [64.0, np.nan],
            "first_seen": pd.to_datetime(["2020-11-01", "2020-11-03"]),
            "last_seen": pd.to_datetime(["2021-01-31", "2021-01-30"]),
            "total_views": [5000, 300],
            "total_purchases": [40, 0],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


# --------------------------------------------------------------------------- synthetic round trip


def test_synthetic_snapshots_round_trip(synthetic_dir: Path) -> None:
    manifest = contracts.read_json(synthetic_dir / "manifest.json")
    present = [name for name, rel in contracts.PARQUET_FILES.items() if (synthetic_dir / rel).exists()]
    assert set(contracts.REQUIRED_SNAPSHOTS) | set(contracts.OPTIONAL_SNAPSHOTS) <= set(present)
    for name in present:
        df = contracts.read_snapshot(synthetic_dir, name)
        assert list(df.columns) == list(contracts.SCHEMAS[name]), name
        assert len(df) == manifest["files"][name]["rows"], name
        for col, spec in contracts.SCHEMAS[name].items():
            if contracts.parse_spec(spec)[0] == "date":
                assert df[col].dtype == "datetime64[ns]", (name, col)


def test_read_snapshot_missing_file(tmp_path: Path) -> None:
    assert contracts.read_snapshot(tmp_path, "promo_daily", required=False) is None
    with pytest.raises(FileNotFoundError):
        contracts.read_snapshot(tmp_path, "item_daily")


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    df = _item_daily()
    df["event_date"] = df["event_date"].dt.date  # python dates (object) are accepted too
    entry = contracts.write_snapshot(df, "item_daily", tmp_path)
    assert entry["rows"] == 2 and entry["file"] == contracts.PARQUET_FILES["item_daily"]
    assert entry["sha256"] == contracts.file_sha256(tmp_path / entry["file"])
    back = contracts.read_snapshot(tmp_path, "item_daily")
    assert back["event_date"].tolist() == list(pd.to_datetime(["2020-11-01", "2020-11-02"]))
    assert back["views"].dtype == "int64" and back["revenue_usd"].tolist() == [12.5, 0.0]
    dim = _product_dim()
    contracts.write_snapshot(dim, "product_dim", tmp_path)
    back = contracts.read_snapshot(tmp_path, "product_dim")
    assert back["category_raw"].isna().tolist() == [False, True]
    assert back.loc[1, "product_key"] == 'android iconic 4" decal'


# --------------------------------------------------------------------------- validate_frame


def test_valid_hand_made_frames_pass() -> None:
    validate_frame(_item_daily(), "item_daily")
    validate_frame(_product_dim(), "product_dim")  # nullable columns may hold nulls


def test_missing_column_raises() -> None:
    with pytest.raises(ContractError, match="missing column 'purchases'"):
        validate_frame(_item_daily().drop(columns=["purchases"]), "item_daily")


@pytest.mark.parametrize(
    ("column", "values", "kind"),
    [
        ("views", ["3", "1"], "int"),
        ("views", [3.5, 1.0], "int"),
        ("event_date", ["2020-11-01", "2020-11-02"], "date"),
        ("revenue_usd", ["12.5", "0"], "float"),
    ],
)
def test_wrong_dtype_raises(column: str, values: list, kind: str) -> None:
    with pytest.raises(ContractError, match=f"'{column}' should be {kind}"):
        validate_frame(_item_daily(**{column: values}), "item_daily")


def test_nulls_in_non_nullable_column_raise() -> None:
    with pytest.raises(ContractError, match="'product_key' has 1 nulls but is not nullable"):
        validate_frame(_item_daily(product_key=["google zip hoodie f/c", None]), "item_daily")
    with pytest.raises(ContractError, match="'item_name' has 1 nulls"):
        validate_frame(_product_dim(item_name=["Google Zip Hoodie F/C", None]), "product_dim")


def test_item_daily_rejects_duplicate_keys() -> None:
    df = _item_daily(event_date=pd.to_datetime(["2020-11-01", "2020-11-01"]), device_category=["desktop", "desktop"])
    with pytest.raises(ContractError, match="duplicate"):
        validate_frame(df, "item_daily")


def test_product_dim_rejects_duplicate_keys() -> None:
    with pytest.raises(ContractError, match="duplicate product_key"):
        validate_frame(_product_dim(product_key=["google zip hoodie f/c"] * 2), "product_dim")


def test_count_invariants() -> None:
    with pytest.raises(ContractError, match="negative counts"):
        validate_frame(_item_daily(units=np.array([3, -1])), "item_daily")
    with pytest.raises(ContractError, match="view_events > view_rows"):
        validate_frame(_item_daily(view_events=np.array([4, 1])), "item_daily")
    with pytest.raises(ContractError, match="purchases > purchase_events_raw"):
        validate_frame(_item_daily(purchases=np.array([4, 1])), "item_daily")


def test_unknown_schema_raises() -> None:
    with pytest.raises(ContractError, match="no schema"):
        validate_frame(_item_daily(), "not_a_snapshot")


# --------------------------------------------------------------------------- SQL placeholders


def test_sql_placeholder_lists_match_config() -> None:
    files = sorted(config.SQL_DIR.glob("*.sql"))
    assert files, "no SQL files found"
    checked = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        lists = [_LITERAL.findall(m.group(1)) for m in _IN_LIST.finditer(text)]
        lists = [lits for lits in lists if "(not set)" in lits]
        # Every '(not set)' in the file must sit inside one of the parsed IN (...) lists.
        assert len(lists) == text.count("'(not set)'"), path.name
        for lits in lists:
            assert tuple(lits) == config.PLACEHOLDERS, path.name
        checked += len(lists)
    assert checked > 0, "no placeholder lists found in sql/"


# --------------------------------------------------------------------------- real snapshots (when built)


def test_real_snapshots_validate(real_data_dir: Path) -> None:
    present = [name for name, rel in contracts.PARQUET_FILES.items() if (real_data_dir / rel).exists()]
    assert set(contracts.REQUIRED_SNAPSHOTS) <= set(present)
    for name in present:
        contracts.read_snapshot(real_data_dir, name)

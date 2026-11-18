"""Snapshot loading (the item-data cut at ``config.ITEM_DATA_END``) and the URL form of the filters."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ga4app import config
from ga4app.types import Filters

data = pytest.importorskip("ga4app.data")
state = pytest.importorskip("ga4app.state")


def test_analysis_windows_stop_at_the_item_data_end():
    assert config.PERIODS["full"].end == config.ITEM_DATA_END == Filters().end
    assert all(p.end <= config.ITEM_DATA_END for k, p in config.PERIODS.items() if k != "export")
    assert (config.PERIODS["export"].start, config.PERIODS["export"].end) == (config.DATA_START, config.DATA_END)
    assert config.PERIODS["january_4wk"].days == config.PERIODS["baseline"].days == 22
    assert set(config.PRESETS) <= set(config.PERIODS) and "export" not in config.PRESETS


def test_split_item_daily_keeps_every_day_in_the_totals():
    days = pd.date_range("2021-01-24", "2021-01-28")
    df = pd.DataFrame({
        "event_date": days.repeat(2), "device_category": ["desktop", "mobile"] * 5, "product_key": "k",
        "views": 10, "checkouts": 2, "purchases": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    })
    kept, totals = data.split_item_daily(df)
    assert kept["event_date"].max() == pd.Timestamp(config.ITEM_DATA_END) and len(kept) == 4
    assert totals["event_date"].tolist() == list(days)
    assert totals["views"].tolist() == [20] * 5 and totals["purchases"].tolist() == [2, 2, 1, 0, 0]


def test_load_snapshot_drops_item_rows_after_the_cut(synthetic_dir):
    snap = data.load_snapshot(synthetic_dir)
    raw = pd.read_parquet(synthetic_dir / "item_daily.parquet")
    assert snap.item_daily["event_date"].max() == pd.Timestamp(config.ITEM_DATA_END)
    assert len(snap.item_daily) < len(raw)
    assert snap.item_daily_totals["views"].sum() == raw["views"].sum()
    assert snap.item_daily_totals["event_date"].max() == pd.Timestamp(config.DATA_END)


def test_real_snapshot_has_no_item_purchases_after_the_cut(real_data_dir):
    totals = data.load_snapshot(real_data_dir).item_daily_totals.set_index("event_date")["purchases"]
    cut = pd.Timestamp(config.ITEM_DATA_END)
    assert totals.loc[cut - pd.Timedelta(days=6): cut].min() > 100  # a normal late-January week
    assert totals.loc[cut + pd.Timedelta(days=2):].max() < 20  # 12, 10, 16, 11, 0 in the sample


def test_query_params_leave_defaults_out():
    assert state.query_params_for(Filters(), state.DEFAULT_PRESET) == {}
    f = Filters(start=date(2020, 11, 23), end=date(2020, 11, 30), devices=("mobile",),
                categories=("Bags", "Office & Stationery"), min_views=500)
    assert state.query_params_for(f, "cyber_week") == {
        "preset": "cyber_week", "dev": ["mobile"], "cat": ["Bags", "Office & Stationery"], "minv": "500",
    }
    custom = state.query_params_for(f, None)
    assert (custom["start"], custom["end"]) == ("2020-11-23", "2020-11-30") and "preset" not in custom

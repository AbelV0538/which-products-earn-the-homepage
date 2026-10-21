"""Tests for the Google Trends signal (ga4app.trends, scripts/fetch_trends.py). No network."""

from __future__ import annotations

import importlib.util
import json
import socket
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ga4app import config, contracts, trends
from ga4app.config import Period

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAME = "2020-10-01 2021-01-31"


# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any socket connection fails, and the real adapters are replaced by failing stubs."""

    def blocked(*_args, **_kwargs):
        raise AssertionError("network access in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(trends, "ADAPTERS", {m: _never for m in trends.METHODS})


def _never(keyword, timeframe, geo):
    raise AssertionError(f"unexpected network call for {keyword!r}")


def _raw(start: str = "2020-10-01", end: str = "2021-01-31", *, value=lambda i: (i * 7) % 90 + 5, lt1=(), partial=()):
    """A raw timelineData payload as Google sends it (epoch seconds at UTC midnight)."""
    days = pd.date_range(start, end, freq="D")
    points = []
    for i, d in enumerate(days):
        v = 0 if i in lt1 else value(i)
        p = {"time": str(int(d.timestamp())), "value": [v], "formattedValue": ["<1" if i in lt1 else str(v)], "hasData": [True]}
        if i in partial:
            p["isPartial"] = True
        points.append(p)
    return {"default": {"timelineData": points, "averages": []}}


def _ok(keyword, timeframe, geo):
    return _raw()


class Flaky:
    """Raises the given exceptions in turn, then returns a payload."""

    def __init__(self, *failures):
        self.failures = list(failures)
        self.calls = []

    def __call__(self, keyword, timeframe, geo):
        self.calls.append(keyword)
        if self.failures:
            raise self.failures.pop(0)
        return _raw()


class Always:
    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def __call__(self, keyword, timeframe, geo):
        self.calls.append(keyword)
        raise self.exc


def _keyword_csv(path: Path, rows) -> Path:
    pd.DataFrame(rows, columns=list(trends.KEYWORD_COLUMNS)).to_csv(path, index=False)
    return path


TWO = [
    ("category", "Apparel", "hoodie", "Primary."),
    ("category", "Drinkware", "water bottle", "Primary."),
]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


MULTI_DAY = """Category: All categories

Day,hoodie: (United States)
2020-10-01,41
2020-10-02,<1
2020-10-03,
2020-10-04,100
"""


# --------------------------------------------------------------------------- parsers


def test_parse_multitimeline_day_preamble_and_lt1(tmp_path):
    df = trends.parse_multitimeline_csv(_write(tmp_path / "multiTimeline.csv", MULTI_DAY))
    assert df["keyword"].unique().tolist() == ["hoodie"]
    assert df["geo_label"].unique().tolist() == ["United States"]
    assert df["freq"].unique().tolist() == ["day"]
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-10-01", "2020-10-02", "2020-10-04"]  # blank skipped
    assert df["interest"].tolist() == [41.0, 0.5, 100.0]
    assert df["date"].dtype == "datetime64[ns]" and df["interest"].dtype == float
    assert not df["is_partial"].any()


def test_parse_multitimeline_week_header_several_keywords(tmp_path):
    text = "﻿Category: All categories\n\nWeek,tote bag: (United States),backpack: (United States)\n2020-10-04,12,80\n2020-10-11,<1,75\n"
    df = trends.parse_multitimeline_csv(_write(tmp_path / "w.csv", text))
    assert set(df["freq"]) == {"week"}
    assert sorted(df["keyword"].unique()) == ["backpack", "tote bag"]
    tote = df[df["keyword"] == "tote bag"]
    assert tote["interest"].tolist() == [12.0, 0.5]
    assert tote["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-10-04", "2020-10-11"]


def test_parse_multitimeline_ispartial_column_and_star(tmp_path):
    text = "date,hoodie,isPartial\n2020-10-01,40,False\n2020-10-02,45,False\n2020-10-03,50,True\n"
    df = trends.parse_multitimeline_csv(_write(tmp_path / "pytrends.csv", text))
    assert df["keyword"].unique().tolist() == ["hoodie"]  # isPartial is not a keyword
    assert df["geo_label"].unique().tolist() == [""]
    assert df["is_partial"].tolist() == [False, False, True]
    starred = trends.parse_multitimeline_csv(_write(tmp_path / "star.csv", "Day,hoodie: (United States)\n2020-10-01,40\n2020-10-02,45*\n"))
    assert starred["is_partial"].tolist() == [False, True]
    assert starred["interest"].tolist() == [40.0, 45.0]


def test_parse_multitimeline_without_header_raises(tmp_path):
    with pytest.raises(ValueError, match="header"):
        trends.parse_multitimeline_csv(_write(tmp_path / "bad.csv", "Category: All categories\n\nfoo,bar\n1,2\n"))


def test_parse_timeline_raw_payload():
    raw = _raw("2020-10-01", "2020-10-05", lt1={1}, partial={4})
    df = trends.parse_timeline(raw)
    assert df.columns.tolist() == ["date", "interest", "is_partial"]
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == [f"2020-10-0{i}" for i in range(1, 6)]
    assert df.loc[1, "interest"] == 0.5
    assert df["is_partial"].tolist() == [False, False, False, False, True]


def test_parse_timeline_rounds_tz_shifted_midnights():
    # Midnight in UTC-6 is 06:00 UTC; midnight in UTC+9 is 15:00 UTC the day before.
    t1 = int(pd.Timestamp("2020-11-02 06:00").timestamp())
    t2 = int(pd.Timestamp("2020-11-02 15:00").timestamp())
    df = trends.parse_timeline([{"time": str(t1), "value": [5]}, {"time": str(t2), "value": [6]}])
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-11-02", "2020-11-03"]


def test_parse_timeline_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        trends.parse_timeline({"default": {"timelineData": []}})


# --------------------------------------------------------------------------- keyword map


def test_load_keyword_map_real_config():
    km = trends.load_keyword_map()
    assert km.columns.tolist() == [*trends.KEYWORD_COLUMNS, "role"]
    primary = km[km["role"] == "primary"]
    expected = [g for g in config.CATEGORY_GROUPS if g != "Other"]
    assert sorted(primary.loc[primary["group_type"] == "category", "group_value"]) == sorted(expected)
    assert not primary.loc[primary["group_type"] == "category", "group_value"].duplicated().any()
    assert (km["notes"].str.len() > 20).all()  # every choice is explained
    assert km["keyword"].str.casefold().is_unique


def test_load_keyword_map_rejects_bad_rows(tmp_path):
    bad = _keyword_csv(tmp_path / "bad.csv", [("category", "Toys", "lego", "x"), ("planet", "Mars", "rover", "x")])
    with pytest.raises(ValueError, match="Toys") as err:
        trends.load_keyword_map(bad)
    assert "planet" in str(err.value)
    dup = _keyword_csv(tmp_path / "dup.csv", [*TWO, ("category", "Bags", "Hoodie", "Secondary.")])
    with pytest.raises(ValueError, match="twice"):
        trends.load_keyword_map(dup)


def test_slug_and_manual_url():
    assert trends.slugify("Water Bottle") == "water-bottle"
    assert trends.slugify("t shirt") == "t-shirt"
    url = trends.manual_download_url("water bottle")
    assert url.startswith("https://trends.google.com/trends/explore?")
    assert "date=2020-10-01%202021-01-31" in url and "geo=US" in url and "q=water%20bottle" in url


# --------------------------------------------------------------------------- snapshot frame


def _series(keyword="hoodie", source="trendspy", **kw):
    data = trends.parse_timeline(_raw())
    base = dict(keyword=keyword, group_type="category", group_value="Apparel", geo="US", source=source, fetched_at="2026-10-20T12:00:00+00:00", data=data)
    return trends.KeywordSeries(**{**base, **kw})


def test_build_trends_daily_matches_contract_and_round_trips(tmp_path):
    df = trends.build_trends_daily([_series(), _series("water bottle", "manual_csv", group_value="Drinkware")])
    assert df.columns.tolist() == list(contracts.SCHEMAS["trends_daily"])
    contracts.validate_frame(df, "trends_daily")
    assert len(df) == 2 * 123
    assert set(df["source"]) == {"trendspy", "manual_csv"}
    assert df["fetched_at"].notna().all() and str(df["fetched_at"].dt.tz) == "UTC"
    contracts.write_snapshot(df, "trends_daily", tmp_path)
    back = contracts.read_snapshot(tmp_path, "trends_daily")
    assert len(back) == len(df) and back["is_partial"].dtype == bool


def test_build_trends_daily_rejects_bad_records():
    with pytest.raises(ValueError, match="source"):
        trends.build_trends_daily([_series(source="synthetic")])
    bad = _series()
    bad.data.loc[0, "interest"] = 140.0
    with pytest.raises(ValueError, match="0-100"):
        trends.build_trends_daily([bad])
    empty = trends.build_trends_daily([])
    assert empty.empty and empty.columns.tolist() == list(contracts.SCHEMAS["trends_daily"])


def test_to_weekly_means_and_sunday_labels():
    days = pd.date_range("2020-11-01", "2020-11-15")  # Sun Nov 1 .. Sun Nov 15
    daily = pd.DataFrame(
        {"date": days, "keyword": "hoodie", "group_type": "category", "group_value": "Apparel", "geo": "US", "interest": np.arange(15, dtype=float), "is_partial": [False] * 14 + [True]}
    )
    wk = trends.to_weekly(daily)
    assert wk["week"].dt.strftime("%Y-%m-%d").tolist() == ["2020-11-01", "2020-11-08", "2020-11-15"]  # Sundays
    assert wk["n_days"].tolist() == [1, 7, 6]  # Sun Nov 1 closes its own week; partial Nov 15 dropped
    assert wk["interest"].tolist() == pytest.approx([0.0, np.mean(np.arange(1, 8)), np.mean(np.arange(8, 14))])
    assert not wk["is_partial"].any()
    kept = trends.to_weekly(daily, drop_partial=False)
    assert kept["n_days"].tolist() == [1, 7, 7] and kept["is_partial"].tolist() == [False, False, True]
    assert trends.to_weekly(daily.iloc[:0]).empty


def test_index_to_base():
    base = Period("b", "Base", date(2020, 11, 1), date(2020, 11, 3))
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.date_range("2020-11-01", periods=4))
    idx = trends.index_to_base(s, base)
    assert idx.iloc[:3].mean() == pytest.approx(100.0)
    assert idx.iloc[3] == pytest.approx(200.0)
    zero = trends.index_to_base(pd.Series([0.0, 0.0, 0.0, 5.0], index=s.index), base)
    assert zero.isna().all()
    outside = trends.index_to_base(s, Period("x", "None", date(2021, 1, 1), date(2021, 1, 2)))
    assert outside.isna().all()


# --------------------------------------------------------------------------- fetch orchestration


def test_fetch_keyword_429_then_success_backs_off():
    flaky = Flaky(trends.RateLimited("HTTP 429"))
    pytrends = Always(AssertionError("pytrends must not run"))
    slept = []
    pacer = trends.Pacer(20, 40, sleep=slept.append, seed=1)
    df, method = trends.fetch_keyword("hoodie", timeframe=TIMEFRAME, geo="US", adapters={"trendspy": flaky, "pytrends": pytrends}, pacer=pacer, backoff_base=30)
    assert method == "trendspy" and len(df) == 123
    assert len(flaky.calls) == 2 and not pytrends.calls
    assert len(slept) == 1 and 30 + 20 <= slept[0] <= 30 + 40  # backoff on top of the jitter


def test_fetch_raw_falls_back_and_caps_tries():
    limited = Always(trends.RateLimited("HTTP 429"))
    outcome = {}
    raw, method = trends.fetch_raw("hoodie", adapters={"trendspy": limited, "pytrends": _ok}, pacer=trends.Pacer(0, 0), backoff_base=0, outcome=outcome)
    assert method == "pytrends"
    assert len(limited.calls) == 3  # at most 3 tries per method
    assert outcome["exhausted"] == {"trendspy"} and "429" in outcome["errors"]["trendspy"]
    broken = Always(ValueError("bad payload"))
    _, method = trends.fetch_raw("hoodie", adapters={"trendspy": broken, "pytrends": _ok}, pacer=trends.Pacer(0, 0))
    assert method == "pytrends" and len(broken.calls) == 1  # non-retryable errors move on at once


def test_all_methods_fail_gives_unavailable(tmp_path):
    kw_csv = _keyword_csv(tmp_path / "kw.csv", TWO)
    data_dir = tmp_path / "data"
    ts, pt = Always(trends.RateLimited("HTTP 429")), Always(trends.RateLimited("HTTP 429"))
    meta = trends.run_fetch(data_dir, keywords_csv=kw_csv, adapters={"trendspy": ts, "pytrends": pt}, pacer=trends.Pacer(0, 0), backoff_base=0)
    assert meta["status"] == "unavailable"
    assert meta["method_used"] == {} and set(meta["errors"]) == {"hoodie", "water bottle"}
    assert "manual_csv" in meta["errors"]["hoodie"]
    assert not (data_dir / contracts.PARQUET_FILES["trends_daily"]).exists()  # nothing synthesised
    on_disk = contracts.read_json(data_dir / "trends_meta.json")
    assert on_disk["status"] == "unavailable" and on_disk["keywords"] == ["hoodie", "water bottle"]
    assert set(on_disk["manual_urls"]) == {"hoodie", "water bottle"}
    assert len(ts.calls) == 6 and len(pt.calls) == 6


def test_rate_limited_method_is_skipped_after_two_keywords(tmp_path):
    rows = [*TWO, ("category", "Bags", "backpack", "Primary."), ("category", "Bags", "tote bag", "Secondary.")]
    km = trends.load_keyword_map(_keyword_csv(tmp_path / "kw.csv", rows))
    limited = Always(trends.RateLimited("HTTP 429"))
    res = trends.fetch_all(km, cache_dir=tmp_path / "raw" / "trends", adapters={"trendspy": limited, "pytrends": _ok}, pacer=trends.Pacer(0, 0), backoff_base=0)
    assert res.meta["status"] == "ok"
    assert set(res.meta["method_used"].values()) == {"pytrends"}
    assert len(limited.calls) == 6  # 3 tries on each of the first two keywords, then skipped
    assert res.meta["blocked_methods"] == ["trendspy"]
    assert "skipped" in res.meta["method_errors"]["tote bag"]["trendspy"]


def test_cache_resume_skips_the_network(tmp_path):
    kw_csv = _keyword_csv(tmp_path / "kw.csv", TWO)
    data_dir = tmp_path / "data"
    first = trends.run_fetch(data_dir, keywords_csv=kw_csv, adapters={"trendspy": _ok}, pacer=trends.Pacer(0, 0))
    assert first["status"] == "ok" and first["from_cache"] == []
    cache = data_dir / config.TRENDS_RAW_SUBDIR / "water-bottle.json"
    obj = json.loads(cache.read_text(encoding="utf-8"))
    assert obj["keyword"] == "water bottle" and obj["method"] == "trendspy" and obj["timeframe"] == TIMEFRAME
    again = trends.run_fetch(data_dir, keywords_csv=kw_csv, adapters={"trendspy": _never, "pytrends": _never}, pacer=trends.Pacer(0, 0))
    assert again["status"] == "ok" and again["from_cache"] == ["hoodie", "water bottle"]
    assert again["method_used"] == {"hoodie": "trendspy", "water bottle": "trendspy"}
    df = contracts.read_snapshot(data_dir, "trends_daily")
    fetched = df.loc[df["keyword"] == "water bottle", "fetched_at"].unique()
    assert list(fetched) == [pd.Timestamp(obj["fetched_at"])]  # provenance is the original fetch time
    # A cache for another timeframe is not reused.
    assert trends.read_cache(cache.parent, "water bottle", timeframe="2020-01-01 2020-04-30", geo="US") is None


def test_manual_csv_fallback(tmp_path):
    kw_csv = _keyword_csv(tmp_path / "kw.csv", TWO)
    data_dir = tmp_path / "data"
    manual = data_dir / config.TRENDS_MANUAL_SUBDIR
    days = pd.date_range("2020-09-25", "2021-01-31")
    body = "\n".join(f"{d:%Y-%m-%d},{'<1' if i == 10 else (i % 60) + 1}" for i, d in enumerate(days))
    _write(manual / "multiTimeline.csv", f"Category: All categories\n\nDay,Hoodie: (United States)\n{body}\n")
    _write(manual / "multiTimeline (1).csv", "Category: All categories\n\nDay,water bottle: (Worldwide)\n2020-10-01,5\n")
    meta = trends.run_fetch(data_dir, keywords_csv=kw_csv, methods=("manual_csv",))
    assert meta["status"] == "partial"
    assert meta["method_used"] == {"hoodie": "manual_csv"}
    assert "Worldwide" in meta["errors"]["water bottle"]
    df = contracts.read_snapshot(data_dir, "trends_daily")
    assert df["date"].min() == pd.Timestamp("2020-10-01") and len(df) == 123  # clipped to the timeframe
    assert (df["interest"] == 0.5).sum() == 1  # the "<1" day (Oct 5)
    assert df["keyword"].unique().tolist() == ["hoodie"]  # the map's spelling, not the CSV's "Hoodie"
    assert df["source"].unique().tolist() == ["manual_csv"]


def test_manual_weekly_csv_is_rejected(tmp_path):
    km = trends.load_keyword_map(_keyword_csv(tmp_path / "kw.csv", TWO[:1]))
    manual = _write(tmp_path / "manual" / "w.csv", "Category: All categories\n\nWeek,hoodie: (United States)\n2020-10-04,40\n2020-10-11,45\n").parent
    res = trends.fetch_all(km, manual_dir=manual, methods=("manual_csv",))
    assert res.meta["status"] == "unavailable" and "weekly" in res.meta["errors"]["hoodie"]


def test_partial_status_updates_existing_manifest(tmp_path):
    kw_csv = _keyword_csv(tmp_path / "kw.csv", TWO)
    data_dir = tmp_path / "data"
    manifest = {"schema_version": 1, "source": "bigquery+csv", "files": {"item_daily": {"rows": 5, "sha256": "x", "built_from": []}}}
    contracts.write_json_atomic(manifest, data_dir / "manifest.json")

    def only_hoodie(keyword, timeframe, geo):
        if keyword != "hoodie":
            raise RuntimeError("HTTP 400")
        return _raw()

    meta = trends.run_fetch(data_dir, keywords_csv=kw_csv, adapters={"trendspy": only_hoodie, "pytrends": only_hoodie}, pacer=trends.Pacer(0, 0))
    assert meta["status"] == "partial" and meta["manifest_updated"]
    man = contracts.read_json(data_dir / "manifest.json")
    entry = man["files"]["trends_daily"]
    assert entry["rows"] == 123 and entry["sha256"] == contracts.file_sha256(data_dir / "trends_daily.parquet")
    assert "raw/trends/hoodie.json" in entry["built_from"]
    assert man["files"]["item_daily"]["sha256"] == "x" and man["source"] == "bigquery+csv"


def test_no_manifest_is_never_created(tmp_path):
    kw_csv = _keyword_csv(tmp_path / "kw.csv", TWO[:1])
    meta = trends.run_fetch(tmp_path / "data", keywords_csv=kw_csv, adapters={"trendspy": _ok}, pacer=trends.Pacer(0, 0))
    assert meta["status"] == "ok" and not meta["manifest_updated"]
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_zero_heavy_series_is_flagged(tmp_path):
    km = trends.load_keyword_map(_keyword_csv(tmp_path / "kw.csv", TWO[:1]))
    sparse = lambda k, t, g: _raw(value=lambda i: 0 if i % 2 else 3)  # noqa: E731
    res = trends.fetch_all(km, adapters={"trendspy": sparse}, methods=("trendspy",), pacer=trends.Pacer(0, 0))
    assert "broader keyword" in res.meta["warnings"]["hoodie"]


# --------------------------------------------------------------------------- real adapters (mocked libraries)


def test_trendspy_adapter_maps_429(monkeypatch):
    import requests
    import trendspy

    class FakeTrends:
        def __init__(self, **_kw):
            pass

        def interest_over_time(self, *_a, **_kw):
            resp = requests.Response()
            resp.status_code = 429
            raise requests.HTTPError("429", response=resp)

    monkeypatch.setattr(trendspy, "Trends", FakeTrends)
    with pytest.raises(trends.RateLimited):
        trends._trendspy_adapter("hoodie", TIMEFRAME, "US")


def test_trendspy_adapter_returns_raw(monkeypatch):
    import trendspy

    seen = {}

    class FakeTrends:
        def __init__(self, **kw):
            seen["init"] = kw

        def interest_over_time(self, keywords, **kw):
            seen["call"] = (keywords, kw)
            return {"type": "fe_line_chart"}, _raw()

    monkeypatch.setattr(trendspy, "Trends", FakeTrends)
    raw = trends._trendspy_adapter("hoodie", TIMEFRAME, "US")
    assert len(trends.parse_timeline(raw)) == 123
    assert seen["call"] == (["hoodie"], {"timeframe": TIMEFRAME, "geo": "US", "return_raw": True})


# --------------------------------------------------------------------------- script


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_trends_script", PROJECT_ROOT / "scripts" / "fetch_trends.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_main_exits_zero_when_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trends, "ADAPTERS", {m: Always(trends.RateLimited("HTTP 429")) for m in trends.METHODS})
    kw_csv = _keyword_csv(tmp_path / "kw.csv", TWO)
    data_dir = tmp_path / "data"
    args = ["--data-dir", str(data_dir), "--keywords", str(kw_csv), "--sleep-min", "0", "--sleep-max", "0", "--backoff-base", "0"]
    assert _load_script().main(args) == 0
    out = capsys.readouterr().out
    assert "status unavailable" in out and "trends.google.com/trends/explore" in out
    assert contracts.read_json(data_dir / "trends_meta.json")["status"] == "unavailable"


def test_script_rejects_unknown_method(tmp_path):
    assert _load_script().main(["--data-dir", str(tmp_path), "--methods", "serpapi"]) == 2

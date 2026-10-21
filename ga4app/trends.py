"""Google Trends signal: keyword map, fetch adapters, CSV parser, snapshot builder, normalisation.

Every Trends request is normalised 0-100 to its own peak, so the pipeline sends ONE keyword
per request and the app only compares a keyword with itself (momentum vs its own baseline).

Per keyword, ``fetch_all`` uses the raw response cache in ``data/raw/trends/<slug>.json`` when
present (resume), else trendspy, then pytrends, then a CSV downloaded by hand from
trends.google.com into ``data/trends_manual/``. Keywords that no method returns are recorded
in ``trends_meta.json``; they are never synthesised or interpolated.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import io
import json
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import config, contracts
from .config import Period

DEFAULT_TIMEFRAME = "2020-10-01 2021-01-31"  # ~4 months: Google returns daily values
DEFAULT_GEO = "US"
METHODS = ("trendspy", "pytrends")  # network adapters, tried in this order
MANUAL_METHOD = "manual_csv"
ALL_METHODS = (*METHODS, MANUAL_METHOD)
GROUP_TYPES = ("category", "brand")
KEYWORD_COLUMNS = ("group_type", "group_value", "keyword", "notes")
LESS_THAN_ONE = 0.5  # Google prints "<1" for a non-zero value below 1
ZERO_SHARE_WARN = 0.30  # DESIGN R8: more zero days than this -> use a broader keyword
CACHE_VERSION = 1
HTTP_TIMEOUT = (10.0, 30.0)  # (connect, read) seconds
TRENDS_COLUMNS: tuple[str, ...] = tuple(contracts.SCHEMAS["trends_daily"])

# Trends CSV header labels for the geos we request ("hoodie: (United States)").
_GEO_LABELS = {"US": "United States", "": "Worldwide"}
_HEADER_FREQ = {"day": "day", "date": "day", "time": "day", "week": "week", "month": "month"}
_SERIES_HEADER = re.compile(r"^(?P<kw>.*?)(?::\s*\((?P<geo>[^()]*)\))?\s*$")
_TIMEFRAME = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})\s*$")


class RateLimited(RuntimeError):
    """HTTP 429 or Google's quota page: retry after an exponential backoff."""


class TransientError(RuntimeError):
    """A connection problem worth retrying."""


class TrendsFetchError(RuntimeError):
    """Every network method failed for one keyword."""

    def __init__(self, keyword: str, errors: Mapping[str, str], exhausted: Iterable[str] = ()):
        self.keyword = keyword
        self.errors = dict(errors)
        self.exhausted = frozenset(exhausted)  # methods still rate-limited / unreachable after every try
        detail = "; ".join(f"{m}: {e}" for m, e in self.errors.items()) or "no method tried"
        super().__init__(f"{keyword!r}: {detail}")


# An adapter takes (keyword, timeframe, geo) and returns the raw interest-over-time payload:
# a dict holding ``timelineData`` (directly or under ``default``), as Google sends it.
Adapter = Callable[[str, str, str], Mapping[str, Any]]


# --------------------------------------------------------------------------- keyword map


def load_keyword_map(path: Path | str = config.TRENDS_KEYWORDS_CSV) -> pd.DataFrame:
    """Read and check ``config/trends_keywords.csv``.

    Returns ``group_type, group_value, keyword, notes, role`` with strings trimmed and
    ``group_type`` lower-cased. ``role`` is "secondary" when the notes start with
    "Secondary", else "primary". Raises ``ValueError`` on unknown groups or a keyword
    listed twice (in any letter case).
    """
    km = pd.read_csv(Path(path), dtype=str, keep_default_na=False)
    missing = [c for c in KEYWORD_COLUMNS if c not in km.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    km = km.loc[:, list(KEYWORD_COLUMNS)].astype(str)
    km = km.assign(**{c: km[c].str.strip() for c in KEYWORD_COLUMNS})
    km["group_type"] = km["group_type"].str.lower()
    km = km[km["keyword"] != ""].reset_index(drop=True)
    problems: list[str] = []
    allowed = {"category": config.CATEGORY_GROUPS, "brand": config.BRANDS}
    for r in km.itertuples(index=False):
        if r.group_type not in allowed:
            problems.append(f"{r.keyword!r}: group_type {r.group_type!r} is not one of {GROUP_TYPES}")
        elif r.group_value not in allowed[r.group_type]:
            problems.append(f"{r.keyword!r}: {r.group_type} {r.group_value!r} is not one of {allowed[r.group_type]}")
    dupes = km.loc[km["keyword"].str.casefold().duplicated(), "keyword"].tolist()
    if dupes:
        problems.append(f"keywords listed twice: {dupes}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    km["role"] = np.where(km["notes"].str.casefold().str.startswith("secondary"), "secondary", "primary")
    return km


def slugify(keyword: str) -> str:
    """File-system-safe cache name: ``"Water Bottle"`` -> ``"water-bottle"``."""
    return re.sub(r"[^a-z0-9]+", "-", keyword.casefold()).strip("-") or "keyword"


def manual_download_url(keyword: str, timeframe: str = DEFAULT_TIMEFRAME, geo: str = DEFAULT_GEO) -> str:
    """The trends.google.com Explore page whose CSV export ``parse_multitimeline_csv`` reads."""
    return f"https://trends.google.com/trends/explore?date={quote(timeframe)}&geo={quote(geo)}&q={quote(keyword)}&hl=en-US"


# --------------------------------------------------------------------------- parsing


def _timeline_points(raw: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        inner = raw.get("default", raw)
        points = inner.get("timelineData") if isinstance(inner, Mapping) else None
    else:
        points = raw
    if not isinstance(points, Sequence) or isinstance(points, str):
        raise ValueError("response has no timelineData")
    if not points:
        raise ValueError("empty timelineData (Google returned no points)")
    return list(points)


def parse_timeline(raw: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Raw single-keyword ``timelineData`` -> ``date, interest, is_partial``.

    ``time`` is epoch seconds; rounding to the nearest midnight keeps the calendar day
    whatever timezone offset Google applied. A formatted value of "<1" becomes 0.5.
    """
    dates, values, partial = [], [], []
    for p in _timeline_points(raw):
        ts = pd.Timestamp(int(p["time"]), unit="s")
        dates.append((ts + pd.Timedelta(hours=12)).floor("D"))
        value = p.get("value") or [np.nan]
        formatted = p.get("formattedValue") or [""]
        v = LESS_THAN_ONE if str(formatted[0]).strip() == "<1" else float(value[0])
        values.append(v)
        partial.append(bool(p.get("isPartial", False)))
    out = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates).astype("datetime64[ns]"),
            "interest": np.asarray(values, dtype=float),
            "is_partial": np.asarray(partial, dtype=bool),
        }
    )
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _parse_value(cell: str) -> tuple[float, bool]:
    """A CSV cell -> (interest, partial marker). "<1" -> 0.5; a trailing "*" marks partial."""
    s = cell.strip()
    star = s.endswith("*")
    s = s.rstrip("*").strip()
    if s == "":
        return np.nan, star
    if s == "<1":
        return LESS_THAN_ONE, star
    return float(s.replace(",", "")), star


def parse_multitimeline_csv(path: Path | str) -> pd.DataFrame:
    """Parse a CSV exported from trends.google.com (``multiTimeline.csv``).

    Handles the preamble ("Category: All categories" and a blank line), a "Day," / "Week," /
    "Month," header with columns like ``hoodie: (United States)``, "<1" values (0.5), and an
    optional ``isPartial`` column (pytrends-style exports). Returns one row per keyword and
    date: ``keyword, geo_label, freq, date, interest, is_partial``.
    """
    path = Path(path)
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    header_at = next(
        (i for i, r in enumerate(rows) if len(r) >= 2 and r[0].strip().casefold() in _HEADER_FREQ),
        None,
    )
    if header_at is None:
        raise ValueError(f"{path.name}: no 'Day,' / 'Week,' header row found")
    header = rows[header_at]
    freq = _HEADER_FREQ[header[0].strip().casefold()]
    partial_col = next((j for j, c in enumerate(header) if c.strip().casefold() in ("ispartial", "is_partial")), None)
    series = []
    for j, col in enumerate(header[1:], start=1):
        if j == partial_col:
            continue
        m = _SERIES_HEADER.match(col.strip())
        kw = (m.group("kw") if m else col).strip()
        if kw:
            series.append((j, kw, ((m.group("geo") or "") if m else "").strip()))
    if not series:
        raise ValueError(f"{path.name}: header has no keyword columns")

    records = []
    for r in rows[header_at + 1 :]:
        if not r or not r[0].strip():
            continue
        d = pd.to_datetime(r[0].strip(), errors="coerce")
        if pd.isna(d):
            continue  # a footer or a second table: not a data row
        row_partial = partial_col is not None and partial_col < len(r) and r[partial_col].strip().casefold() == "true"
        for j, kw, geo_label in series:
            if j >= len(r):
                continue
            value, star = _parse_value(r[j])
            if np.isnan(value):
                continue
            records.append((kw, geo_label, freq, d.normalize(), value, bool(row_partial or star)))
    out = pd.DataFrame(records, columns=["keyword", "geo_label", "freq", "date", "interest", "is_partial"])
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out["interest"] = out["interest"].astype(float)
    out["is_partial"] = out["is_partial"].astype(bool)
    return out.sort_values(["keyword", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- snapshot frame


@dataclass(frozen=True, eq=False)
class KeywordSeries:
    """One keyword's daily series plus its provenance."""

    keyword: str
    group_type: str
    group_value: str
    geo: str
    source: str  # "trendspy" | "pytrends" | "manual_csv"
    fetched_at: datetime | str
    data: pd.DataFrame  # date, interest, is_partial
    origin: str = ""  # cache or CSV file it came from (data-dir relative)


def _empty_trends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            **{c: pd.Series([], dtype="str") for c in ("keyword", "group_type", "group_value", "geo")},
            "interest": pd.Series([], dtype=float),
            "is_partial": pd.Series([], dtype=bool),
            "source": pd.Series([], dtype="str"),
            "fetched_at": pd.Series([], dtype="datetime64[us, UTC]"),
        }
    )


def _utc(value: datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def build_trends_daily(records: Iterable[KeywordSeries | Mapping[str, Any]]) -> pd.DataFrame:
    """Stack keyword series into a frame matching ``contracts.SCHEMAS["trends_daily"]``.

    Each record has ``keyword, group_type, group_value, geo, source, fetched_at`` and ``data``
    (``date, interest`` and optionally ``is_partial``). Raises ``ValueError`` on an unknown
    source or group type, or interest outside 0-100.
    """
    parts = []
    for rec in records:
        r = rec if isinstance(rec, Mapping) else vars(rec)
        d = r["data"]
        if r["source"] not in ALL_METHODS:
            raise ValueError(f"{r['keyword']!r}: unknown source {r['source']!r}")
        if r["group_type"] not in GROUP_TYPES:
            raise ValueError(f"{r['keyword']!r}: unknown group_type {r['group_type']!r}")
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(d["date"]).dt.normalize().astype("datetime64[ns]"),
                "interest": pd.to_numeric(d["interest"], errors="coerce").astype(float),
                "is_partial": d["is_partial"].fillna(False).astype(bool) if "is_partial" in d else False,
            }
        ).dropna(subset=["date", "interest"])
        if ((part["interest"] < 0) | (part["interest"] > 100)).any():
            raise ValueError(f"{r['keyword']!r}: interest outside 0-100")
        part = part.assign(
            keyword=str(r["keyword"]),
            group_type=str(r["group_type"]),
            group_value=str(r["group_value"]),
            geo=str(r["geo"]),
            source=str(r["source"]),
            fetched_at=_utc(r["fetched_at"]),
        )
        parts.append(part)
    parts = [p for p in parts if len(p)]
    if not parts:
        return _empty_trends()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["keyword", "geo", "date"], keep="last")
    out = out.sort_values(["group_type", "group_value", "keyword", "date"]).reset_index(drop=True)
    out = out.loc[:, list(TRENDS_COLUMNS)]
    contracts.validate_frame(out, "trends_daily")
    return out


def to_weekly(trends_daily: pd.DataFrame, freq: str = "W-SUN", *, drop_partial: bool = True) -> pd.DataFrame:
    """Weekly mean interest per keyword; ``week`` is the period's last day (W-SUN: Sunday).

    Matches ``phase2.trend_index``: a mean of daily values, so a short first or last week is
    not biased low. ``n_days`` counts the days behind each mean.
    """
    keys = [c for c in ("keyword", "group_type", "group_value", "geo", "source") if c in trends_daily.columns]
    cols = [*keys, "week", "interest", "n_days", "is_partial"]
    t = trends_daily
    if drop_partial and "is_partial" in t.columns:
        t = t[~t["is_partial"].fillna(False).astype(bool)]
    if t.empty:
        return pd.DataFrame(columns=cols)
    dates = pd.to_datetime(t["date"])
    week = dates.dt.to_period(freq).dt.end_time.dt.normalize().astype("datetime64[ns]")
    partial = t["is_partial"].fillna(False).astype(bool) if "is_partial" in t.columns else pd.Series(False, index=t.index)
    out = (
        t.assign(week=week, is_partial=partial)
        .groupby([*keys, "week"], as_index=False, sort=True)
        .agg(interest=("interest", "mean"), n_days=("interest", "size"), is_partial=("is_partial", "any"))
    )
    out["n_days"] = out["n_days"].astype("int64")
    return out.loc[:, cols]


def index_to_base(series: pd.Series, base: Period) -> pd.Series:
    """Rescale a date-indexed series so its mean over ``base`` (inclusive) is 100.

    All NaN when the base window has no values or a mean of 0.
    """
    idx = pd.to_datetime(series.index)
    in_base = (idx >= pd.Timestamp(base.start)) & (idx <= pd.Timestamp(base.end))
    base_mean = pd.to_numeric(series[in_base], errors="coerce").mean()
    if not np.isfinite(base_mean) or base_mean <= 0:
        return pd.Series(np.nan, index=series.index, name=series.name, dtype=float)
    return 100.0 * pd.to_numeric(series, errors="coerce").astype(float) / base_mean


# --------------------------------------------------------------------------- network adapters


def _raise_http(exc: Exception, response: Any) -> None:
    status = getattr(response, "status_code", None)
    if status == 429:
        raise RateLimited("HTTP 429 Too Many Requests") from exc
    if isinstance(status, int) and status >= 500:
        raise TransientError(f"HTTP {status}") from exc
    raise RuntimeError(f"HTTP {status}: {exc}") from exc


def _trendspy_adapter(keyword: str, timeframe: str, geo: str) -> Mapping[str, Any]:
    """trendspy 0.1.6: ``Trends().interest_over_time([kw], timeframe, geo, return_raw=True)``."""
    import requests
    from trendspy import Trends
    from trendspy.client import TrendsQuotaExceededError

    # One attempt per call: retries and backoff are ours. trendspy prints rate-limit hints,
    # and sends requests without a timeout, so one is bound onto its session.
    client = Trends(request_delay=0, max_retries=1)
    session = getattr(client, "session", None)
    if session is not None:
        session.request = functools.partial(session.request, timeout=HTTP_TIMEOUT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _token, data = client.interest_over_time([keyword], timeframe=timeframe, geo=geo, return_raw=True)
    except TrendsQuotaExceededError as exc:
        raise RateLimited("trendspy quota exceeded") from exc
    except requests.HTTPError as exc:
        _raise_http(exc, exc.response)
    except requests.RequestException as exc:
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc
    except AttributeError as exc:
        # trendspy swallows a connection error and then calls None.raise_for_status().
        if "raise_for_status" in str(exc):
            raise TransientError("request failed before any response") from exc
        raise
    if not isinstance(data, Mapping):
        raise ValueError(f"unexpected trendspy payload {type(data).__name__}")
    return data


def _pytrends_adapter(keyword: str, timeframe: str, geo: str) -> Mapping[str, Any]:
    """pytrends 4.9.2 (archived): the interest-over-time widget's raw JSON.

    ``TrendReq.interest_over_time`` drops the "<1" marks, so the widget request is replayed
    with ``_get_data`` to keep ``formattedValue``.
    """
    import requests
    from pytrends import exceptions as pex
    from pytrends.request import TrendReq

    try:
        client = TrendReq(hl="en-US", tz=360, timeout=HTTP_TIMEOUT)
        client.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo)
        widget = client.interest_over_time_widget
        params = {"req": json.dumps(widget["request"]), "token": widget["token"], "tz": client.tz}
        data = client._get_data(url=TrendReq.INTEREST_OVER_TIME_URL, method=TrendReq.GET_METHOD, trim_chars=5, params=params)
    except pex.TooManyRequestsError as exc:
        raise RateLimited("HTTP 429 Too Many Requests") from exc
    except pex.ResponseError as exc:
        _raise_http(exc, exc.response)
    except requests.RequestException as exc:
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"unexpected pytrends payload {type(data).__name__}")
    return data


ADAPTERS: dict[str, Adapter] = {"trendspy": _trendspy_adapter, "pytrends": _pytrends_adapter}


class Pacer:
    """Spaces network requests: a jittered pause before every request but the first.

    ``wait(extra)`` adds a backoff on top of the jitter. ``sleep`` is injectable for tests.
    """

    def __init__(self, lo: float = 20.0, hi: float = 40.0, *, sleep: Callable[[float], None] | None = None, seed: int | None = None):
        self.lo, self.hi = float(lo), float(max(lo, hi))
        self._sleep = sleep
        self._rng = random.Random(seed)
        self.requests = 0
        self.slept = 0.0

    def wait(self, extra: float = 0.0) -> None:
        if self.requests:
            delay = self._rng.uniform(self.lo, self.hi) + max(extra, 0.0)
            if delay > 0:
                (self._sleep or time.sleep)(delay)
                self.slept += delay
        self.requests += 1


def fetch_raw(
    keyword: str,
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    geo: str = DEFAULT_GEO,
    methods: Sequence[str] = METHODS,
    adapters: Mapping[str, Adapter] | None = None,
    max_tries: int = 3,
    backoff_base: float = 30.0,
    pacer: Pacer | None = None,
    skip: Iterable[str] = (),
    outcome: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Mapping[str, Any], str]:
    """Try each network method in order; return (raw payload, method).

    A 429 or connection error is retried after ``backoff_base * 2**(try-1)`` seconds (plus
    the pacer's jitter), at most ``max_tries`` tries per method; any other error moves on to
    the next method. Raises ``TrendsFetchError`` when all fail. When given, ``outcome`` is
    filled with ``errors`` {method: message} and ``exhausted`` {methods that stayed
    rate-limited or unreachable}, on success too.
    """
    table = {**ADAPTERS, **(adapters or {})}
    pacer = pacer or Pacer(0.0, 0.0)
    skip = set(skip)
    errors: dict[str, str] = {}
    exhausted: set[str] = set()
    try:
        for method in methods:
            if method in skip:
                errors[method] = "skipped: rate-limited on earlier keywords in this run"
                continue
            adapter = table.get(method)
            if adapter is None:
                errors[method] = "unknown method"
                continue
            backoff = 0.0
            for attempt in range(1, max_tries + 1):
                pacer.wait(backoff)
                try:
                    raw = adapter(keyword, timeframe, geo)
                    _timeline_points(raw)
                    return raw, method
                except (RateLimited, TransientError) as exc:
                    errors[method] = f"{exc} (try {attempt}/{max_tries})"
                    backoff = backoff_base * 2 ** (attempt - 1)
                    if attempt == max_tries:
                        exhausted.add(method)
                    if log:
                        log(f"  {method}: {errors[method]}")
                except Exception as exc:  # parse errors, HTTP 4xx, a missing package: next method
                    errors[method] = f"{type(exc).__name__}: {exc}"
                    if log:
                        log(f"  {method}: {errors[method]}")
                    break
        raise TrendsFetchError(keyword, errors, exhausted)
    finally:
        if outcome is not None:
            outcome.update(errors=dict(errors), exhausted=set(exhausted))


def fetch_keyword(
    keyword: str,
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    geo: str = DEFAULT_GEO,
    methods: Sequence[str] = METHODS,
    **kwargs: Any,
) -> tuple[pd.DataFrame, str]:
    """Fetch one keyword -> (``date, interest, is_partial`` frame, method used).

    Extra keyword arguments go to ``fetch_raw`` (adapters, max_tries, backoff_base, pacer).
    """
    raw, method = fetch_raw(keyword, timeframe=timeframe, geo=geo, methods=methods, **kwargs)
    return parse_timeline(raw), method


# --------------------------------------------------------------------------- cache + manual CSVs


def cache_path(cache_dir: Path | str, keyword: str) -> Path:
    return Path(cache_dir) / f"{slugify(keyword)}.json"


def read_cache(cache_dir: Path | str, keyword: str, *, timeframe: str, geo: str) -> dict | None:
    """The cached raw response for this keyword, timeframe and geo, or None."""
    path = cache_path(cache_dir, keyword)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    same = (
        str(obj.get("keyword", "")).casefold() == keyword.casefold()
        and obj.get("timeframe") == timeframe
        and obj.get("geo") == geo
        and obj.get("method") in METHODS
    )
    if not same:
        return None
    try:
        _timeline_points(obj.get("response") or {})
    except ValueError:
        return None
    return obj


def write_cache(
    cache_dir: Path | str, keyword: str, *, timeframe: str, geo: str, method: str, response: Mapping[str, Any], fetched_at: str
) -> Path:
    path = cache_path(cache_dir, keyword)
    payload = {
        "cache_version": CACHE_VERSION,
        "keyword": keyword,
        "geo": geo,
        "timeframe": timeframe,
        "method": method,
        "fetched_at": fetched_at,
        "response": response,
    }
    contracts.write_json_atomic(payload, path)
    return path


@dataclass
class ManualSeries:
    path: Path
    geo_label: str
    freq: str
    data: pd.DataFrame  # date, interest, is_partial
    fetched_at: str  # the file's modification time (when it was downloaded)


def scan_manual_dir(manual_dir: Path | str | None) -> tuple[dict[str, ManualSeries], dict[str, str]]:
    """Index hand-downloaded CSVs by keyword (case-folded) -> series; also per-file errors.

    A keyword found in several files takes the most recently modified one.
    """
    found: dict[str, ManualSeries] = {}
    errors: dict[str, str] = {}
    if manual_dir is None or not Path(manual_dir).is_dir():
        return found, errors
    files = sorted(Path(manual_dir).glob("*.csv"), key=lambda p: p.stat().st_mtime)
    for path in files:
        try:
            parsed = parse_multitimeline_csv(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            errors[path.name] = str(exc)
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
        for (kw, geo_label, freq), g in parsed.groupby(["keyword", "geo_label", "freq"], sort=False):
            data = g.loc[:, ["date", "interest", "is_partial"]].reset_index(drop=True)
            found[str(kw).casefold()] = ManualSeries(path, str(geo_label), str(freq), data, mtime)
    return found, errors


def _timeframe_dates(timeframe: str) -> tuple[date, date] | None:
    m = _TIMEFRAME.match(timeframe)
    return (date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))) if m else None


def _clip(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    span = _timeframe_dates(timeframe)
    if span is None:
        return data
    d = pd.to_datetime(data["date"])
    return data[(d >= pd.Timestamp(span[0])) & (d <= pd.Timestamp(span[1]))].reset_index(drop=True)


def _manual_problem(hit: ManualSeries, geo: str) -> str | None:
    if hit.freq != "day":
        return (
            f"{hit.path.name} has {hit.freq}ly values; download a window under 270 days "
            f"(e.g. {DEFAULT_TIMEFRAME}) so Google returns daily values"
        )
    want = _GEO_LABELS.get(geo)
    if hit.geo_label and want is not None and hit.geo_label.casefold() != want.casefold():
        return f"{hit.path.name} is for {hit.geo_label!r}, not {want!r}"
    return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- orchestration


@dataclass
class FetchResult:
    trends_daily: pd.DataFrame
    meta: dict
    series: list[KeywordSeries] = field(default_factory=list)


def fetch_all(
    keyword_map: pd.DataFrame,
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    geo: str = DEFAULT_GEO,
    cache_dir: Path | str | None = None,
    manual_dir: Path | str | None = None,
    methods: Sequence[str] = ALL_METHODS,
    adapters: Mapping[str, Adapter] | None = None,
    pacer: Pacer | None = None,
    max_tries: int = 3,
    backoff_base: float = 30.0,
    block_after: int = 2,
    refresh: bool = False,
    data_root: Path | str | None = None,
    log: Callable[[str], None] | None = None,
) -> FetchResult:
    """Fetch every keyword in the map: cache -> network methods -> manual CSV.

    A network method that stays rate-limited for ``block_after`` keywords in a row is
    skipped for the rest of the run. Status is "ok" when every keyword has data,
    "unavailable" when none has, else "partial".
    """
    say = log or (lambda _msg: None)
    pacer = pacer or Pacer()
    net_methods = [m for m in methods if m != MANUAL_METHOD]
    root = Path(data_root) if data_root is not None else (Path(cache_dir).parent.parent if cache_dir else Path.cwd())
    manual, manual_errors = scan_manual_dir(manual_dir) if MANUAL_METHOD in methods else ({}, {})
    streak = {m: 0 for m in net_methods}
    blocked: set[str] = set()

    series: list[KeywordSeries] = []
    method_used: dict[str, str] = {}
    errors: dict[str, str] = {}
    method_errors: dict[str, dict[str, str]] = {}
    warnings: dict[str, str] = {}
    from_cache: list[str] = []

    for i, row in enumerate(keyword_map.itertuples(index=False), start=1):
        kw = row.keyword
        tag = f"[{i}/{len(keyword_map)}] {kw!r}"
        data, source, fetched_at, origin = None, None, None, ""
        failures: dict[str, str] = {}

        cached = None if (refresh or cache_dir is None) else read_cache(cache_dir, kw, timeframe=timeframe, geo=geo)
        if cached is not None:
            data, source, fetched_at = parse_timeline(cached["response"]), cached["method"], cached["fetched_at"]
            origin = _rel(cache_path(cache_dir, kw), root)
            from_cache.append(kw)
            say(f"{tag}: cache ({source})")
        elif net_methods:
            outcome: dict[str, Any] = {}
            try:
                raw, source = fetch_raw(
                    kw,
                    timeframe=timeframe,
                    geo=geo,
                    methods=net_methods,
                    adapters=adapters,
                    max_tries=max_tries,
                    backoff_base=backoff_base,
                    pacer=pacer,
                    skip=blocked,
                    outcome=outcome,
                    log=log,
                )
            except TrendsFetchError as exc:
                source = None
                say(f"{tag}: network failed ({exc})")
            failures.update(outcome.get("errors", {}))
            # Circuit breaker: a method blocked for `block_after` keywords in a row is dropped.
            for m in net_methods:
                if m in outcome.get("exhausted", ()):
                    streak[m] += 1
                    if streak[m] >= block_after and m not in blocked:
                        blocked.add(m)
                        say(f"  {m} keeps failing with rate limits; skipping it for the remaining keywords")
                elif m == source:
                    streak[m] = 0
            if source is not None:
                fetched_at = contracts.utc_now_iso()
                data = parse_timeline(raw)
                if cache_dir is not None:
                    path = write_cache(cache_dir, kw, timeframe=timeframe, geo=geo, method=source, response=raw, fetched_at=fetched_at)
                    origin = _rel(path, root)
                say(f"{tag}: {source}")

        if data is None and MANUAL_METHOD in methods:
            hit = manual.get(kw.casefold())
            problem = None if hit is None else _manual_problem(hit, geo)
            if hit is None:
                failures[MANUAL_METHOD] = "no CSV with this keyword in its header"
            elif problem:
                failures[MANUAL_METHOD] = problem
            else:
                clipped = _clip(hit.data, timeframe)
                if clipped.empty:
                    failures[MANUAL_METHOD] = f"{hit.path.name} has no dates inside {timeframe}"
                else:
                    data, source, fetched_at, origin = hit.data, MANUAL_METHOD, hit.fetched_at, _rel(hit.path, root)
                    say(f"{tag}: manual CSV {hit.path.name}")

        if failures:
            method_errors[kw] = failures
        if data is None:
            errors[kw] = "; ".join(f"{m}: {e}" for m, e in failures.items()) or "no method enabled"
            say(f"{tag}: NO DATA")
            continue
        data = _clip(data, timeframe)
        zero_share = float((data["interest"] == 0).mean()) if len(data) else 1.0
        if zero_share > ZERO_SHARE_WARN:
            warnings[kw] = f"{zero_share:.0%} of days are 0; consider a broader keyword"
        method_used[kw] = source
        series.append(KeywordSeries(kw, row.group_type, row.group_value, geo, source, fetched_at, data, origin))

    df = build_trends_daily(series)
    keywords = keyword_map["keyword"].tolist()
    n_ok = len(method_used)
    status = "ok" if keywords and n_ok == len(keywords) else ("unavailable" if n_ok == 0 else "partial")
    meta = {
        "keywords": keywords,
        "geo": geo,
        "timeframe": timeframe,
        "method_used": method_used,
        "errors": errors,
        "status": status,
        "fetched_at": contracts.utc_now_iso(),
        "rows": int(len(df)),
        "from_cache": from_cache,
        "method_errors": method_errors,
        "warnings": warnings,
        "manual_file_errors": manual_errors,
        "blocked_methods": sorted(blocked),
        "manual_urls": {kw: manual_download_url(kw, timeframe, geo) for kw in errors},
    }
    return FetchResult(df, meta, series)


def update_manifest(data_dir: Path | str, entry: Mapping[str, Any]) -> bool:
    """Replace ``files["trends_daily"]`` in an existing manifest (never creates one)."""
    path = Path(data_dir) / contracts.JSON_FILES["manifest"]
    manifest = contracts.read_json(path)
    if not isinstance(manifest, dict):
        return False
    files = manifest.get("files")
    if not isinstance(files, dict):
        files = {}
    files["trends_daily"] = dict(entry)
    manifest["files"] = files
    contracts.write_json_atomic(manifest, path)
    return True


def run_fetch(
    data_dir: Path | str,
    *,
    keywords_csv: Path | str = config.TRENDS_KEYWORDS_CSV,
    timeframe: str = DEFAULT_TIMEFRAME,
    geo: str = DEFAULT_GEO,
    methods: Sequence[str] = ALL_METHODS,
    adapters: Mapping[str, Adapter] | None = None,
    pacer: Pacer | None = None,
    max_tries: int = 3,
    backoff_base: float = 30.0,
    refresh: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Fetch all keywords, write ``trends_daily.parquet`` (when any data came back) and
    ``trends_meta.json``, and update the manifest's entry if a manifest exists. Returns the meta.
    """
    data_dir = Path(data_dir)
    keywords_csv = Path(keywords_csv)
    km = load_keyword_map(keywords_csv)
    res = fetch_all(
        km,
        timeframe=timeframe,
        geo=geo,
        cache_dir=data_dir / config.TRENDS_RAW_SUBDIR,
        manual_dir=data_dir / config.TRENDS_MANUAL_SUBDIR,
        methods=methods,
        adapters=adapters,
        pacer=pacer,
        max_tries=max_tries,
        backoff_base=backoff_base,
        refresh=refresh,
        data_root=data_dir,
        log=log,
    )
    meta = res.meta
    meta["keywords_csv"] = _rel(keywords_csv, config.PROJECT_ROOT)
    meta["trends_daily_written"] = False
    meta["manifest_updated"] = False
    if len(res.trends_daily):
        entry = contracts.write_snapshot(res.trends_daily, "trends_daily", data_dir)
        built_from = sorted({meta["keywords_csv"], *(s.origin for s in res.series if s.origin)})
        meta["trends_daily_written"] = True
        meta["manifest_updated"] = update_manifest(data_dir, {**entry, "built_from": built_from})
    contracts.write_json_atomic(meta, data_dir / contracts.JSON_FILES["trends_meta"])
    return meta

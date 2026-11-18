"""Snapshot loading and cached derived tables for the Streamlit app.

``load_snapshot`` is pure (no Streamlit). Everything else is cached per data version: cached
functions take small hashable keys (version, Filters, strings) plus an ``_snap`` argument that
Streamlit does not hash, so large DataFrames are never hashed.

``Snapshot.item_daily`` stops at ``config.ITEM_DATA_END``: after that day the sample's purchase
events carry no items, so keeping those rows would read as days with no sales.
``Snapshot.item_daily_totals`` keeps every day's totals for the Data & Method evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from . import config, contracts, metrics, phase2
from .types import Filters, Thresholds

KEYWORDS_FILE = "trends_keywords.csv"
FULL_PERIOD = config.PERIODS["full"]


@dataclass(frozen=True, eq=False)
class Snapshot:
    """Every snapshot file the app reads. Optional tables are None when missing."""

    item_daily: pd.DataFrame
    product_dim: pd.DataFrame
    site_daily: pd.DataFrame
    site_funnel: pd.DataFrame | None
    promo_daily: pd.DataFrame | None
    trends_daily: pd.DataFrame | None
    overview: dict
    profile: dict
    manifest: dict
    bq_proof: dict
    trends_meta: dict
    keyword_map: pd.DataFrame | None
    version: str
    data_dir: Path = Path(".")
    item_daily_totals: pd.DataFrame | None = None  # per day, before the ITEM_DATA_END cut

    @property
    def is_synthetic(self) -> bool:
        return str(self.manifest.get("source", "")) == "synthetic"

    @property
    def category_colors(self) -> dict[str, str]:
        return dict(self.manifest.get("category_colors") or {})

    @property
    def has_trends(self) -> bool:
        return self.trends_daily is not None and len(self.trends_daily) > 0 and self.keyword_map is not None


def resolve_data_dir() -> Path:
    return config.resolve_data_dir()


def missing_required(data_dir: Path) -> list[str]:
    """Required snapshot files (parquet + manifest) that do not exist in ``data_dir``."""
    names = [contracts.PARQUET_FILES[n] for n in contracts.REQUIRED_SNAPSHOTS] + ["manifest.json"]
    return [n for n in names if not (Path(data_dir) / n).exists()]


def _version(data_dir: Path) -> str:
    """Data version: the manifest's sha256 (plus the directory, so two dirs never collide)."""
    path = Path(data_dir) / "manifest.json"
    digest = contracts.file_sha256(path) if path.exists() else "no-manifest"
    return hashlib.sha256(f"{Path(data_dir).resolve()}|{digest}".encode()).hexdigest()[:16]


def _keyword_map(data_dir: Path) -> pd.DataFrame | None:
    local = Path(data_dir) / KEYWORDS_FILE
    km = phase2.load_keyword_map(local if local.exists() else config.TRENDS_KEYWORDS_CSV)
    return km if len(km) else None


def split_item_daily(item_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows up to ``config.ITEM_DATA_END``, per-day totals of every row).

    The totals (``event_date, views, checkouts, purchases``) cover the whole export so the gap
    after the cut can still be shown.
    """
    dates = pd.to_datetime(item_daily["event_date"]).dt.normalize()
    cols = [c for c in ("views", "checkouts", "purchases") if c in item_daily.columns]
    totals = item_daily[cols].groupby(dates.rename("event_date")).sum().reset_index()
    keep = (dates <= pd.Timestamp(config.ITEM_DATA_END)).to_numpy()
    return item_daily.loc[keep].reset_index(drop=True), totals


def load_snapshot(data_dir: Path) -> Snapshot:
    """Read and validate every snapshot in ``data_dir`` (pure; raises FileNotFoundError when a
    required file is missing). Item rows after ``config.ITEM_DATA_END`` are dropped."""
    data_dir = Path(data_dir)
    json = {name: contracts.read_json(data_dir / f"{name}.json", default={}) or {}
            for name in ("overview", "profile", "manifest", "bq_proof", "trends_meta")}
    item_daily, item_totals = split_item_daily(contracts.read_snapshot(data_dir, "item_daily"))
    return Snapshot(
        item_daily=item_daily,
        item_daily_totals=item_totals,
        product_dim=contracts.read_snapshot(data_dir, "product_dim"),
        site_daily=contracts.read_snapshot(data_dir, "site_daily"),
        site_funnel=contracts.read_snapshot(data_dir, "site_funnel", required=False),
        promo_daily=contracts.read_snapshot(data_dir, "promo_daily", required=False),
        trends_daily=contracts.read_snapshot(data_dir, "trends_daily", required=False),
        keyword_map=_keyword_map(data_dir),
        version=_version(data_dir),
        data_dir=data_dir,
        **json,
    )


@st.cache_resource(show_spinner="Loading snapshots…")
def _cached_snapshot(data_dir: str, version: str) -> Snapshot:
    return load_snapshot(Path(data_dir))


def get_snapshot() -> Snapshot:
    """The current snapshot, cached by data dir and manifest hash."""
    data_dir = resolve_data_dir()
    return _cached_snapshot(str(data_dir), _version(data_dir))


# --------------------------------------------------------------------------- cached derived tables


@st.cache_data(show_spinner=False)
def _shrinkage_m(version: str, _snap: Snapshot) -> float:
    return metrics.season_prior_strength(_snap.item_daily, _snap.product_dim)


def shrinkage_m(snapshot: Snapshot) -> float:
    """Empirical-Bayes prior strength m over the full season, all devices (cached per version)."""
    return _shrinkage_m(snapshot.version, snapshot)


@st.cache_data(show_spinner=False, max_entries=4)
def _shrinkage_m_raw(version: str, _snap: Snapshot) -> float:
    return metrics.season_prior_strength_raw(_snap.item_daily, _snap.product_dim)


def shrinkage_m_raw(snapshot: Snapshot) -> float:
    """The unclamped method-of-moments m (NaN if not estimable, inf if no extra heterogeneity)."""
    return _shrinkage_m_raw(snapshot.version, snapshot)


def season_filters(f: Filters) -> Filters:
    """``f`` with its dates widened to the full season."""
    return replace(f, start=FULL_PERIOD.start, end=FULL_PERIOD.end)


@st.cache_data(show_spinner="Scoring products…", max_entries=64)
def _products(version: str, f: Filters, views_rule: str, y: str, _snap: Snapshot) -> tuple[pd.DataFrame, Thresholds]:
    return metrics.compute_products(
        _snap.item_daily, _snap.product_dim, f, shrinkage_m(_snap), views_rule=views_rule, y=y
    )


def products(snapshot: Snapshot, f: Filters, *, views_rule: str = "median", y: str = "cr_raw") -> tuple[pd.DataFrame, Thresholds]:
    """Scored product table + quadrant thresholds for scope ``f``."""
    return _products(snapshot.version, f, views_rule, y, snapshot)


@st.cache_data(show_spinner=False, max_entries=64)
def _device_lean(version: str, keys: tuple[str, ...], f: Filters, _snap: Snapshot) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=["product_key", "lean"])
    matrix = metrics.device_matrix(_snap.item_daily, _snap.product_dim, list(keys), f, shrinkage_m(_snap))
    return metrics.device_lean(matrix)


def device_lean(snapshot: Snapshot, keys: tuple[str, ...], f: Filters) -> pd.DataFrame:
    """Mobile-vs-desktop lean for ``keys`` (device filter ignored)."""
    return _device_lean(snapshot.version, tuple(sorted(set(keys))), f, snapshot)


@st.cache_data(show_spinner=False, max_entries=32)
def _device_summary(version: str, f: Filters, _snap: Snapshot) -> pd.DataFrame:
    return metrics.device_summary(_snap.item_daily, _snap.product_dim, f)


def device_summary(snapshot: Snapshot, f: Filters) -> pd.DataFrame:
    return _device_summary(snapshot.version, f, snapshot)


@st.cache_data(show_spinner="Scoring campaign windows…", max_entries=32)
def _campaign_scores(version: str, campaign: str, f: Filters, _snap: Snapshot) -> pd.DataFrame:
    return metrics.campaign_scores(_snap.item_daily, _snap.product_dim, config.CAMPAIGNS[campaign], f, shrinkage_m(_snap))


def campaign_scores(snapshot: Snapshot, campaign: str, f: Filters) -> pd.DataFrame:
    """``metrics.campaign_scores`` for a ``config.CAMPAIGNS`` name (dates of ``f`` ignored)."""
    return _campaign_scores(snapshot.version, campaign, f, snapshot)


@st.cache_data(show_spinner="Testing for cannibalization…")
def _season_cannibal(version: str, _snap: Snapshot) -> pd.DataFrame:
    f = Filters(min_views=0)
    spikes = phase2.exposure_spikes(_snap.item_daily, _snap.product_dim, f)
    return phase2.cannibalization_tests(_snap.item_daily, _snap.product_dim, spikes, f)


def season_cannibal(snapshot: Snapshot) -> pd.DataFrame:
    """Full-season, all-device cannibalization tests (cached per version)."""
    return _season_cannibal(snapshot.version, snapshot)


def conflicts(snapshot: Snapshot) -> set[frozenset[str]]:
    return phase2.conflict_pairs(season_cannibal(snapshot))


@st.cache_data(show_spinner=False, max_entries=32)
def _alerts(version: str, f: Filters, base: str, focus: str, _snap: Snapshot) -> pd.DataFrame | None:
    if not _snap.has_trends:
        return None
    scored, th = _products(version, season_filters(f), "median", "cr_raw", _snap)
    return phase2.trending_alerts(
        _snap.trends_daily, _snap.item_daily, _snap.product_dim, _snap.keyword_map,
        config.PERIODS[base], config.PERIODS[focus], f, scored, th,
    )


def trending_alerts(snapshot: Snapshot, f: Filters, base: str = "baseline", focus: str = "december") -> pd.DataFrame | None:
    """Google-Trends alerts for ``focus`` vs ``base``; None when trends data or the keyword map is missing."""
    return _alerts(snapshot.version, f, base, focus, snapshot)


@st.cache_data(show_spinner=False, max_entries=32)
def _holiday_shift(version: str, f: Filters, _snap: Snapshot) -> pd.DataFrame:
    keys = ("baseline", *dict.fromkeys(config.CAMPAIGNS.values()))
    periods = [config.PERIODS[k] for k in keys]
    frames = [metrics.period_comparison(_snap.item_daily, _snap.product_dim, periods, f, by=by)
              for by in (None, "category_group")]
    return pd.concat([fr for fr in frames if len(fr)] or frames[:1], ignore_index=True)


def holiday_shift(snapshot: Snapshot, f: Filters) -> pd.DataFrame:
    """Conversion in each campaign window vs the pre-holiday baseline, site-wide ("All products")
    and per category (``metrics.period_comparison`` rows; the dates of ``f`` are ignored)."""
    return _holiday_shift(snapshot.version, season_filters(f), snapshot)


def campaign_alerts(snapshot: Snapshot, f: Filters) -> pd.DataFrame | None:
    """Trend alerts for each campaign's Trends window vs the pre-holiday baseline, with
    ``base_label`` / ``focus_label`` columns; None without Trends data."""
    base = config.PERIODS["baseline"]
    frames = []
    for focus in dict.fromkeys(config.CAMPAIGN_TREND_FOCUS.values()):
        alerts = trending_alerts(snapshot, f, base.key, focus)
        if alerts is not None and len(alerts):
            frames.append(alerts.assign(base_label=base.label, focus_label=config.PERIODS[focus].label))
    return pd.concat(frames, ignore_index=True) if frames else None


def verdict_context(snapshot: Snapshot, f: Filters) -> dict[str, pd.DataFrame | None]:
    """The holiday-shift and Google-Trends inputs of ``narrative.build_payload``."""
    return {"period_shift": holiday_shift(snapshot, f), "alerts": campaign_alerts(snapshot, f)}


@st.cache_data(show_spinner=False, max_entries=128)
def _product_daily(version: str, key: str, f: Filters, _snap: Snapshot) -> pd.DataFrame:
    return metrics.daily_totals(_snap.item_daily, _snap.product_dim, f, keys=[key], ignore_dates=True)


def product_daily(snapshot: Snapshot, key: str, f: Filters) -> pd.DataFrame:
    """One product's daily views and purchases over the full season (devices of ``f`` apply)."""
    return _product_daily(snapshot.version, key, f, snapshot)


@st.cache_data(show_spinner=False, max_entries=128)
def _product_devices(version: str, key: str, f: Filters, _snap: Snapshot) -> pd.DataFrame:
    return metrics.device_matrix(_snap.item_daily, _snap.product_dim, [key], f, shrinkage_m(_snap))


def product_devices(snapshot: Snapshot, key: str, f: Filters) -> pd.DataFrame:
    """One product's ``metrics.device_matrix`` rows (one per device; the device filter is ignored)."""
    return _product_devices(snapshot.version, key, f, snapshot)


def name_of(snapshot: Snapshot, key: Any) -> str:
    """Display name for a product key (the key itself when unknown)."""
    dim = snapshot.product_dim
    hit = dim.loc[dim["product_key"] == key, "item_name"]
    return str(hit.iloc[0]) if len(hit) else str(key)

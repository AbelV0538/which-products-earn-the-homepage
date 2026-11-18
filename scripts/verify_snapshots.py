"""Verify the app snapshots against the contracts and the numbers the pitch and profiling proved.

Usage:
    python scripts/verify_snapshots.py [--data-dir data]

Every parquet file is loaded through ``contracts.read_snapshot`` (which validates it). HARD checks
exit with status 1 when any fails; WARN checks only flag something worth a look.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ga4app import config, taxonomy  # noqa: E402
from ga4app.contracts import (  # noqa: E402
    JSON_FILES,
    OPTIONAL_SNAPSHOTS,
    PARQUET_FILES,
    REQUIRED_SNAPSHOTS,
    file_sha256,
    read_json,
    read_snapshot,
)

EXPECTED_DAYS = (config.DATA_END - config.DATA_START).days + 1  # 92
# Site purchase / view_item conversion (%) per period, from the CSV (DESIGN section 8). "export" is
# all 92 days (the pitch's full-season 1.47%); the app's "full" season stops at ITEM_DATA_END.
SITE_CONVERSION_PCT = {"baseline": 1.18, "cyber_week": 1.85, "december": 1.70, "export": 1.47}
# Item-level purchase rows per site purchase event: an order lists ~2.7 products, so a day below
# 1.0 has lost its item lists. config.ITEM_DATA_END must be the last day at or above it.
ITEMS_PER_PURCHASE_MIN = 1.0
SITE_CONVERSION_TOL_PP = 0.01
PURCHASE_EVENTS = 5692
VIEW_SUM_TOL = 0.005
OTHER_SHARE_MAX = 0.15
REVENUE_EXPECTED = 339_357.0  # deduplicated purchase revenue (profiling)
REVENUE_TOL = 0.10
TOP_REVENUE_PRODUCT = "google zip hoodie f/c"
ITEMS_PER_VIEW_TOL = 0.005
MANIFEST_KEYS = ("schema_version", "source", "decisions", "category_colors", "files", "periods", "placeholders")


@dataclass(frozen=True)
class Check:
    """One verification result. ``level`` is "HARD" or "WARN"."""

    level: str
    name: str
    ok: bool
    detail: str

    @property
    def status(self) -> str:
        if self.ok:
            return "PASS"
        return "FAIL" if self.level == "HARD" else "WARN"


def _hard(name: str, ok: bool, detail: str) -> Check:
    return Check("HARD", name, bool(ok), detail)


def _warn(name: str, ok: bool, detail: str) -> Check:
    return Check("WARN", name, bool(ok), detail)


def _load(data_dir: Path) -> tuple[dict[str, pd.DataFrame | None], dict[str, str]]:
    frames: dict[str, pd.DataFrame | None] = {}
    errors: dict[str, str] = {}
    for name in (*REQUIRED_SNAPSHOTS, *OPTIONAL_SNAPSHOTS, "bq_item_daily"):
        required = name in REQUIRED_SNAPSHOTS or name == "bq_item_daily"
        try:
            frames[name] = read_snapshot(data_dir, name, required=required)
        except Exception as exc:  # report every broken file instead of stopping at the first
            frames[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"[:160]
    return frames, errors


def _dates_check(name: str, df: pd.DataFrame | None, column: str = "event_date") -> Check:
    if df is None:
        return _hard(f"{EXPECTED_DAYS} dates in {name}", False, "not loaded")
    days = pd.to_datetime(df[column]).dt.date
    n, lo, hi = days.nunique(), days.min(), days.max()
    ok = n == EXPECTED_DAYS and lo == config.DATA_START and hi == config.DATA_END
    return _hard(f"{EXPECTED_DAYS} dates in {name}", ok, f"{n} dates, {lo} .. {hi}")


def _site_conversion(site_daily: pd.DataFrame, key: str) -> float:
    period = config.PERIODS[key]
    days = site_daily["event_date"].dt.date
    events = site_daily[(days >= period.start) & (days <= period.end)].groupby("event_name")["events"].sum()
    views = float(events.get("view_item", 0))
    return 100.0 * float(events.get("purchase", 0)) / views if views else float("nan")


def _item_gap_check(item: pd.DataFrame | None, site: pd.DataFrame | None) -> Check:
    """``config.ITEM_DATA_END`` sits exactly where item-level purchases stop: every day up to it has
    at least ``ITEMS_PER_PURCHASE_MIN`` item purchase rows per site purchase event, every later day less."""
    name = f"item purchases end on {config.ITEM_DATA_END:%b %d}"
    if item is None or site is None:
        return _hard(name, False, "not loaded")
    items = item.groupby(pd.to_datetime(item["event_date"]).dt.normalize())["purchases"].sum()
    buys = site[site["event_name"] == "purchase"]
    events = buys.groupby(pd.to_datetime(buys["event_date"]).dt.normalize())["events"].sum()
    ratio = (items.reindex(events.index).fillna(0) / events.where(events > 0)).dropna()
    cut = pd.Timestamp(config.ITEM_DATA_END)
    before, after = ratio[ratio.index <= cut], ratio[ratio.index > cut]
    ok = len(before) > 0 and bool((before >= ITEMS_PER_PURCHASE_MIN).all()) and bool((after < ITEMS_PER_PURCHASE_MIN).all())
    detail = (f"items per purchase event: min {before.min():.2f} through {cut:%b %d}"
              + (f", max {after.max():.2f} after" if len(after) else ", no later days"))
    return _hard(name, ok, detail)


def run_checks(data_dir: Path) -> list[Check]:
    """Run every check against the snapshots in ``data_dir``."""
    frames, errors = _load(data_dir)
    item, dim, site = frames["item_daily"], frames["product_dim"], frames["site_daily"]
    overview = read_json(data_dir / JSON_FILES["overview"], default={}) or {}
    proof = read_json(data_dir / JSON_FILES["bq_proof"], default={}) or {}
    profile = read_json(data_dir / JSON_FILES["profile"], default={}) or {}
    manifest = read_json(data_dir / JSON_FILES["manifest"], default=None)
    checks: list[Check] = []

    present = [n for n, df in frames.items() if df is not None]
    checks.append(
        _hard(
            "snapshots load and validate",
            not errors,
            "; ".join(f"{n}: {e}" for n, e in errors.items()) or f"{len(present)} files valid: {', '.join(present)}",
        )
    )
    missing_keys = [k for k in MANIFEST_KEYS if k not in manifest] if isinstance(manifest, dict) else list(MANIFEST_KEYS)
    hues = [c for c in (manifest or {}).get("category_colors", {}).values() if c != config.PALETTE["other"]]
    checks.append(
        _hard(
            "manifest.json present, <= 3 scatter hues",
            not missing_keys and len(hues) <= len(config.SCATTER_CATEGORICAL),
            f"missing {missing_keys}" if missing_keys else f"source={manifest['source']}, {len(hues)} category hues",
        )
    )

    # ---- shape --------------------------------------------------------------------------
    checks.append(_dates_check("item_daily", item))
    checks.append(_dates_check("site_daily", site))
    checks.append(_item_gap_check(item, site))
    devices: set[str] = set()
    for name in ("item_daily", "site_daily", "promo_daily", "site_funnel"):
        if frames.get(name) is not None:
            devices |= set(frames[name]["device_category"].astype(str).unique())
    checks.append(
        _hard("devices within config.DEVICES", devices <= set(config.DEVICES), f"found {sorted(devices) or 'none'}")
    )
    if item is not None and dim is not None:
        uncovered = set(item["product_key"].unique()) - set(dim["product_key"])
        checks.append(
            _hard("product_dim covers item_daily", not uncovered, f"{len(dim)} products, {len(uncovered)} uncovered keys")
        )
        bad_groups = set(dim["category_group"]) - set(config.CATEGORY_GROUPS)
        bad_brands = set(dim["brand"]) - set(config.BRANDS)
        checks.append(
            _hard(
                "category groups and brands from config",
                not bad_groups and not bad_brands,
                f"unknown groups {sorted(bad_groups)}, brands {sorted(bad_brands)}" if bad_groups or bad_brands else "ok",
            )
        )
    else:
        checks.append(_hard("product_dim covers item_daily", False, "not loaded"))

    # ---- proof numbers ------------------------------------------------------------------
    for label, source in (("overview.json", overview), ("bq_proof.json", proof)):
        got = {k: source.get(k) for k in config.PROOF_EXPECTED}
        checks.append(_hard(f"{label} matches the proof query", got == config.PROOF_EXPECTED, str(got)))

    if site is not None:
        parts, ok = [], True
        for key, target in SITE_CONVERSION_PCT.items():
            got = _site_conversion(site, key)
            ok &= abs(got - target) <= SITE_CONVERSION_TOL_PP + 1e-9
            parts.append(f"{key} {got:.3f}% (want {target:.2f})")
        checks.append(_hard("site purchase/view_item by period", ok, ", ".join(parts)))
    else:
        checks.append(_hard("site purchase/view_item by period", False, "site_daily not loaded"))

    prof_purchases = (profile.get("purchases") or {}).get("purchase_events")
    over_purchases = (overview.get("events_by_name") or {}).get("purchase")
    site_purchases = int(site.loc[site["event_name"] == "purchase", "events"].sum()) if site is not None else None
    checks.append(
        _hard(
            f"purchase events = {PURCHASE_EVENTS:,}",
            prof_purchases == over_purchases == site_purchases == PURCHASE_EVENTS,
            f"profile {prof_purchases}, overview {over_purchases}, site_daily {site_purchases}",
        )
    )

    raw = frames["bq_item_daily"]
    if item is not None and raw is not None:
        got, want = int(item["view_events"].sum()), int(raw["view_events"].sum())
        rel = abs(got - want) / want if want else float("inf")
        checks.append(
            _hard(
                "item_daily view_events vs raw export",
                rel <= VIEW_SUM_TOL,
                f"{got:,} vs {want:,} ({rel:.3%}, limit {VIEW_SUM_TOL:.1%})",
            )
        )
    else:
        checks.append(_hard("item_daily view_events vs raw export", False, "not loaded"))

    # ---- warnings -----------------------------------------------------------------------
    if item is not None and dim is not None:
        views = item.groupby("product_key")["views"].sum()
        groups = dim.set_index("product_key")["category_group"]
        other = float(views[groups.reindex(views.index) == taxonomy.OTHER].sum()) / float(views.sum() or 1)
        checks.append(_warn("'Other' share of views <= 15%", other <= OTHER_SHARE_MAX, f"{other:.2%}"))

        revenue = float(item["revenue_usd"].sum())
        rel = abs(revenue - REVENUE_EXPECTED) / REVENUE_EXPECTED
        checks.append(
            _warn(
                f"revenue near ${REVENUE_EXPECTED:,.0f}",
                rel <= REVENUE_TOL,
                f"${revenue:,.0f} ({rel:+.1%} off, limit {REVENUE_TOL:.0%})",
            )
        )
        top = item.groupby("product_key")["revenue_usd"].sum().nlargest(5)
        names = dim.set_index("product_key")["item_name"].reindex(top.index)
        top_keys = [taxonomy.name_key(n) or k for k, n in names.items()]
        checks.append(
            _warn(
                "top-5 revenue includes the Zip Hoodie",
                TOP_REVENUE_PRODUCT in top_keys,
                ", ".join(f"{k} ${v:,.0f}" for k, v in zip(top_keys, top.to_numpy())),
            )
        )

        avg_items = next(
            (r.get("avg_items") for r in profile.get("items_per_event") or [] if r.get("event_name") == "view_item"), None
        )
        view_item_events = (overview.get("events_by_name") or {}).get("view_item")
        if avg_items and view_item_events:
            got = float(item["view_rows"].sum()) / float(view_item_events)
            rel = abs(got - avg_items) / avg_items
            checks.append(
                _warn(
                    "view_rows per view_item event = profile",
                    rel <= ITEMS_PER_VIEW_TOL,
                    f"{got:.4f} vs {avg_items:.4f} ({rel:.3%})",
                )
            )

    if isinstance(manifest, dict) and isinstance(manifest.get("files"), dict):
        stale = []
        for name, entry in manifest["files"].items():
            path = data_dir / entry.get("file", PARQUET_FILES.get(name, name))
            if not path.exists() or file_sha256(path) != entry.get("sha256"):
                stale.append(name)
        on_disk = [n for n in (*REQUIRED_SNAPSHOTS, *OPTIONAL_SNAPSHOTS) if (data_dir / PARQUET_FILES[n]).exists()]
        unlisted = [n for n in on_disk if n not in manifest["files"]]
        checks.append(
            _warn(
                "manifest hashes match the files",
                not stale and not unlisted,
                f"stale {stale}, unlisted {unlisted} (re-run build_snapshots.py)" if stale or unlisted else "ok",
            )
        )
    return checks


def format_table(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{'STATUS':<6}  {'LEVEL':<5}  {'CHECK':<{width}}  DETAIL", "-" * (width + 40)]
    lines += [f"{c.status:<6}  {c.level:<5}  {c.name:<{width}}  {c.detail}" for c in checks]
    counts = {s: sum(c.status == s for c in checks) for s in ("PASS", "WARN", "FAIL")}
    lines.append(f"{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="defaults to $GA4APP_DATA_DIR or data/")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.resolve_data_dir()
    checks = run_checks(data_dir)
    print(f"verifying {data_dir}")
    print(format_table(checks))
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())

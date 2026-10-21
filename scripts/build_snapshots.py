"""Build the app snapshots from the raw BigQuery exports and the profiling decisions.

Usage:
    python scripts/build_snapshots.py [--data-dir data] [--product-key name_key|item_id]
                                      [--view-metric view_events|view_rows|primary_views] [--no-aliases]

Reads (in the data dir):
  raw/bq_item_daily.parquet, raw/bq_item_attrs.parquet, raw/bq_promo_daily.parquet (optional),
  profile.json (decisions), overview.json (CSV provenance), and any site/trends snapshots present.

Writes:
  item_daily.parquet   event_date x device x product_key; ``views`` is a copy of the view metric
  product_dim.parquet  one row per product: display name, ids, category, group, brand, price, totals
  promo_daily.parquet  banner impressions / clicks by promotion (no product link in this sample)
  manifest.json        decisions, frozen category colours, file hashes, periods, build summary

The product key defaults to the profiled ``name_key``; legacy purchase-catalogue names are folded
into their viewed product (``taxonomy.NAME_ALIASES``) unless ``--no-aliases`` is given. Re-running
on unchanged inputs rewrites byte-identical files and keeps the manifest's ``generated_at``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ga4app import config, taxonomy  # noqa: E402
from ga4app.contracts import (  # noqa: E402
    JSON_FILES,
    PARQUET_FILES,
    SCHEMA_VERSION,
    SCHEMAS,
    file_sha256,
    read_json,
    read_snapshot,
    utc_now_iso,
    write_json_atomic,
    write_snapshot,
)

PRODUCT_KEYS = ("name_key", "item_id")
VIEW_METRICS = ("view_events", "view_rows", "primary_views")
COUNT_COLS = tuple(c for c, spec in SCHEMAS["bq_item_daily"].items() if spec == "int")
SUM_COLS = (*COUNT_COLS, "revenue_usd")
DEVICE_DROP_SHARE = 0.001  # unknown devices below 0.1% of views are dropped, else folded into "other"
OTHER_DEVICE = "other"
UNKNOWN_PROMOTION = "(unknown)"
DEFAULT_DECISIONS = {"product_key": "name_key", "view_metric": "view_events", "featured_signal": "views_share"}
# Where the non-BigQuery snapshots come from, for the manifest's ``built_from``.
SITE_SNAPSHOTS = ("site_daily", "site_funnel")


# --------------------------------------------------------------------------- keys and devices


def _clean_id(value: object) -> str | None:
    """Trimmed item id; ``None`` for missing ids and GA4 placeholders."""
    return taxonomy.clean_text(value)


def key_function(product_key: str, aliases: bool):
    """Return ``f(item_id, item_name) -> product_key | None`` for a key strategy."""
    if product_key == "item_id":
        return lambda item_id, item_name: _clean_id(item_id)
    name_fn = taxonomy.canonical_key if aliases else taxonomy.name_key
    return lambda item_id, item_name: name_fn(item_name)


def add_product_key(df: pd.DataFrame, product_key: str, aliases: bool) -> pd.Series:
    """Product keys for every row of a raw frame with ``item_id`` / ``item_name`` columns."""
    fn = key_function(product_key, aliases)
    pairs = df[["item_id", "item_name"]].astype(object).where(df[["item_id", "item_name"]].notna(), None)
    uniq = pairs.drop_duplicates()
    lookup = {(i, n): fn(i, n) for i, n in uniq.itertuples(index=False, name=None)}
    return pd.Series(
        [lookup[(i, n)] for i, n in pairs.itertuples(index=False, name=None)], index=df.index, dtype=object
    )


def _norm_device(value: object) -> str:
    text = taxonomy.clean_text(value)
    return text.lower() if text else "(not set)"


def device_plan(devices: pd.Series, views: pd.Series) -> tuple[dict[str, str | None], list[dict]]:
    """Map each device value to a config device, ``"other"`` or ``None`` (dropped).

    Values outside ``config.DEVICES`` are dropped together when they hold under 0.1% of views,
    and folded into ``"other"`` otherwise (DESIGN R7).
    """
    norm = devices.map(_norm_device)
    by_dev = views.groupby(norm.to_numpy()).sum()
    total = float(by_dev.sum()) or 1.0
    unknown = [d for d in by_dev.index if d not in config.DEVICES]
    unknown_share = float(by_dev.reindex(unknown).sum()) / total if unknown else 0.0
    action = "dropped" if unknown_share < DEVICE_DROP_SHARE else f"mapped to {OTHER_DEVICE!r}"
    plan: dict[str, str | None] = {d: d for d in by_dev.index if d in config.DEVICES}
    report = []
    for d in unknown:
        plan[d] = None if action == "dropped" else OTHER_DEVICE
        report.append({"device": d, "views": int(by_dev[d]), "share": float(by_dev[d]) / total, "action": action})
    return plan, report


def apply_devices(devices: pd.Series, plan: dict[str, str | None]) -> pd.Series:
    """Mapped device per row; values the plan has never seen follow the plan's unknown rule."""
    fallback = OTHER_DEVICE if OTHER_DEVICE in plan.values() else None
    norm = devices.map(_norm_device)
    return norm.map(lambda d: plan.get(d, d if d in config.DEVICES else fallback)).astype(object)


# --------------------------------------------------------------------------- item_daily


def build_item_daily(raw: pd.DataFrame, view_metric: str, device_map: dict[str, str | None]) -> pd.DataFrame:
    """Aggregate keyed raw rows to (event_date, device_category, product_key)."""
    df = raw.copy()
    df["device_category"] = apply_devices(df["device_category"], device_map)
    df = df[df["product_key"].notna() & df["device_category"].notna()]
    keys = ["event_date", "device_category", "product_key"]
    out = df.groupby(keys, sort=True)[list(SUM_COLS)].sum().reset_index()
    out["views"] = out[view_metric].astype("int64")
    for c in COUNT_COLS:
        out[c] = out[c].astype("int64")
    out["revenue_usd"] = out["revenue_usd"].astype("float64").round(6)
    cols = list(SCHEMAS["item_daily"])
    return out[cols].sort_values(keys, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- product_dim


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    """Lower weighted median of ``values``; ``None`` when no value has positive weight."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    if not ok.any():
        return None
    order = np.argsort(v[ok], kind="stable")
    v, w = v[ok][order], w[ok][order]
    cum = np.cumsum(w)
    return float(v[int(np.searchsorted(cum, cum[-1] / 2.0))])


def _top_by_weight(df: pd.DataFrame, value: str, *weights: str) -> pd.Series:
    """Per product, the ``value`` with the largest summed ``weights`` (compared in order, then alphabetical)."""
    g = df.groupby(["product_key", value], sort=False)[list(weights)].sum().reset_index()
    g = g.sort_values(["product_key", *weights, value], ascending=[True, *[False] * len(weights), True], kind="stable")
    return g.drop_duplicates("product_key").set_index("product_key")[value]


def _price_for(group: pd.DataFrame) -> float | None:
    priced = group[group["median_price_usd"].notna()]
    if priced.empty:
        return None
    # Prices come from cart/purchase rows (view_item rows carry none), so the view weights of the
    # priced combinations are often all zero; item rows are the fallback weight.
    for weight in ("view_rows", "item_rows"):
        value = weighted_median(priced["median_price_usd"].to_numpy(), priced[weight].to_numpy())
        if value is not None:
            return value
    return None


def build_product_dim(raw: pd.DataFrame, attrs: pd.DataFrame, item_daily: pd.DataFrame, view_metric: str) -> pd.DataFrame:
    """One row per ``item_daily`` product (see DESIGN section 3)."""
    keys = pd.Index(sorted(item_daily["product_key"].unique()), name="product_key")
    raw = raw[raw["product_key"].isin(keys)].copy()
    attrs = attrs[attrs["product_key"].isin(keys)].copy()

    # Display name: the cleaned raw spelling with the most views (then purchases, then rows).
    raw["display"] = raw["item_name"].astype(object).map(taxonomy.clean_text)
    named = raw[raw["display"].notna()].assign(_rows=1)
    display = _top_by_weight(named, "display", view_metric, "purchases", "_rows")

    # Distinct ids and raw spellings across both raw sources.
    ids = pd.concat([raw[["product_key", "item_id"]], attrs[["product_key", "item_id"]]])
    ids["item_id"] = ids["item_id"].astype(object).map(_clean_id)
    id_lists = ids.dropna().groupby("product_key")["item_id"].agg(lambda s: sorted(set(s)))
    names = pd.concat([raw[["product_key", "item_name"]], attrs[["product_key", "item_name"]]])
    names["item_name"] = names["item_name"].astype(object).map(lambda v: v.strip() if taxonomy.clean_text(v) else None)
    n_names = names.dropna().groupby("product_key")["item_name"].nunique()

    # Category: most view rows among real categories, else most item rows.
    attrs["category"] = attrs["item_category"].astype(object).map(taxonomy.clean_text)
    cats = attrs[attrs["category"].notna()]
    viewed = cats[cats["view_rows"] > 0]
    category = _top_by_weight(viewed, "category", "view_rows").combine_first(_top_by_weight(cats, "category", "item_rows"))

    price = {k: _price_for(g) for k, g in attrs.groupby("product_key", sort=False)}
    seen = pd.concat(
        [
            raw[["product_key", "event_date"]].rename(columns={"event_date": "first_seen"}).assign(last_seen=raw["event_date"]),
            attrs[["product_key", "first_seen", "last_seen"]],
        ]
    )
    first_seen = seen.groupby("product_key")["first_seen"].min()
    last_seen = seen.groupby("product_key")["last_seen"].max()
    totals = item_daily.groupby("product_key")[["views", "purchases"]].sum()

    rows = []
    for key in keys:
        name = display.get(key) or key
        raw_cat = category.get(key)
        raw_cat = raw_cat if isinstance(raw_cat, str) else None
        item_ids = id_lists.get(key, [])
        rows.append(
            {
                "product_key": key,
                "item_name": name,
                "item_ids": "|".join(item_ids),
                "n_item_ids": len(item_ids),
                "n_names": int(n_names.get(key, 0)),
                "category_raw": raw_cat,
                "category_path": taxonomy.category_path(raw_cat),
                "category_group": taxonomy.category_group(raw_cat, name),
                "brand": taxonomy.brand_of(name),
                "median_price_usd": price.get(key),
                "first_seen": first_seen.get(key),
                "last_seen": last_seen.get(key),
                "total_views": int(totals.at[key, "views"]),
                "total_purchases": int(totals.at[key, "purchases"]),
            }
        )
    out = pd.DataFrame(rows, columns=list(SCHEMAS["product_dim"]))
    for c in ("n_item_ids", "n_names", "total_views", "total_purchases"):
        out[c] = out[c].astype("int64")
    out["median_price_usd"] = out["median_price_usd"].astype("float64")
    return out


# --------------------------------------------------------------------------- promo_daily


def build_promo_daily(promo: pd.DataFrame, product_key: str, aliases: bool, device_map: dict[str, str | None]) -> pd.DataFrame:
    """Banner impressions and clicks by date, device, promotion, slot and (rarely) product."""
    df = promo.copy()
    df["device_category"] = apply_devices(df["device_category"], device_map)
    df = df[df["device_category"].notna()]
    name = df["promotion_name"].astype(object).map(taxonomy.clean_text)
    creative = df["creative_name"].astype(object).map(taxonomy.clean_text)
    df["promotion_name"] = name.where(name.notna(), creative).where(lambda s: s.notna(), UNKNOWN_PROMOTION)
    df["creative_slot"] = df["creative_slot"].astype(object).map(taxonomy.clean_text)
    df["product_key"] = add_product_key(df, product_key, aliases)
    df["promo_views"] = np.where(df["event_name"] == "view_promotion", df["events"], 0)
    df["promo_clicks"] = np.where(df["event_name"] == "select_promotion", df["events"], 0)
    keys = ["event_date", "device_category", "promotion_name", "creative_slot", "product_key"]
    out = df.groupby(keys, sort=True, dropna=False)[["promo_views", "promo_clicks"]].sum().reset_index()
    out = out[(out["promo_views"] > 0) | (out["promo_clicks"] > 0)]
    for c in ("promo_views", "promo_clicks"):
        out[c] = out[c].astype("int64")
    for c in ("creative_slot", "product_key"):  # an all-null key column comes back as float NaN
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out[list(SCHEMAS["promo_daily"])].reset_index(drop=True)


# --------------------------------------------------------------------------- manifest


def category_colors(product_dim: pd.DataFrame) -> dict[str, str]:
    """Top 3 real groups by season views get the scatter hues in order; the rest are gray."""
    views = product_dim.groupby("category_group")["total_views"].sum()
    order = {g: i for i, g in enumerate(config.CATEGORY_GROUPS)}
    ranked = sorted(
        (g for g in views.index if g != taxonomy.OTHER and views[g] > 0), key=lambda g: (-int(views[g]), order.get(g, 99))
    )
    hues = dict(zip(ranked, config.SCATTER_CATEGORICAL))
    groups = [*config.CATEGORY_GROUPS, *sorted(set(views.index) - set(config.CATEGORY_GROUPS))]
    return {g: hues.get(g, config.PALETTE["other"]) for g in groups}


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _trend_sources(path: Path) -> list[str]:
    """Distinct ``source`` values of the trends file (it is validated by its own pipeline)."""
    import pyarrow.parquet as pq

    try:
        sources = pq.read_table(path, columns=["source"]).column("source").unique().to_pylist()
    except Exception:  # unreadable or old layout: still list the file, with a generic origin
        sources = []
    return sorted(str(s) for s in sources if s) or ["google_trends"]


def file_entries(data_dir: Path, written: dict[str, dict], overview: dict | None) -> dict[str, dict]:
    """Manifest ``files`` block: the snapshots just written plus the site/trends files present."""
    built_from = {
        "item_daily": [PARQUET_FILES["bq_item_daily"]],
        "product_dim": [PARQUET_FILES["bq_item_daily"], PARQUET_FILES["bq_item_attrs"]],
        "promo_daily": [PARQUET_FILES["bq_promo_daily"]],
    }
    csv_name = (overview or {}).get("source_csv") or config.CSV_DEFAULT.name
    files: dict[str, dict] = {}
    for name, entry in written.items():
        files[name] = {**entry, "built_from": built_from[name]}
    for name in (*SITE_SNAPSHOTS, "trends_daily"):
        path = data_dir / PARQUET_FILES[name]
        if not path.exists():
            continue
        if name == "trends_daily":
            origin = _trend_sources(path)
        else:
            origin = [csv_name]
        files[name] = {
            "file": PARQUET_FILES[name],
            "rows": _parquet_rows(path),
            "sha256": file_sha256(path),
            "built_from": origin,
        }
    return files


def periods_block() -> dict[str, dict]:
    return {
        k: {"label": p.label, "start": p.start.isoformat(), "end": p.end.isoformat(), "days": p.days}
        for k, p in config.PERIODS.items()
    }


def build_summary(item_daily: pd.DataFrame, product_dim: pd.DataFrame) -> dict:
    """Totals printed by the builder and stored in the manifest."""
    views = int(item_daily["views"].sum())
    by_group = product_dim.groupby("category_group")["total_views"].sum()
    share = {g: (float(by_group.get(g, 0)) / views if views else 0.0) for g in config.CATEGORY_GROUPS}
    viewed = product_dim[product_dim["total_views"] > 0]
    pooled = float(viewed["total_purchases"].sum()) / float(viewed["total_views"].sum()) if len(viewed) else 0.0
    return {
        "products": int(len(product_dim)),
        "products_viewed": int(len(viewed)),
        "products_purchase_only": int(((product_dim["total_views"] == 0) & (product_dim["total_purchases"] > 0)).sum()),
        "views": views,
        "purchases": int(item_daily["purchases"].sum()),
        "revenue_usd": round(float(item_daily["revenue_usd"].sum()), 2),
        "pooled_conversion": pooled,
        "view_share_by_group": share,
        "other_share": share.get(taxonomy.OTHER, 0.0),
        "dates": int(item_daily["event_date"].nunique()),
    }


# --------------------------------------------------------------------------- main


def build(data_dir: Path, *, product_key: str | None = None, view_metric: str | None = None, aliases: bool = True) -> dict:
    """Build every snapshot in ``data_dir`` and return the manifest."""
    profile = read_json(data_dir / JSON_FILES["profile"], default={}) or {}
    decisions = {**DEFAULT_DECISIONS, **(profile.get("decisions") or {})}
    overrides = {}
    if product_key and product_key != decisions["product_key"]:
        overrides["product_key"] = {"profile": decisions["product_key"], "used": product_key}
    if view_metric and view_metric != decisions["view_metric"]:
        overrides["view_metric"] = {"profile": decisions["view_metric"], "used": view_metric}
    product_key = product_key or decisions["product_key"]
    view_metric = view_metric or decisions["view_metric"]
    if product_key not in PRODUCT_KEYS or view_metric not in VIEW_METRICS:
        raise ValueError(f"unsupported decisions: product_key={product_key!r}, view_metric={view_metric!r}")
    decisions.update(product_key=product_key, view_metric=view_metric)
    if overrides:
        decisions["overrides"] = overrides
    use_aliases = aliases and product_key == "name_key"

    raw = read_snapshot(data_dir, "bq_item_daily")
    attrs = read_snapshot(data_dir, "bq_item_attrs")
    promo = read_snapshot(data_dir, "bq_promo_daily", required=False)
    overview = read_json(data_dir / JSON_FILES["overview"])

    raw["product_key"] = add_product_key(raw, product_key, use_aliases)
    attrs["product_key"] = add_product_key(attrs, product_key, use_aliases)
    unkeyed = raw[raw["product_key"].isna()]
    device_map, device_report = device_plan(raw["device_category"], raw[view_metric])

    item_daily = build_item_daily(raw, view_metric, device_map)
    product_dim = build_product_dim(raw, attrs, item_daily, view_metric)
    written = {
        "item_daily": write_snapshot(item_daily, "item_daily", data_dir),
        "product_dim": write_snapshot(product_dim, "product_dim", data_dir),
    }
    if promo is not None:
        promo_daily = build_promo_daily(promo, product_key, use_aliases, device_map)
        written["promo_daily"] = write_snapshot(promo_daily, "promo_daily", data_dir)

    applied = {}
    if use_aliases:
        seen_keys = set(taxonomy.name_key(n) for n in raw["item_name"].dropna().unique())
        applied = {k: v for k, v in taxonomy.NAME_ALIASES.items() if k in seen_keys}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "bigquery+csv",
        "generated_at": None,
        "decisions": decisions,
        "product_key": product_key,
        "view_metric": view_metric,
        "name_aliases": applied,
        "category_colors": category_colors(product_dim),
        "files": file_entries(data_dir, written, overview),
        "periods": periods_block(),
        "placeholders": list(config.PLACEHOLDERS),
        "devices": {"kept": sorted({d for d in device_map.values() if d}), "unknown": device_report},
        "dropped_rows": {
            "no_product_key": int(len(unkeyed)),
            "no_product_key_views": int(unkeyed[view_metric].sum()),
            "no_product_key_purchases": int(unkeyed["purchases"].sum()),
        },
        "summary": build_summary(item_daily, product_dim),
    }
    manifest_path = data_dir / JSON_FILES["manifest"]
    previous = read_json(manifest_path, default=None)
    if isinstance(previous, dict) and {**previous, "generated_at": None} == manifest:
        manifest["generated_at"] = previous.get("generated_at")  # unchanged inputs: keep the file identical
    else:
        manifest["generated_at"] = utc_now_iso()
    write_json_atomic(manifest, manifest_path)
    return manifest


def print_summary(manifest: dict) -> None:
    s = manifest["summary"]
    print(f"product_key={manifest['product_key']} view_metric={manifest['view_metric']} aliases={len(manifest['name_aliases'])}")
    print(
        f"products {s['products']:,} ({s['products_viewed']:,} viewed, {s['products_purchase_only']} purchase-only)"
        f" | dates {s['dates']}"
    )
    print(
        f"views {s['views']:,} | purchases {s['purchases']:,} | pooled conversion {s['pooled_conversion']:.3%}"
        f" | revenue ${s['revenue_usd']:,.0f}"
    )
    print("view share by category group:")
    colors = manifest["category_colors"]
    for group, share in sorted(s["view_share_by_group"].items(), key=lambda kv: -kv[1]):
        tag = "" if colors.get(group) == config.PALETTE["other"] else f"  [{colors[group]}]"
        print(f"  {group:<26} {share:7.2%}{tag}")
    print(f"'Other' share {s['other_share']:.2%}")
    for d in manifest["devices"]["unknown"]:
        print(f"device {d['device']!r}: {d['views']:,} views ({d['share']:.3%}) {d['action']}")
    dropped = manifest["dropped_rows"]
    if dropped["no_product_key"]:
        print(
            f"dropped {dropped['no_product_key']:,} rows without a product key "
            f"({dropped['no_product_key_views']:,} views, {dropped['no_product_key_purchases']:,} purchases)"
        )
    built = [e for k, e in manifest["files"].items() if k not in (*SITE_SNAPSHOTS, "trends_daily")]
    print("wrote " + ", ".join(f"{e['file']} ({e['rows']:,} rows)" for e in built) + ", manifest.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="defaults to $GA4APP_DATA_DIR or data/")
    ap.add_argument("--product-key", choices=PRODUCT_KEYS, default=None, help="override profile.json decisions")
    ap.add_argument("--view-metric", choices=VIEW_METRICS, default=None, help="override profile.json decisions")
    ap.add_argument("--no-aliases", action="store_true", help="do not fold legacy purchase names (name_key only)")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.resolve_data_dir()
    for name in ("bq_item_daily", "bq_item_attrs"):
        if not (data_dir / PARQUET_FILES[name]).exists():
            print(f"error: {data_dir / PARQUET_FILES[name]} not found (run scripts/export_bigquery.py)")
            return 1
    manifest = build(data_dir, product_key=args.product_key, view_metric=args.view_metric, aliases=not args.no_aliases)
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Snapshot data contracts: column schemas, validation, typed atomic writers and readers.

Every parquet file the pipeline writes and the app reads is described in ``SCHEMAS``.
A column spec is a kind (``date``, ``str``, ``int``, ``float``, ``bool``, ``datetime``);
a trailing ``?`` marks the column nullable. Writers cast to fixed Arrow types so files are
identical no matter which pandas dtypes produced them, and write through a temp file +
``os.replace`` (OneDrive can briefly lock files, so the replace is retried).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

_COUNTS = (
    "view_rows",
    "view_events",
    "primary_views",
    "add_to_carts",
    "checkouts",
    "purchases",
    "purchase_events_raw",
    "purchases_missing_txn",
    "units",
    "promo_views",
    "promo_clicks",
)

SCHEMAS: dict[str, dict[str, str]] = {
    # ---- raw BigQuery exports (data/raw/) -------------------------------------------
    "bq_item_daily": {
        "event_date": "date",
        "device_category": "str",
        "item_id": "str?",
        "item_name": "str?",
        **{c: "int" for c in _COUNTS},
        "revenue_usd": "float",
    },
    "bq_item_attrs": {
        "item_id": "str?",
        "item_name": "str?",
        "item_category": "str?",
        "item_category2": "str?",
        "item_brand": "str?",
        "item_rows": "int",
        "view_rows": "int",
        "purchase_rows": "int",
        "median_price_usd": "float?",
        "first_seen": "date",
        "last_seen": "date",
    },
    "bq_promo_daily": {
        "event_date": "date",
        "device_category": "str",
        "event_name": "str",
        "promotion_id": "str?",
        "promotion_name": "str?",
        "creative_name": "str?",
        "creative_slot": "str?",
        "item_id": "str?",
        "item_name": "str?",
        "item_rows": "int",
        "events": "int",
    },
    # ---- app snapshots (data/) ------------------------------------------------------
    "item_daily": {
        "event_date": "date",
        "device_category": "str",
        "product_key": "str",
        "views": "int",  # copy of the view metric chosen by profiling
        **{c: "int" for c in _COUNTS},
        "revenue_usd": "float",
    },
    "product_dim": {
        "product_key": "str",
        "item_name": "str",
        "item_ids": "str",
        "n_item_ids": "int",
        "n_names": "int",
        "category_raw": "str?",
        "category_path": "str",
        "category_group": "str",
        "brand": "str",
        "median_price_usd": "float?",
        "first_seen": "date",
        "last_seen": "date",
        "total_views": "int",
        "total_purchases": "int",
    },
    "promo_daily": {
        "event_date": "date",
        "device_category": "str",
        "promotion_name": "str",
        "creative_slot": "str?",
        "product_key": "str?",
        "promo_views": "int",
        "promo_clicks": "int",
    },
    "site_daily": {
        "event_date": "date",
        "device_category": "str",
        "event_name": "str",
        "events": "int",
        "users": "int",
        "sessions": "int",
    },
    "site_funnel": {
        "event_date": "date",
        "device_category": "str",
        "sessions": "int",
        "view_sessions": "int",
        "cart_sessions": "int",
        "checkout_sessions": "int",
        "purchase_sessions": "int",
        "view_purchase_sessions": "int",
    },
    "trends_daily": {
        "date": "date",
        "keyword": "str",
        "group_type": "str",
        "group_value": "str",
        "geo": "str",
        "interest": "float",
        "is_partial": "bool",
        "source": "str",
        "fetched_at": "datetime",
    },
}

# File names, relative to the data dir.
PARQUET_FILES = {
    "bq_item_daily": "raw/bq_item_daily.parquet",
    "bq_item_attrs": "raw/bq_item_attrs.parquet",
    "bq_promo_daily": "raw/bq_promo_daily.parquet",
    "item_daily": "item_daily.parquet",
    "product_dim": "product_dim.parquet",
    "promo_daily": "promo_daily.parquet",
    "site_daily": "site_daily.parquet",
    "site_funnel": "site_funnel.parquet",
    "trends_daily": "trends_daily.parquet",
}
JSON_FILES = {
    "overview": "overview.json",
    "profile": "profile.json",
    "bq_proof": "bq_proof.json",
    "trends_meta": "trends_meta.json",
    "manifest": "manifest.json",
    "export_log": "raw/export_log.json",
    "profile_raw": "raw/profile_raw.json",
}
REQUIRED_SNAPSHOTS = ("item_daily", "product_dim", "site_daily")
OPTIONAL_SNAPSHOTS = ("promo_daily", "site_funnel", "trends_daily")

# Rows sampled when checking that an object column really holds dates.
_DATE_SAMPLE = 2000


class ContractError(ValueError):
    """A frame or file does not match its declared schema."""


def parse_spec(spec: str) -> tuple[str, bool]:
    """``"int?"`` -> ``("int", True)``."""
    return (spec[:-1], True) if spec.endswith("?") else (spec, False)


# --------------------------------------------------------------------------- validation


def _is_date_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        sample = s.dropna().head(_DATE_SAMPLE)
        return all(isinstance(v, date) for v in sample)
    return False


def _kind_ok(s: pd.Series, kind: str) -> bool:
    if kind == "date":
        return _is_date_like(s)
    if kind == "datetime":
        return pd.api.types.is_datetime64_any_dtype(s)
    if kind == "int":
        return pd.api.types.is_integer_dtype(s)
    if kind == "float":
        return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(s)
    if kind == "str":
        if isinstance(s.dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_string_dtype(s):
            return True
        if s.dtype == object:
            return all(isinstance(v, str) for v in s.dropna().head(_DATE_SAMPLE))
        return False
    raise ContractError(f"unknown column kind {kind!r}")


def validate_frame(df: pd.DataFrame, name: str) -> None:
    """Raise ``ContractError`` listing every column/dtype/null/invariant problem."""
    if name not in SCHEMAS:
        raise ContractError(f"no schema named {name!r}")
    problems: list[str] = []
    for col, spec in SCHEMAS[name].items():
        kind, nullable = parse_spec(spec)
        if col not in df.columns:
            problems.append(f"missing column {col!r}")
            continue
        s = df[col]
        if not _kind_ok(s, kind):
            problems.append(f"{col!r} should be {kind}, got {s.dtype}")
        if not nullable and len(s) and s.isna().any():
            problems.append(f"{col!r} has {int(s.isna().sum())} nulls but is not nullable")
    if not problems:
        problems.extend(_invariants(df, name))
    if problems:
        raise ContractError(f"{name}: " + "; ".join(problems))


def _invariants(df: pd.DataFrame, name: str) -> list[str]:
    out: list[str] = []
    if name in ("item_daily", "bq_item_daily"):
        counts = [c for c in (*_COUNTS, "views") if c in df.columns]
        neg = [c for c in counts if (df[c] < 0).any()]
        if neg:
            out.append(f"negative counts in {neg}")
        if (df["revenue_usd"] < -1e-9).any():
            out.append("negative revenue_usd")
        if (df["view_events"] > df["view_rows"]).any():
            out.append("view_events > view_rows")
        if (df["purchases"] > df["purchase_events_raw"]).any():
            out.append("purchases > purchase_events_raw")
    if name == "item_daily":
        key = ["event_date", "device_category", "product_key"]
        dupes = int(df.duplicated(key).sum())
        if dupes:
            out.append(f"{dupes} duplicate (event_date, device_category, product_key) rows")
    if name == "product_dim" and df["product_key"].duplicated().any():
        out.append("duplicate product_key")
    if name == "site_daily":
        if (df[["events", "users", "sessions"]] < 0).to_numpy().any():
            out.append("negative site counts")
    return out


# --------------------------------------------------------------------------- arrow typing


def _arrow_type(kind: str):
    import pyarrow as pa

    return {
        "date": pa.date32(),
        "str": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
        "datetime": pa.timestamp("us", tz="UTC"),
    }[kind]


def _column_values(s: pd.Series, kind: str) -> Any:
    if kind == "date":
        conv = pd.to_datetime(s, errors="raise")
        return [None if pd.isna(v) else v.date() for v in conv]
    if kind == "datetime":
        conv = pd.to_datetime(s, utc=True)
        return [None if pd.isna(v) else v.to_pydatetime() for v in conv]
    if kind == "str":
        return [None if (v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA) else str(v) for v in s.astype(object)]
    if kind == "int":
        return s.astype("Int64").astype(object).where(s.notna(), None).tolist()
    if kind == "float":
        return [None if pd.isna(v) else float(v) for v in s]
    if kind == "bool":
        return [None if pd.isna(v) else bool(v) for v in s]
    raise ContractError(f"unknown column kind {kind!r}")


def to_arrow_table(df: pd.DataFrame, name: str):
    """Validate ``df`` and return a pyarrow Table with exactly the schema's columns/types."""
    import pyarrow as pa

    validate_frame(df, name)
    arrays, fields = [], []
    for col, spec in SCHEMAS[name].items():
        kind, nullable = parse_spec(spec)
        typ = _arrow_type(kind)
        arrays.append(pa.array(_column_values(df[col], kind), type=typ))
        fields.append(pa.field(col, typ, nullable=nullable))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def normalize_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Apply read-side dtype conventions: dates -> ``datetime64[ns]``, ids stay ``str``."""
    out = df.copy()
    for col, spec in SCHEMAS[name].items():
        kind, _ = parse_spec(spec)
        if col not in out.columns:
            continue
        if kind == "date":
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
        elif kind == "datetime":
            out[col] = pd.to_datetime(out[col], utc=True)
        elif kind == "int" and not parse_spec(spec)[1]:
            out[col] = out[col].astype("int64")
    return out


# --------------------------------------------------------------------------- file IO


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _replace_with_retry(tmp: Path, dest: Path, attempts: int = 8) -> None:
    for i in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:  # OneDrive / antivirus briefly holding the file
            if i == attempts - 1:
                raise
            time.sleep(0.25 * (i + 1))


def write_parquet_atomic(obj: Any, path: Path | str) -> str:
    """Write a DataFrame or pyarrow Table to ``path`` atomically; return its sha256."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    table = obj if isinstance(obj, pa.Table) else pa.Table.from_pandas(obj, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    _replace_with_retry(tmp, path)
    return file_sha256(path)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return sorted(o) if isinstance(o, (set, frozenset)) else list(o)
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_json_atomic(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")
    _replace_with_retry(tmp, path)


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_snapshot(df: pd.DataFrame, name: str, data_dir: Path | str) -> dict:
    """Validate + write a named parquet snapshot. Returns its manifest entry."""
    path = Path(data_dir) / PARQUET_FILES[name]
    sha = write_parquet_atomic(to_arrow_table(df, name), path)
    return {"file": PARQUET_FILES[name], "rows": int(len(df)), "sha256": sha}


def read_snapshot(data_dir: Path | str, name: str, *, required: bool = True) -> pd.DataFrame | None:
    """Read a named parquet snapshot, normalise dtypes and validate it."""
    import pyarrow.parquet as pq

    path = Path(data_dir) / PARQUET_FILES[name]
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    df = pq.read_table(path).to_pandas()
    df = normalize_frame(df, name)
    validate_frame(df, name)
    return df


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

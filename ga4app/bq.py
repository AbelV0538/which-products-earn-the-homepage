"""BigQuery access for the snapshot pipeline.

Authentication uses ``pydata-google-auth`` (a one-time browser sign-in, cached under
``%APPDATA%/.config/itcs5122_team9``) so no gcloud install is needed. Application Default
Credentials are used instead when ``GOOGLE_APPLICATION_CREDENTIALS`` is set or ``--adc`` is
passed. Every query is dry-run first; real runs carry a ``maximum_bytes_billed`` cap that
respects BigQuery's 10 MiB-per-referenced-table billing floor (``events_*`` = 92 shards, so
every query bills at least ~0.9 GiB).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import config
from .contracts import read_json, write_json_atomic

CRED_CACHE_DIR = "itcs5122_team9"
CRED_CACHE_FILE = "bq_user_credentials.json"
GIB = 1024**3


class BudgetExceeded(RuntimeError):
    """A dry run estimated more bytes than the caller allowed."""


# --------------------------------------------------------------------------- SQL + project


def read_sql(name: str) -> str:
    return (config.SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")


def sql_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def resolve_project(cli_value: str | None = None) -> str | None:
    """GCP project id: CLI > env > .streamlit/secrets.toml [gcp] > config/local.json."""
    if cli_value:
        return cli_value.strip()
    env = os.environ.get(config.GCP_PROJECT_ENV, "").strip()
    if env:
        return env
    if config.SECRETS_TOML.exists():
        try:
            secrets = tomllib.loads(config.SECRETS_TOML.read_text(encoding="utf-8"))
            proj = (secrets.get("gcp") or {}).get("project_id")
            if proj:
                return str(proj).strip()
        except (tomllib.TOMLDecodeError, OSError):
            pass
    local = read_json(config.LOCAL_CONFIG, default={}) or {}
    return local.get("gcp_project") or None


def remember_project(project: str) -> None:
    local = read_json(config.LOCAL_CONFIG, default={}) or {}
    local["gcp_project"] = project
    write_json_atomic(local, config.LOCAL_CONFIG)


# --------------------------------------------------------------------------- auth + client


def get_credentials(*, reauth: bool = False, use_local_webserver: bool = True, prefer_adc: bool = False):
    """Return Google credentials with the BigQuery scope.

    First call opens a browser for Google sign-in (local redirect on ports 8080-8089) and
    caches a refresh token; later calls reuse the cache silently.
    """
    scopes = list(config.BQ_SCOPES)
    if prefer_adc or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        import google.auth

        creds, _ = google.auth.default(scopes=scopes)
        return creds

    import pydata_google_auth
    from pydata_google_auth import cache

    cache_cls = cache.WriteOnlyCredentialsCache if reauth else cache.ReadWriteCredentialsCache
    return pydata_google_auth.get_user_credentials(
        scopes,
        credentials_cache=cache_cls(dirname=CRED_CACHE_DIR, filename=CRED_CACHE_FILE),
        use_local_webserver=use_local_webserver,
    )


def get_client(project: str, credentials: Any = None):
    from google.cloud import bigquery

    return bigquery.Client(project=project, credentials=credentials, location=config.BQ_LOCATION)


# --------------------------------------------------------------------------- dry run + caps


@dataclass(frozen=True)
class DryRun:
    name: str
    bytes_processed: int
    referenced_tables: int

    @property
    def gib(self) -> float:
        return self.bytes_processed / GIB


def dry_run(client, sql: str, name: str = "query") -> DryRun:
    from google.cloud import bigquery

    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    n_tables = len(job.referenced_tables or []) or config.BQ_SHARDS
    return DryRun(name, int(job.total_bytes_processed or 0), int(n_tables))


def billing_floor(referenced_tables: int) -> int:
    return referenced_tables * config.BQ_MIN_BILL_PER_TABLE + config.BQ_MIN_BILL_PER_QUERY


def effective_cap(dr: DryRun, cap_bytes: int) -> int:
    """Byte cap for the real run: never below 1.2x the estimate or 1.1x the billing floor."""
    return int(max(cap_bytes, math.ceil(1.2 * dr.bytes_processed), math.ceil(1.1 * billing_floor(dr.referenced_tables))))


def check_budget(dry_runs: Iterable[DryRun], *, max_gib_per_query: float, max_gib_total: float, assume_yes: bool) -> None:
    runs = list(dry_runs)
    over = [d for d in runs if d.gib > max_gib_per_query]
    total = sum(d.gib for d in runs)
    if (over or total > max_gib_total) and not assume_yes:
        names = ", ".join(f"{d.name} ({d.gib:.2f} GiB)" for d in over) or "none individually"
        raise BudgetExceeded(
            f"estimated {total:.2f} GiB total (limit {max_gib_total} GiB); over the per-query "
            f"limit of {max_gib_per_query} GiB: {names}. Re-run with --yes or raise --max-gib."
        )


def print_dry_runs(dry_runs: Iterable[DryRun]) -> None:
    runs = list(dry_runs)
    width = max((len(d.name) for d in runs), default=5)
    print(f"{'query':<{width}}  {'est. GiB':>9}  tables  billed floor GiB")
    for d in runs:
        print(f"{d.name:<{width}}  {d.gib:>9.3f}  {d.referenced_tables:>6}  {billing_floor(d.referenced_tables) / GIB:>16.3f}")
    print(f"{'TOTAL':<{width}}  {sum(d.gib for d in runs):>9.3f}")


# --------------------------------------------------------------------------- run


def run_to_arrow(client, sql: str, *, cap_bytes: int, labels: dict[str, str] | None = None, timeout_s: float = 900):
    """Run ``sql`` under a byte cap; return ``(pyarrow.Table, job_meta)``."""
    from google.cloud import bigquery

    job_labels = {**config.BQ_JOB_LABELS, **(labels or {})}
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=int(cap_bytes), use_query_cache=True, labels=job_labels)
    job = client.query(sql, job_config=cfg)
    rows = job.result(timeout=timeout_s)
    table = rows.to_arrow(create_bqstorage_client=False)
    meta = {
        "job_id": job.job_id,
        "location": job.location,
        "bytes_processed": int(job.total_bytes_processed or 0),
        "bytes_billed": int(job.total_bytes_billed or 0),
        "cache_hit": bool(job.cache_hit),
        "rows": int(table.num_rows),
        "cap_bytes": int(cap_bytes),
        "sql_sha256": sql_sha256(sql),
        "labels": job_labels,
    }
    return table, meta


def append_log(entry: dict, log_path: Path) -> None:
    log = read_json(log_path, default=[]) or []
    log.append(entry)
    write_json_atomic(log, log_path)


def to_jsonable_rows(table) -> list[dict]:
    """Arrow table -> list of plain dicts (nested STRUCT/ARRAY results from profiling)."""
    return json.loads(json.dumps(table.to_pylist(), default=str))

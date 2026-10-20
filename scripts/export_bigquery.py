"""Export the item-level tables from BigQuery into data/raw/*.parquet.

Usage:
    python scripts/export_bigquery.py [--project ID] [--max-gib 6] [--total-gib 15]
                                      [--only export_item_daily ...] [--dry-run-only] [--yes]

Each query is dry-run first; the run is refused when an estimate exceeds the limits (unless
--yes). Results are validated against ga4app.contracts and written atomically; every job is
appended to data/raw/export_log.json. Run scripts/profile_bigquery.py first, then
scripts/build_snapshots.py to turn the raw exports into the app snapshots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga4app import bq, config  # noqa: E402
from ga4app.contracts import JSON_FILES, utc_now_iso, write_snapshot  # noqa: E402

# (sql file, raw snapshot name)
EXPORTS = (
    ("export_item_daily", "bq_item_daily"),
    ("export_item_attrs", "bq_item_attrs"),
    ("export_promo_daily", "bq_promo_daily"),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project")
    ap.add_argument("--max-gib", type=float, default=6.0, help="per-query dry-run limit")
    ap.add_argument("--total-gib", type=float, default=15.0, help="total dry-run limit")
    ap.add_argument("--only", nargs="*", help="subset of export sql names")
    ap.add_argument("--dry-run-only", action="store_true")
    ap.add_argument("--yes", action="store_true", help="run even if the estimates exceed the limits")
    args = ap.parse_args(argv)

    project = bq.resolve_project(args.project)
    if not project:
        print("error: no GCP project id; run scripts/bq_auth.py --project <id> first.")
        return 2
    exports = [e for e in EXPORTS if not args.only or e[0] in args.only]
    client = bq.get_client(project, bq.get_credentials())
    sqls = {sql_name: bq.read_sql(sql_name) for sql_name, _ in exports}

    dry = [bq.dry_run(client, sqls[s], s) for s, _ in exports]
    bq.print_dry_runs(dry)
    bq.check_budget(dry, max_gib_per_query=args.max_gib, max_gib_total=args.total_gib, assume_yes=args.yes)
    if args.dry_run_only:
        return 0

    data_dir = config.resolve_data_dir()
    log_path = data_dir / JSON_FILES["export_log"]
    for d, (sql_name, snap) in zip(dry, exports):
        cap = bq.effective_cap(d, int(args.max_gib * bq.GIB))
        print(f"running {sql_name} (cap {cap / bq.GIB:.2f} GiB)...", flush=True)
        table, meta = bq.run_to_arrow(client, sqls[sql_name], cap_bytes=cap, labels={"step": sql_name})
        df = table.to_pandas()
        entry = write_snapshot(df, snap, data_dir)
        bq.append_log({"step": sql_name, "snapshot": snap, "run_at": utc_now_iso(), "project": project, **meta, **entry}, log_path)
        print(f"  -> {entry['file']}: {entry['rows']:,} rows, billed {meta['bytes_billed'] / bq.GIB:.2f} GiB")
    print("Done. Next: python scripts/build_snapshots.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

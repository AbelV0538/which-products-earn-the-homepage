"""One-time BigQuery sign-in + the pitch's "proof it loads" query.

Usage:
    python scripts/bq_auth.py --project <GCP_PROJECT_ID> [--reauth] [--adc]

The first run opens a browser for Google sign-in (the redirect lands on a local port
8080-8089) and caches a refresh token so later scripts run without a browser. It then runs
the proof query from pitch slide 5 and checks it returns 4,295,584 events / 270,154 users /
92 days. The result is written to data/bq_proof.json and the project id is remembered in
config/local.json (git-ignored).

Any Google Cloud project works, including a free BigQuery sandbox project; the BigQuery API
must be enabled on it (https://console.cloud.google.com/apis/library/bigquery.googleapis.com).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga4app import bq, config  # noqa: E402
from ga4app.contracts import JSON_FILES, utc_now_iso, write_json_atomic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="GCP project id that runs (and is billed for) the queries")
    ap.add_argument("--reauth", action="store_true", help="force a fresh browser sign-in")
    ap.add_argument("--adc", action="store_true", help="use Application Default Credentials (gcloud / service account)")
    ap.add_argument("--max-gib", type=float, default=3.0, help="refuse to run if the dry run estimates more than this")
    args = ap.parse_args(argv)

    project = bq.resolve_project(args.project)
    if not project:
        print("error: no GCP project id. Pass --project <id> (any project with the BigQuery API enabled).")
        return 2

    print(f"Project: {project}")
    print("Signing in to Google - a browser window opens on the first run...", flush=True)
    try:
        creds = bq.get_credentials(reauth=args.reauth, prefer_adc=args.adc)
        client = bq.get_client(project, creds)
        sql = bq.read_sql("proof_counts")
        dr = bq.dry_run(client, sql, "proof_counts")
    except Exception as exc:  # surface auth / API-enablement problems with guidance
        print(f"\nBigQuery access failed: {type(exc).__name__}: {exc}")
        print(
            "\nChecks:\n"
            "  1. Is the BigQuery API enabled on the project?\n"
            "     https://console.cloud.google.com/apis/library/bigquery.googleapis.com\n"
            "  2. Is the project id right (not the project name or number)?\n"
            "  3. Retry with --reauth to sign in again, or --adc after `gcloud auth application-default login`."
        )
        return 1

    bq.print_dry_runs([dr])
    bq.check_budget([dr], max_gib_per_query=args.max_gib, max_gib_total=args.max_gib, assume_yes=False)
    cap = bq.effective_cap(dr, int(args.max_gib * bq.GIB))
    table, meta = bq.run_to_arrow(client, sql, cap_bytes=cap, labels={"step": "proof_counts"})
    row = {k: int(v) for k, v in table.to_pylist()[0].items()}
    ok = all(row.get(k) == v for k, v in config.PROOF_EXPECTED.items())

    print("\nProof query result (pitch slide 5):")
    for k, v in config.PROOF_EXPECTED.items():
        mark = "OK " if row.get(k) == v else "MISMATCH"
        print(f"  {k:<12} {row.get(k):>12,}   expected {v:>12,}   {mark}")
    print(f"  bytes billed: {meta['bytes_billed'] / bq.GIB:.3f} GiB  (cache hit: {meta['cache_hit']})")

    out = config.resolve_data_dir() / JSON_FILES["bq_proof"]
    write_json_atomic(
        {**row, "matches_pitch": ok, "expected": config.PROOF_EXPECTED, "sql": sql, "project": project, "run_at": utc_now_iso(), **meta},
        out,
    )
    bq.remember_project(project)
    print(f"\nWrote {out}")
    print("BigQuery access works." if ok else "WARNING: counts differ from the pitch.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

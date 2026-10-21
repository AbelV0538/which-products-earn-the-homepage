"""Fetch the Google Trends signal: one daily series per keyword in config/trends_keywords.csv.

Usage:
    python scripts/fetch_trends.py [--data-dir data] [--keywords config/trends_keywords.csv]
                                   [--timeframe "2020-10-01 2021-01-31"] [--geo US]
                                   [--methods trendspy,pytrends,manual_csv] [--refresh]
                                   [--sleep-min 20] [--sleep-max 40] [--max-tries 3] [--backoff-base 30]

Per keyword: the raw cache data/raw/trends/<slug>.json (resume), else trendspy, else pytrends,
else a CSV downloaded by hand from trends.google.com into data/trends_manual/ (matched by the
keyword in its header). One keyword per request, because Google normalises every request to
its own 0-100 peak. Requests are spaced by a jittered pause; a 429 backs off exponentially,
at most --max-tries tries per method. Missing keywords are recorded, never synthesised.

Outputs (in the data dir):
  trends_daily.parquet   date x keyword -> interest (0-100, "<1" = 0.5), with provenance
  trends_meta.json       keywords, geo, timeframe, method_used, errors, status ok|partial|unavailable
  manifest.json          files.trends_daily is updated if the manifest already exists

Exits 0 even when some or all keywords are missing; trends_meta.json records the status.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga4app import config, trends  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="defaults to $GA4APP_DATA_DIR or data/")
    ap.add_argument("--keywords", default=str(config.TRENDS_KEYWORDS_CSV), help="keyword map CSV")
    ap.add_argument("--timeframe", default=trends.DEFAULT_TIMEFRAME)
    ap.add_argument("--geo", default=trends.DEFAULT_GEO)
    ap.add_argument("--methods", default=",".join(trends.ALL_METHODS), help="comma-separated, in order")
    ap.add_argument("--refresh", action="store_true", help="ignore the raw cache and fetch again")
    ap.add_argument("--sleep-min", type=float, default=20.0, help="seconds between network requests (min)")
    ap.add_argument("--sleep-max", type=float, default=40.0, help="seconds between network requests (max)")
    ap.add_argument("--max-tries", type=int, default=3, help="tries per method on HTTP 429")
    ap.add_argument("--backoff-base", type=float, default=30.0, help="first 429 backoff in seconds; doubles")
    args = ap.parse_args(argv)

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in methods if m not in trends.ALL_METHODS]
    if unknown:
        print(f"error: unknown methods {unknown}; choose from {', '.join(trends.ALL_METHODS)}")
        return 2
    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.resolve_data_dir()
    keywords = Path(args.keywords)
    if not keywords.exists():
        print(f"error: {keywords} not found")
        return 2

    t0 = time.time()
    print(f"Google Trends: {keywords.name} | geo {args.geo or 'worldwide'} | {args.timeframe} | methods {', '.join(methods)}", flush=True)
    meta = trends.run_fetch(
        data_dir,
        keywords_csv=keywords,
        timeframe=args.timeframe,
        geo=args.geo,
        methods=methods,
        pacer=trends.Pacer(args.sleep_min, args.sleep_max),
        max_tries=max(1, args.max_tries),
        backoff_base=args.backoff_base,
        refresh=args.refresh,
        log=lambda msg: print(msg, flush=True),
    )

    print(f"status {meta['status']}: {len(meta['method_used'])}/{len(meta['keywords'])} keywords, {meta['rows']:,} rows")
    for kw, method in meta["method_used"].items():
        note = f"  ({meta['warnings'][kw]})" if kw in meta["warnings"] else ""
        print(f"  {kw:<22} {method}{note}")
    if meta["errors"]:
        print("missing keywords: download each CSV by hand (Explore page -> Interest over time -> download),")
        print(f"save it in {data_dir / config.TRENDS_MANUAL_SUBDIR}, then re-run with --methods manual_csv:")
        for kw, url in meta["manual_urls"].items():
            print(f"  {kw:<22} {url}")
    written = ["trends_daily.parquet"] if meta["trends_daily_written"] else []
    written.append("trends_meta.json")
    if meta["manifest_updated"]:
        written.append("manifest.json")
    print(f"wrote {', '.join(written)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

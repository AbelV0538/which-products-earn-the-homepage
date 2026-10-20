"""Aggregate the event-level CSV export into small site-level snapshots with DuckDB.

Usage:
    python scripts/build_site_aggregates.py [--csv ga4_events_export.csv] [--data-dir data]

Outputs (in the data dir):
  site_daily.parquet   event_date x device x event_name -> events, distinct users, distinct sessions
  site_funnel.parquet  sessions by start date x device with view / cart / checkout / purchase flags
  overview.json        totals that re-prove the pitch (4,295,584 events, 270,154 users, 92 days),
                       events by device and name, top countries, and per-event tracking gaps

The CSV has no item columns; it supplies the site-level context and the proof counts.
All id columns are read as text (user_pseudo_id looks like a float and would collide).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ga4app import config  # noqa: E402
from ga4app.contracts import JSON_FILES, utc_now_iso, write_json_atomic, write_snapshot  # noqa: E402

CSV_COLUMNS = {
    "event_date": "DATE",
    "event_timestamp": "VARCHAR",
    "event_name": "VARCHAR",
    "user_pseudo_id": "VARCHAR",
    "ga_session_id": "VARCHAR",
    "country": "VARCHAR",
    "device_category": "VARCHAR",
}
GAP_SHARE = 0.10  # a day is a tracking gap when its count is below 10% of the event's median day


def tracking_gaps(site_daily: pd.DataFrame) -> dict:
    """Per event: days whose count is < 10% of the median day, and the first reliable date."""
    daily = site_daily.groupby(["event_name", "event_date"], as_index=False)["events"].sum()
    all_days = pd.date_range(config.DATA_START, config.DATA_END, freq="D")
    out = {}
    for name, g in daily.groupby("event_name"):
        s = g.set_index("event_date")["events"].reindex(all_days, fill_value=0)
        med = float(s.median())
        gaps = s[s < GAP_SHARE * med].index if med > 0 else s.index[:0]
        if len(gaps):
            last_gap = gaps.max()
            later = s.index[s.index > last_gap]
            first_reliable = later.min().date().isoformat() if len(later) else None
        else:
            first_reliable = all_days[0].date().isoformat()
        out[name] = {
            "median_daily": med,
            "gap_days": int(len(gaps)),
            "gap_dates": [d.date().isoformat() for d in gaps],
            "first_reliable_date": first_reliable,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(config.CSV_DEFAULT))
    ap.add_argument("--data-dir", default=None, help="defaults to $GA4APP_DATA_DIR or data/")
    args = ap.parse_args(argv)

    import duckdb

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found")
        return 2
    data_dir = Path(args.data_dir) if args.data_dir else config.resolve_data_dir()
    t0 = time.time()
    con = duckdb.connect()
    cols = ", ".join(f"'{k}': '{v}'" for k, v in CSV_COLUMNS.items())
    con.execute(
        f"CREATE TABLE ev AS SELECT * FROM read_csv(?, header = true, columns = {{{cols}}})",
        [str(csv_path)],
    )
    con.execute("UPDATE ev SET device_category = LOWER(COALESCE(device_category, 'unknown'))")
    print(f"loaded {csv_path.name} in {time.time() - t0:.1f}s")

    totals = con.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT user_pseudo_id), COUNT(DISTINCT event_date),
               COUNT(DISTINCT user_pseudo_id || '|' || ga_session_id), MIN(event_date), MAX(event_date)
        FROM ev
        """
    ).fetchone()
    event_count, user_count, day_count, session_count, dmin, dmax = totals

    site_daily = con.execute(
        """
        SELECT event_date, device_category, event_name,
               COUNT(*) AS events,
               COUNT(DISTINCT user_pseudo_id) AS users,
               COUNT(DISTINCT user_pseudo_id || '|' || ga_session_id) AS sessions
        FROM ev
        GROUP BY ALL
        ORDER BY ALL
        """
    ).df()

    site_funnel = con.execute(
        """
        WITH s AS (
          SELECT user_pseudo_id, ga_session_id,
                 MIN(event_date) AS event_date,
                 arg_min(device_category, event_timestamp) AS device_category,
                 bool_or(event_name = 'view_item') AS v,
                 bool_or(event_name = 'add_to_cart') AS c,
                 bool_or(event_name = 'begin_checkout') AS k,
                 bool_or(event_name = 'purchase') AS p
          FROM ev
          GROUP BY 1, 2
        )
        SELECT event_date, device_category,
               COUNT(*) AS sessions,
               COUNT(*) FILTER (WHERE v) AS view_sessions,
               COUNT(*) FILTER (WHERE c) AS cart_sessions,
               COUNT(*) FILTER (WHERE k) AS checkout_sessions,
               COUNT(*) FILTER (WHERE p) AS purchase_sessions,
               COUNT(*) FILTER (WHERE v AND p) AS view_purchase_sessions
        FROM s
        GROUP BY ALL
        ORDER BY ALL
        """
    ).df()

    by_device = dict(con.execute("SELECT device_category, COUNT(*) FROM ev GROUP BY 1 ORDER BY 2 DESC").fetchall())
    by_name = dict(con.execute("SELECT event_name, COUNT(*) FROM ev GROUP BY 1 ORDER BY 2 DESC").fetchall())
    countries = con.execute(
        "SELECT country, COUNT(*) AS events, COUNT(DISTINCT user_pseudo_id) AS users FROM ev GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    con.close()

    for df in (site_daily, site_funnel):
        df["event_date"] = pd.to_datetime(df["event_date"])
    gaps = tracking_gaps(site_daily)

    entries = {
        "site_daily": write_snapshot(site_daily, "site_daily", data_dir),
        "site_funnel": write_snapshot(site_funnel, "site_funnel", data_dir),
    }
    overview = {
        "event_count": int(event_count),
        "user_count": int(user_count),
        "day_count": int(day_count),
        "session_count": int(session_count),
        "date_min": str(dmin),
        "date_max": str(dmax),
        "matches_pitch": {
            k: int(v) == config.PROOF_EXPECTED[k]
            for k, v in (("event_count", event_count), ("user_count", user_count), ("day_count", day_count))
        },
        "events_by_device": {k: int(v) for k, v in by_device.items()},
        "events_by_name": {k: int(v) for k, v in by_name.items()},
        "n_countries": int(len(countries)),
        "top_countries": [
            {"country": r.country, "events": int(r.events), "users": int(r.users)} for r in countries.head(15).itertuples()
        ],
        "tracking_gaps": gaps,
        "source_csv": csv_path.name,
        "files": entries,
        "generated_at": utc_now_iso(),
    }
    write_json_atomic(overview, data_dir / JSON_FILES["overview"])

    # Console cross-checks against the numbers verified while planning.
    print(f"events {event_count:,} | users {user_count:,} | days {day_count} | sessions {session_count:,}")
    tot = site_daily.pivot_table(index="event_date", columns="event_name", values="events", aggfunc="sum").fillna(0)
    for key in ("baseline", "cyber_week", "december", "full"):
        p = config.PERIODS[key]
        sl = tot.loc[str(p.start) : str(p.end)]
        conv = sl["purchase"].sum() / sl["view_item"].sum()
        print(f"  site purchase/view_item {p.label:<22} {conv:.2%}")
    print(f"  add_to_cart first reliable date: {gaps.get('add_to_cart', {}).get('first_reliable_date')}")
    print(f"wrote {', '.join(e['file'] for e in entries.values())}, overview.json in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

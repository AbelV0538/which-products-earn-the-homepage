"""Profile the GA4 items array before exporting, and record the data-driven decisions.

Usage:
    python scripts/profile_bigquery.py [--project ID] [--max-gib 3] [--total-gib 25]
                                       [--with-event-params] [--yes]

Runs the sql/profile_*.sql queries (dry-run first, byte-capped), saves the raw results to
data/raw/profile_raw.json and the decisions to data/profile.json:

  R1 product_key     item_id if purchases are matched by id >= 97% of rows and names with
                     several ids cover < 3% of rows; otherwise a normalised item name.
  R2 view_metric     view_events when view_item carries ~1 item (avg <= 1.05) or the extra
                     items are the same product; primary_views when they are other products
                     and the first item matches the page title >= 80% of the time.
  R4 featured_signal "promo" when >= 50% of view_promotion item rows carry a real item id,
                     else "views_share" (spike in a product's share of category views).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga4app import bq, config  # noqa: E402
from ga4app.contracts import JSON_FILES, read_json, utc_now_iso, write_json_atomic  # noqa: E402

PROFILE_QUERIES = (
    "profile_items_per_event",
    "profile_item_fields",
    "profile_categories",
    "profile_keys",
    "profile_purchases",
    "profile_devices",
)
OPTIONAL_QUERIES = ("profile_view_item_composition", "profile_view_offsets")


def decide(results: dict[str, list[dict]]) -> dict:
    """Apply decision rules R1, R2 and R4 to the profiling results."""
    notes: list[str] = []

    # ---- R1: product key
    keys = (results.get("profile_keys") or [{}])[0]
    purchase_rows = keys.get("purchase_rows") or 0
    id_cov = (keys.get("purchase_rows_id_viewed") or 0) / purchase_rows if purchase_rows else 0.0
    name_cov = (keys.get("purchase_rows_name_viewed") or 0) / purchase_rows if purchase_rows else 0.0
    multi_id_share = keys.get("row_share_multi_id_names") or 0.0
    if id_cov >= 0.97 and multi_id_share < 0.03:
        product_key = "item_id"
    else:
        product_key = "name_key"
    notes.append(
        f"R1: purchase rows matched to a viewed product by id {id_cov:.1%}, by name {name_cov:.1%}; "
        f"names with several ids cover {multi_id_share:.1%} of rows -> product_key = {product_key}."
    )

    # ---- R2: view metric
    per_event = {r["event_name"]: r for r in results.get("profile_items_per_event", [])}
    vi = per_event.get("view_item", {})
    avg_items = float(vi.get("avg_items_nonempty") or vi.get("avg_items") or 1.0)
    comp = (results.get("profile_view_item_composition") or [None])[0]
    offs = (results.get("profile_view_offsets") or [None])[0]
    if avg_items <= 1.05:
        view_metric = "view_events"
        notes.append(f"R2: view_item carries {avg_items:.3f} items on average -> view_metric = view_events.")
    elif comp:
        multi = comp.get("multi_item_events") or 0
        same = comp.get("multi_item_same_name") or 0
        first_in_title = comp.get("first_item_in_title") or 0
        events = comp.get("view_events") or 1
        seen_any = (offs or {}).get("seen_in_session_any_offset") or 0
        seen_0 = (offs or {}).get("seen_in_session_at_offset0") or 0
        bought = (offs or {}).get("purchased_items") or 0
        if multi and same / multi >= 0.80:
            view_metric = "view_events"
            notes.append(f"R2: {same / multi:.0%} of multi-item view_item events repeat one product -> view_events.")
        elif first_in_title / events >= 0.80 or (seen_any and seen_0 / seen_any >= 0.80):
            view_metric = "primary_views"
            notes.append("R2: offset 0 identifies the viewed product (title or in-session purchase test) -> primary_views.")
        else:
            view_metric = "view_events"
            if seen_any:
                notes.append(
                    f"R2: view_item events list ~{avg_items:.1f} different products (only {same:,} of {multi:,} multi-item "
                    f"events repeat one product). Of {bought:,} purchased products, {seen_any / bought:.0%} appeared in a "
                    f"view_item list earlier in the same session but only {seen_0 / seen_any:.0%} of those at offset 0, so "
                    "list position carries no 'main product' signal -> view_metric = view_events (a 'view' = the product "
                    "appeared in a view_item event)."
                )
            else:
                notes.append("R2: multi-item view_item events are mixed and the title test is inconclusive -> view_events (caveated).")
    else:
        view_metric = "view_events"
        notes.append(
            f"R2: view_item carries {avg_items:.3f} items on average; composition not profiled "
            "(run with --with-event-params) -> view_events (caveated)."
        )

    # ---- R4: featured signal
    fields = {r["event_name"]: r for r in results.get("profile_item_fields", [])}
    vp = fields.get("view_promotion")
    if vp and vp.get("item_rows"):
        good = 1 - (vp.get("bad_item_id") or 0) / vp["item_rows"]
        featured_signal = "promo" if good >= 0.50 else "views_share"
        notes.append(f"R4: {good:.0%} of view_promotion item rows carry a real item id -> featured_signal = {featured_signal}.")
    else:
        featured_signal = "views_share"
        notes.append("R4: view_promotion events carry no item rows -> featured_signal = views_share.")

    return {
        "product_key": product_key,
        "view_metric": view_metric,
        "featured_signal": featured_signal,
        "stats": {
            "purchase_id_coverage": id_cov,
            "purchase_name_coverage": name_cov,
            "row_share_multi_id_names": multi_id_share,
            "avg_items_per_view_item": avg_items,
        },
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project")
    ap.add_argument("--max-gib", type=float, default=3.0, help="per-query dry-run limit")
    ap.add_argument("--total-gib", type=float, default=25.0, help="total dry-run limit")
    ap.add_argument("--with-event-params", action="store_true", help="also run the event_params composition query")
    ap.add_argument("--yes", action="store_true", help="run even if the estimates exceed the limits")
    ap.add_argument("--dry-run-only", action="store_true")
    args = ap.parse_args(argv)

    project = bq.resolve_project(args.project)
    if not project:
        print("error: no GCP project id; run scripts/bq_auth.py --project <id> first.")
        return 2
    client = bq.get_client(project, bq.get_credentials())
    names = list(PROFILE_QUERIES) + (list(OPTIONAL_QUERIES) if args.with_event_params else [])
    sqls = {n: bq.read_sql(n) for n in names}

    dry = [bq.dry_run(client, sqls[n], n) for n in names]
    bq.print_dry_runs(dry)
    bq.check_budget(dry, max_gib_per_query=args.max_gib, max_gib_total=args.total_gib, assume_yes=args.yes)
    if args.dry_run_only:
        return 0

    data_dir = config.resolve_data_dir()
    log_path = data_dir / JSON_FILES["export_log"]
    results: dict[str, list[dict]] = {}
    bytes_billed: dict[str, int] = {}
    for d in dry:
        cap = bq.effective_cap(d, int(args.max_gib * bq.GIB))
        print(f"running {d.name} (cap {cap / bq.GIB:.2f} GiB)...", flush=True)
        table, meta = bq.run_to_arrow(client, sqls[d.name], cap_bytes=cap, labels={"step": d.name})
        results[d.name] = bq.to_jsonable_rows(table)
        bytes_billed[d.name] = meta["bytes_billed"]
        bq.append_log({"step": d.name, "run_at": utc_now_iso(), "project": project, **meta}, log_path)

    write_json_atomic({"generated_at": utc_now_iso(), "project": project, "results": results}, data_dir / JSON_FILES["profile_raw"])
    decisions = decide(results)
    previous = read_json(data_dir / JSON_FILES["profile"], default={}) or {}
    profile = {
        **previous,
        "generated_at": utc_now_iso(),
        "project": project,
        "queries": {k: len(v) for k, v in results.items()},
        "bytes_billed": bytes_billed,
        "decisions": decisions,
        "items_per_event": results.get("profile_items_per_event", []),
        "item_fields": results.get("profile_item_fields", []),
        "purchases": (results.get("profile_purchases") or [{}])[0],
        "devices": results.get("profile_devices", []),
        "keys": (results.get("profile_keys") or [{}])[0],
        "view_item_composition": (results.get("profile_view_item_composition") or [None])[0],
    }
    write_json_atomic(profile, data_dir / JSON_FILES["profile"])
    print("\nDecisions:")
    for note in decisions["notes"]:
        print("  " + note)
    print(f"\nTotal billed: {sum(bytes_billed.values()) / bq.GIB:.2f} GiB. Wrote {data_dir / JSON_FILES['profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

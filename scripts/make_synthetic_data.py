"""Write a synthetic snapshot set with planted effects (tests and development only).

Usage:
    python scripts/make_synthetic_data.py [--out data_synthetic] [--seed 7] [--n-products 240] [--force]

Point the app at the result with GA4APP_DATA_DIR=data_synthetic; it then shows a SYNTHETIC
banner. Refuses to write into the real data/ directory unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga4app import config  # noqa: E402
from ga4app.synthetic import make_synthetic_snapshots  # noqa: E402


def _inside_real_data(path: Path) -> bool:
    real = config.DEFAULT_DATA_DIR.resolve()
    return path == real or real in path.parents


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_synthetic", help="output directory (default: data_synthetic)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-products", type=int, default=240)
    ap.add_argument("--force", action="store_true", help="allow writing into the real data/ directory")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    if _inside_real_data(out) and not args.force:
        print(f"error: {out} is the real data directory; pick another --out or pass --force")
        return 2
    t0 = time.time()
    try:
        truth = make_synthetic_snapshots(out, n_products=args.n_products, seed=args.seed)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote synthetic snapshots (seed {args.seed}, {args.n_products} products) to {out} in {time.time() - t0:.1f}s")
    print(f"  hidden gems:      {', '.join(truth['hidden_gems'])}")
    print(f"  wasted exposure:  {', '.join(truth['wasted_exposure'])}")
    print(f"  mobile / desktop: {truth['mobile_first']} / {truth['desktop_first']}")
    cp = truth["cannibal_pair"]
    print(f"  cannibal pair:    {cp['a']} -> {cp['b']} ({cp['category_group']})")
    print(f"  trending group:   {truth['trending_group']} (control: {truth['control_group']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

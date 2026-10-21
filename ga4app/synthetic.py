"""Synthetic GA4-shaped snapshots with planted effects, for tests and development only.

``make_synthetic_snapshots`` writes a complete, contract-valid snapshot set (the files the real
pipeline writes to ``data/``) and returns the "truth" dict of the effects it planted, which
``manifest.json`` also records under ``synthetic_truth``. Names, categories, devices, volumes and
seasonality mimic the real store (DESIGN "Real-data findings"). Every count is drawn from
``numpy.random.default_rng(seed)``, so one seed always gives byte-identical parquet files.

Planted effects (see ``make_synthetic_snapshots`` for the truth keys):
- hidden gems: 6 products in the bottom 40% by views, converting at 3x the site base rate;
- wasted exposure: 6 products in the top 15% by views, converting at 0.15x the base rate;
- one Mobile-first and one Desktop-first product: 3x conversion on the favoured device;
- holiday uplift: Apparel converts 1.8x more in Cyber Week, on top of the site seasonality;
- a cannibal pair (A's share of its category's views triples on 8 days, B's conversion falls to
  40% of its peers' on those days) and a null pair (same spikes, no dip);
- a trending group (Google interest +60% on Dec 2-18 vs Nov 1-22, on-site views flat) and a
  control group (flat on both).

The app shows a SYNTHETIC banner whenever ``manifest["source"] == "synthetic"``. Never write
these files into the real ``data/`` directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from . import config, contracts

SOURCE = "synthetic"
MIN_PRODUCTS = 120  # below this the planted roles do not fit in their view-rank bands
TRENDS_START = "2020-10-01"
FETCHED_AT = datetime(2021, 2, 1, tzinfo=timezone.utc)  # fixed so the parquet bytes are reproducible
BUILT_FROM = ["ga4app.synthetic"]

# --------------------------------------------------------------------------- volumes (real-data findings)

BASE_RATE = 0.0055  # pooled purchase / view the calibration aims for
VIEWS_MEDIAN = 2_500.0  # per-product season views: lognormal median ...
VIEWS_P90 = 16_000.0  # ... and 90th percentile
_Z90 = 1.2815515655446004  # standard normal 90th percentile
RATE_SPREAD_M = 1_500.0  # Beta prior strength of the per-product conversion spread
ITEMS_PER_VIEW_EVENT = 11.4
ITEMS_PER_PURCHASE_EVENT = 2.8
ITEMS_PER_CHECKOUT_EVENT = 2.4
DEVICE_VIEW_SHARE = (0.58, 0.40, 0.02)  # config.DEVICES order
DEVICE_RATE = (1.0, 0.95, 0.85)
BRAND_SHARE = {"Google": 0.66, "YouTube": 0.18, "Android": 0.16}
GROUP_PRODUCT_SHARE = {
    "Apparel": 0.40,
    "Drinkware": 0.14,
    "Bags": 0.10,
    "Office & Stationery": 0.13,
    "Electronics & Accessories": 0.07,
    "Lifestyle & Gifts": 0.10,
    "Other": 0.06,
}
# Relative odds of holding the large view quantiles (Apparel dominates the real store's traffic).
GROUP_VIEW_WEIGHT = {
    "Apparel": 1.3,
    "Drinkware": 1.1,
    "Bags": 1.0,
    "Office & Stationery": 1.0,
    "Electronics & Accessories": 0.9,
    "Lifestyle & Gifts": 0.8,
    "Other": 0.7,
}
GROUP_RATE = {
    "Apparel": 1.10,
    "Drinkware": 0.95,
    "Bags": 0.85,
    "Office & Stationery": 1.00,
    "Electronics & Accessories": 0.80,
    "Lifestyle & Gifts": 0.90,
    "Other": 0.80,
}

# --------------------------------------------------------------------------- planted effects

CANNIBAL_GROUP = "Apparel"
NULL_PAIR_GROUP = "Drinkware"
TRENDING_GROUP = "Bags"
CONTROL_GROUP = "Office & Stationery"
HOLIDAY_UPLIFT = {"Apparel": 1.8, "Drinkware": 1.0}  # Cyber Week rate multiplier on top of the season
N_PLANTED = 6
GEM_RATE = 3.0
WASTED_RATE = 0.15
VICTIM_RATE = 2.0  # both pair Bs are high-converting best sellers, so their tests are well powered
PAIR_B_MIN_RANK = {"cannibal": 1, "null": 3}  # B gets at least the season views of this overall rank
DEVICE_FIRST = {"mobile_first": (0.6, 1.8, 0.6), "desktop_first": (1.8, 0.6, 0.6)}  # 3x on the favoured device
DEVICE_FIRST_MIN_VIEWS = 20_000.0  # keeps both device-first products far above 40 purchases
GEM_MIN_VIEWS = 600.0
SPIKE_SHARE = 3.0  # A's share of category views on its spike days, as a multiple of its median share
B_DIP = 0.4
TREND_RISE = 1.6
GIFT_CARD = {"views": 20.0, "purchases": 60.0, "price": 25}  # bought far more often than it is listed

# --------------------------------------------------------------------------- calendar (inclusive ISO windows)

VIEW_SEASON = (
    ("2020-11-01", "2020-11-22", 1.0),
    ("2020-11-23", "2020-12-01", 1.6),
    ("2020-12-02", "2020-12-18", 1.4),
    ("2020-12-19", "2020-12-31", 0.95),
    ("2021-01-01", "2021-01-31", 0.75),
)
VIEW_PEAKS = {"2020-11-27": 1.25, "2020-11-30": 1.25, "2020-12-24": 0.6, "2020-12-25": 0.5}
WEEKEND_VIEWS = 0.82
FLAT_UNTIL = "2020-12-18"  # trending and control groups: flat on-site views from Nov 1 to this day
RATE_SEASON = (
    ("2020-11-01", "2020-11-22", 1.0),
    ("2020-11-23", "2020-12-01", 1.55),
    ("2020-12-02", "2020-12-18", 1.45),
    ("2020-12-19", "2021-01-31", 0.8),
)
UPLIFT_WINDOW = ("2020-11-23", "2020-12-01")  # Cyber Week plus the Cyber Monday tail
SPIKE_STRATA = (("2020-11-02", "2020-11-21", 3), ("2020-12-02", "2020-12-18", 3), ("2021-01-04", "2021-01-29", 2))
CART_GAPS = (("2020-11-01", "2020-11-15"), ("2020-11-20", "2020-11-24"))  # add_to_cart barely tracked (real CSV)
SELECT_ITEM_START = "2020-11-23"
TREND_SEASON = {
    "trending": (
        ("2020-11-23", "2020-12-01", 1.25),
        ("2020-12-02", "2020-12-18", TREND_RISE),
        ("2020-12-19", "2020-12-31", 1.2),
        ("2021-01-01", "2021-01-31", 0.95),
    ),
    "control": (),
    # Interest rises a little less than on-site views, so these groups are never flagged.
    "seasonal": (("2020-11-23", "2020-12-01", 1.35), ("2020-12-02", "2020-12-18", 1.3), ("2021-01-01", "2021-01-31", 0.85)),
}
TREND_KEYWORDS = {
    "Apparel": "hoodie",
    "Drinkware": "water bottle",
    "Bags": "backpack",
    "Office & Stationery": "notebook",
    "Electronics & Accessories": "phone accessories",
    "Lifestyle & Gifts": "gift ideas",
}

# --------------------------------------------------------------------------- site-level ratios (real CSV)

# event -> (events per view_item event, events per session)
SITE_EVENTS = {
    "page_view": (3.50, 3.75),
    "user_engagement": (2.74, 3.0),
    "scroll": (1.28, 2.2),
    "first_visit": (0.67, 1.0),
    "view_promotion": (0.49, 1.6),
    "select_item": (0.080, 1.6),
    "view_search_results": (0.068, 1.9),
    "select_promotion": (0.0245, 1.25),
    "click": (0.0037, 1.15),
}
SESSIONS_PER_VIEW_ITEM = 0.92
EXTRA_SESSIONS = 0.0145  # sessions without a session_start event
CHECKOUT_STEPS = {"add_shipping_info": (0.51, 2.0), "add_payment_info": (0.36, 1.8)}  # per begin_checkout
FUNNEL_EVENTS_PER_SESSION = {"view_item": 5.0, "add_to_cart": 3.85, "begin_checkout": 3.5, "purchase": 1.17}
USERS_PER_SESSION = 0.9
PROMOTIONS = {
    "Reach New Heights": 0.915,
    "Act Responsible": 0.048,
    "Complete Your Collection": 0.023,
    "Google Mural Collection": 0.014,
}
TOP_COUNTRIES = (
    ("United States", 0.4448),
    ("India", 0.0912),
    ("Canada", 0.0755),
    ("United Kingdom", 0.0311),
    ("France", 0.0192),
    ("Spain", 0.0190),
    ("China", 0.0177),
    ("Taiwan", 0.0169),
)
GAP_SHARE = 0.10  # same rule as scripts/build_site_aggregates.py

# --------------------------------------------------------------------------- catalogue

# group -> (product type, price low, price high, category_raw from the real view-row vocabulary)
CATALOG: dict[str, tuple[tuple[str, int, int, str], ...]] = {
    "Apparel": (
        ("Zip Hoodie", 52, 75, "Home/Apparel/Men's / Unisex/"),
        ("Pullover Hoodie", 45, 65, "Home/Apparel/Men's / Unisex/"),
        ("Crewneck Sweatshirt", 40, 55, "Home/Apparel/Men's / Unisex/"),
        ("Tee", 16, 24, "Home/Apparel/Men's / Unisex/"),
        ("Long Sleeve Tee", 22, 30, "Home/Apparel/"),
        ("Polo", 30, 42, "Home/Apparel/Men's / Unisex/"),
        ("Fleece Jacket", 60, 85, "Home/Apparel/"),
        ("Puffer Vest", 55, 70, "Home/Apparel/"),
        ("Windbreaker", 45, 60, "Home/Apparel/"),
        ("Women's Tee", 16, 24, "Home/Apparel/Women's/"),
        ("Women's Fleece Jacket", 60, 80, "Home/Apparel/Women's/"),
        ("Cap", 18, 26, "Home/Apparel/Hats/"),
        ("Beanie", 16, 22, "Home/Apparel/Hats/"),
        ("Socks", 10, 15, "Home/Apparel/Socks/"),
        ("Kids Tee", 14, 20, "Home/Apparel/Kids/"),
        ("Onesie", 18, 24, "Home/Apparel/Kids/"),
    ),
    "Drinkware": (
        ("Water Bottle", 14, 24, "Lifestyle/Drinkware/"),
        ("Nalgene Bottle", 16, 22, "Lifestyle/Drinkware/"),
        ("Glass Bottle", 18, 26, "Water Bottles/"),
        ("Insulated Bottle", 24, 32, "Lifestyle/Drinkware/"),
        ("Camp Mug", 12, 18, "Mugs & Tumblers/"),
        ("Ceramic Mug", 10, 16, "Lifestyle/Drinkware/"),
        ("Thermal Tumbler", 20, 30, "Mugs & Tumblers/"),
        ("Travel Mug", 16, 24, "Lifestyle/Drinkware/"),
        ("Keepcup", 14, 20, "Drinkware"),
    ),
    "Bags": (
        ("Utility Backpack", 60, 95, "Lifestyle/Bags/"),
        ("Commuter Backpack", 55, 85, "Backpacks/"),
        ("Techpack", 70, 95, "Lifestyle/Bags/"),
        ("Tote Bag", 15, 25, "Lifestyle/Bags/"),
        ("Canvas Tote", 14, 22, "Bags"),
        ("Laptop Sleeve", 25, 38, "Lifestyle/Bags/"),
        ("Packable Duffel", 35, 50, "Lifestyle/Bags/"),
        ("Sling Bag", 28, 40, "Lifestyle/Bags/"),
        ("Lunch Bag", 16, 24, "Lifestyle/Bags/"),
        ("Zip Pouch", 10, 16, "Lifestyle/Bags/"),
    ),
    "Office & Stationery": (
        ("Journal", 12, 18, "Home/Stationery/Notebooks/"),
        ("Spiral Notebook", 6, 10, "Home/Stationery/Notebooks/"),
        ("Hardcover Notebook", 10, 16, "Home/Stationery/Notebooks/"),
        ("Sticker Sheet", 3, 6, "Home/Stationery/Stickers/"),
        ("Sticker Pack", 4, 8, "Home/Stationery/Stickers/"),
        ("Ballpoint Pen", 3, 6, "Home/Stationery/Writing/"),
        ("Pen Set", 8, 14, "Home/Stationery/Writing/"),
        ("Task Pad", 5, 9, "Home/Stationery/"),
        ("Highlighter Set", 6, 10, "Home/Stationery/Writing/"),
    ),
    "Electronics & Accessories": (
        ("Phone Stand", 12, 20, "Lifestyle/Small Goods/"),
        ("Phone Grip", 8, 12, "Lifestyle/Small Goods/"),
        ("Cable Organizer", 10, 16, "Lifestyle/Small Goods/"),
        ("Tablet Case", 24, 35, "Small Goods"),
        ("Mouse Pad", 10, 16, "Lifestyle/Small Goods/"),
        ("Webcam Cover", 6, 9, "Lifestyle/Small Goods/"),
        ("Wireless Charger", 25, 35, "Lifestyle/Small Goods/"),
        ("Earbud Case", 12, 18, "Small Goods"),
    ),
    "Lifestyle & Gifts": (
        ("Enamel Pin", 6, 10, "Lifestyle/"),
        ("Magnet Set", 6, 10, "Lifestyle/Small Goods/"),
        ("Fleece Blanket", 30, 45, "Lifestyle/"),
        ("Ornament", 8, 14, "Home/New/"),
        ("Framed Print", 25, 40, "Lifestyle/"),
        ("Keychain", 5, 9, "Lifestyle/Small Goods/"),
        ("Plush Toy", 14, 22, "Lifestyle/"),
        ("Puzzle", 18, 26, "Lifestyle/"),
        ("Figure", 12, 20, "Lifestyle/"),
    ),
    "Other": (
        ("Umbrella", 18, 28, "Home/Sale/"),
        ("Lip Balm", 3, 5, "Home/Eco-Friendly/"),
        ("Sunglasses", 12, 18, "Home/Sale/"),
        ("Face Mask", 8, 12, "Home/Sale/"),
        ("Lanyard", 4, 7, ""),
        ("Pet Bandana", 10, 14, "Home/Eco-Friendly/"),
        ("Seed Kit", 8, 12, ""),
    ),
}
# Collections are not product types: real products often sit under one of these instead.
COLLECTION_PATHS = (
    "Home/Sale/",
    "Home/New/",
    "Home/Shop by Brand/{brand}/",
    "Home/Campus Collection/",
    "Home/Eco-Friendly/",
    "Sale",
    "Campus Collection",
)
COLLECTION_SHARE = 0.14
MODIFIERS = (
    "Iconic", "Classic", "Campus", "Eco", "Cloud", "Logo", "Pride", "Bike", "Explorer", "Summit",
    "Super G", "Lockup", "Stripe", "Heritage", "Retro", "Doodle", "Chrome Dino", "Badge",
)  # fmt: skip
COLOURS = (
    "Black", "Navy", "Grey", "White", "Blue", "Green", "Red", "Charcoal", "Heather Grey",
    "Lavender", "Yellow", "Ocean Blue", "Sage", "Natural", "Maroon", "Teal",
)  # fmt: skip
GROUP_CODE = {
    "Apparel": "HPX",
    "Drinkware": "DRX",
    "Bags": "BGX",
    "Office & Stationery": "OFX",
    "Electronics & Accessories": "ELX",
    "Lifestyle & Gifts": "LFX",
    "Other": "XXX",
}
BRAND_CODE = {"Google": "G", "YouTube": "Y", "Android": "A"}
DECAL_NAME = 'Android Iconic 4" Decal'  # the real store also lists it as `Android Iconic 4&quot; Decal`
GIFT_CARD_NAME = "Google Store Gift Card"


# --------------------------------------------------------------------------- calendar helpers


def _day(iso: str) -> np.datetime64:
    return np.datetime64(iso, "D")


def _mask(days: np.ndarray, start: str, end: str) -> np.ndarray:
    return (days >= _day(start)) & (days <= _day(end))


def _windowed(days: np.ndarray, spec: tuple[tuple[str, str, float], ...]) -> np.ndarray:
    """Per-day multiplier: 1.0 except inside the listed windows."""
    out = np.ones(len(days))
    for start, end, mult in spec:
        out[_mask(days, start, end)] = mult
    return out


def _weekday(days: np.ndarray) -> np.ndarray:
    """Monday = 0 (1970-01-01 was a Thursday)."""
    return (days.astype("int64") + 3) % 7


def _iso(days: np.ndarray) -> list[str]:
    return [str(d) for d in days]


def _display_path(raw: str | None) -> str:
    """``"Home/Apparel/Men's / Unisex/"`` -> ``"Apparel > Men's / Unisex"`` (slashes inside a
    segment are spaced, so only unspaced ones separate segments)."""
    if not raw:
        return ""
    parts: list[str] = []
    for chunk in raw.replace(" / ", "\0").split("/"):
        seg = chunk.replace("\0", " / ").strip()
        if seg and seg not in config.PLACEHOLDERS:
            parts.append(seg)
    if parts and parts[0].lower() == "home":
        parts = parts[1:]
    return " > ".join(parts)


# --------------------------------------------------------------------------- catalogue and roles


def _allocate(n: int, shares: dict[str, float]) -> list[str]:
    """Largest-remainder split of ``n`` products over the groups."""
    raw = {g: n * s / sum(shares.values()) for g, s in shares.items()}
    counts = {g: int(np.floor(v)) for g, v in raw.items()}
    short = n - sum(counts.values())
    for g in sorted(raw, key=lambda g: (-(raw[g] - counts[g]), g))[:short]:
        counts[g] += 1
    return [g for g in shares for _ in range(counts[g])]


def _pair(group: np.ndarray, lam: np.ndarray, grp: str, min_rank: int, taken: set[int]) -> tuple[int, int]:
    """(A, B) for a spike pair; may swap two entries of ``lam`` in place.

    B is the group's most viewed product, first given at least the ``min_rank``-th largest
    season views overall by swapping views with that product (the view distribution is kept;
    products in ``taken`` are never swapped). A is the next product whose season share is
    small enough for its share to triple.
    """
    members = np.flatnonzero(group == grp)
    top = members[int(np.argmax(lam[members]))]
    free = [int(i) for i in np.argsort(-lam, kind="stable") if int(i) not in taken]
    holder = free[min_rank]
    if lam[top] < lam[holder]:
        lam[[top, holder]] = lam[[holder, top]]
    members = members[np.argsort(-lam[members], kind="stable")]
    total = lam[members].sum()
    for a in members[1:8]:
        if lam[a] / total <= 0.2:
            return int(a), int(members[0])
    raise ValueError(f"no spike candidate in {grp!r}; use more products")


def _plant_roles(rng: np.random.Generator, group: np.ndarray, lam: np.ndarray) -> dict[str, Any]:
    """Pick disjoint products for every planted effect, from view-rank bands with safe margins.

    The spike pairs come first because choosing them may swap two products' season views.
    """
    n = len(lam)
    roles: dict[str, Any] = {}
    taken: set[int] = set()
    for role, grp in (("cannibal", CANNIBAL_GROUP), ("null", NULL_PAIR_GROUP)):
        roles[f"{role}_a"], roles[f"{role}_b"] = _pair(group, lam, grp, PAIR_B_MIN_RANK[role], taken)
        taken |= {roles[f"{role}_a"], roles[f"{role}_b"]}
    order = np.argsort(-lam, kind="stable")  # most viewed first
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)

    # Wasted exposure: top 13% by expected views (the contract says top 15%; the margin absorbs noise).
    top = [int(i) for i in order[: int(0.13 * n)] if int(i) not in taken]
    if len(top) < N_PLANTED:
        raise ValueError("not enough high-traffic products for the wasted-exposure set; use more products")
    roles["wasted"] = sorted((int(i) for i in rng.choice(top, size=N_PLANTED, replace=False)), key=lambda i: rank[i])
    taken |= set(roles["wasted"])

    # Device-first: the two most viewed free products below the top 3%, so even the minority
    # device has enough purchases for a stable rate ratio.
    rest = [int(i) for i in order[int(0.03 * n) :] if int(i) not in taken]
    roles["mobile_first"], roles["desktop_first"] = rest[0], rest[1]
    taken |= {rest[0], rest[1]}

    # Hidden gems: 16-34% from the bottom by expected views (the contract says bottom 40%).
    pct_from_bottom = (n - 1 - rank) / n
    pool = [
        int(i)
        for i in order[::-1]
        if 0.16 <= pct_from_bottom[i] < 0.34 and lam[i] >= GEM_MIN_VIEWS and int(i) not in taken
    ]
    trending = [i for i in pool if group[i] == TRENDING_GROUP]
    gems = [int(rng.choice(trending))] if trending else []  # gives the trending alert a product to promote
    others = [i for i in pool if i not in gems]
    if len(others) < N_PLANTED - len(gems):
        raise ValueError("not enough low-traffic products for the hidden-gem set; use more products")
    gems += [int(i) for i in rng.choice(others, size=N_PLANTED - len(gems), replace=False)]
    roles["gems"] = sorted(gems, key=lambda i: rank[i])
    taken |= set(gems)

    lifestyle = [int(i) for i in order[::-1] if group[i] == "Lifestyle & Gifts" and int(i) not in taken]
    roles["gift_card"] = lifestyle[0]
    taken.add(lifestyle[0])
    office = [int(i) for i in order if group[i] == "Office & Stationery" and int(i) not in taken]
    roles["decal"] = int(rng.choice(office))
    return roles


def _catalogue(rng: np.random.Generator, group: np.ndarray, brand: np.ndarray, roles: dict[str, Any]) -> pd.DataFrame:
    """Names, keys, prices, raw categories and item ids, one row per product index."""
    n = len(group)
    rows: list[dict[str, Any]] = []
    keys: set[str] = {DECAL_NAME.lower(), GIFT_CARD_NAME.lower()}
    for i in range(n):
        g, b = str(group[i]), str(brand[i])
        if i == roles["decal"]:
            b, name, ptype, lo, hi, path = "Android", DECAL_NAME, "Decal", 3, 5, "Home/Stationery/Stickers/"
        elif i == roles["gift_card"]:
            b, name, ptype = "Google", GIFT_CARD_NAME, "Gift Card"
            lo = hi = GIFT_CARD["price"]
            path = "Lifestyle/"
        else:
            name = ""
            for _ in range(200):
                ptype, lo, hi, path = CATALOG[g][int(rng.integers(len(CATALOG[g])))]
                mod = MODIFIERS[int(rng.integers(len(MODIFIERS)))] if rng.random() < 0.5 else ""
                if g == "Apparel" and "Hoodie" in ptype and rng.random() < 0.25:
                    tail = "F/C"
                elif g in ("Apparel", "Drinkware", "Bags") or rng.random() < 0.5:
                    tail = COLOURS[int(rng.integers(len(COLOURS)))]
                else:
                    tail = ""
                name = " ".join(p for p in (b, mod, ptype, tail) if p)
                if name.lower() not in keys:
                    break
            else:
                raise ValueError("could not draw a unique product name")
            keys.add(name.lower())
            if rng.random() < COLLECTION_SHARE:
                path = COLLECTION_PATHS[int(rng.integers(len(COLLECTION_PATHS)))].format(brand=b)
        sku = f"GGOE{BRAND_CODE[b]}{GROUP_CODE[g]}{int(rng.integers(100_000, 1_000_000))}"
        reissued = [str(int(rng.integers(9_000_000, 10_000_000))) for _ in range(min(int(rng.poisson(2.2)), 5))]
        rows.append(
            {
                "product_key": name.lower(),
                "item_name": name,
                "item_ids": "|".join([sku, *reissued]),
                "n_item_ids": 1 + len(reissued),
                "n_names": 2 if i == roles["decal"] else 1,
                "category_raw": path or None,
                "category_path": _display_path(path),
                "category_group": g,
                "brand": b,
                "price": float(rng.integers(lo, hi + 1)),
                "product_type": ptype,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- counts


def _profiles(rng: np.random.Generator, days: np.ndarray, group: np.ndarray) -> np.ndarray:
    """(n, D) share of each product's season views falling on each day (rows sum to 1)."""
    weekend = np.where(_weekday(days) >= 5, WEEKEND_VIEWS, 1.0)
    noise = np.exp(rng.normal(0.0, 0.05, len(days)))
    peaks = np.array([VIEW_PEAKS.get(d, 1.0) for d in _iso(days)])
    site = _windowed(days, VIEW_SEASON) * peaks * weekend * noise
    flat = np.where(days <= _day(FLAT_UNTIL), weekend * noise, site)
    is_flat = np.isin(group, (TRENDING_GROUP, CONTROL_GROUP))[:, None]
    prof = np.where(is_flat, flat[None, :], site[None, :]) * np.exp(rng.normal(0.0, 0.10, (len(group), len(days))))
    return prof / prof.sum(axis=1, keepdims=True)


def _spike_days(rng: np.random.Generator, days: np.ndarray) -> np.ndarray:
    """8 day indices outside Cyber Week, spread over the baseline, December and January."""
    idx: list[int] = []
    for start, end, k in SPIKE_STRATA:
        idx += [int(i) for i in rng.choice(np.flatnonzero(_mask(days, start, end)), size=k, replace=False)]
    return np.sort(np.asarray(idx))


def _apply_spike(lam_cell: np.ndarray, in_group: np.ndarray, a: int, spike_idx: np.ndarray) -> None:
    """Scale A's expected views on its spike days so its category share is SPIKE_SHARE x its median."""
    daily = lam_cell.sum(axis=2)
    share = daily[a] / daily[in_group].sum(axis=0)
    target = SPIKE_SHARE * float(np.median(share))
    s = share[spike_idx]
    lam_cell[a, spike_idx, :] *= (target * (1 - s) / (s * (1 - target)))[:, None]


def _rates(
    lam_cell: np.ndarray,
    group: np.ndarray,
    spread: np.ndarray,
    dev_rate: np.ndarray,
    pinned: dict[int, float],
    victim: int,
    dip_days: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """Purchase probability per (product, day, device) that pools to BASE_RATE in expectation.

    Pinned products (gems, wasted, the cannibal victim) convert at exactly ``mult x BASE_RATE``
    over the season, before the victim's dip; all other products are rescaled together so the
    site still pools to BASE_RATE. Seasonality, the holiday uplift and device effects keep their
    relative sizes.
    """
    n = len(group)
    day_rate = np.tile(_windowed(days, RATE_SEASON), (n, 1))
    uplift_days = _mask(days, *UPLIFT_WINDOW)
    for grp, mult in HOLIDAY_UPLIFT.items():
        day_rate[np.ix_(group == grp, uplift_days)] *= mult
    group_rate = np.array([GROUP_RATE[g] for g in group])
    p = (group_rate * spread)[:, None, None] * day_rate[:, :, None] * dev_rate[:, None, :]
    views = lam_cell.sum(axis=(1, 2))
    for i, mult in pinned.items():
        p[i] *= mult * BASE_RATE * views[i] / (lam_cell[i] * p[i]).sum()
    p[victim, dip_days, :] *= B_DIP
    free = np.ones(n, dtype=bool)
    free[list(pinned)] = False
    need = BASE_RATE * lam_cell.sum() - (lam_cell[~free] * p[~free]).sum()
    p[free] *= need / (lam_cell[free] * p[free]).sum()
    return np.minimum(p, 0.5)


def _draw(
    rng: np.random.Generator, lam_cell: np.ndarray, p: np.ndarray, days: np.ndarray, price: np.ndarray
) -> dict[str, np.ndarray]:
    """Every item_daily count, as (n, D, devices) arrays."""
    views = rng.poisson(lam_cell)
    purchases = rng.binomial(views, p)
    cart_track = np.ones(len(days))
    for start, end in CART_GAPS:
        cart_track[_mask(days, start, end)] = 0.004
    promo_views = rng.binomial(views, 0.0063)
    return {
        "views": views,
        "purchases": purchases,
        "view_rows": views + rng.binomial(views, 0.035),
        "primary_views": rng.binomial(views, 1 / 11.0),
        "add_to_carts": rng.binomial(views, 0.25 * cart_track[None, :, None]),
        "checkouts": purchases + rng.binomial(views, 0.024),
        "purchase_events_raw": purchases + rng.binomial(purchases, 0.068),
        "purchases_missing_txn": rng.binomial(purchases, 0.093),
        "units": purchases + rng.binomial(purchases, 0.35) + rng.binomial(purchases, 0.12),
        "promo_views": promo_views,
        "promo_clicks": rng.binomial(promo_views, 0.058),
        "revenue_usd": purchases * price[:, None, None],
    }


def _item_daily(counts: dict[str, np.ndarray], days: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
    """Flatten the count arrays to the item_daily contract, keeping cells with any activity."""
    pi, di, ki = (a.ravel() for a in np.indices(counts["views"].shape))
    keep = (counts["views"] + counts["purchases"]).ravel() > 0
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(days[di[keep]]),
            "device_category": np.asarray(config.DEVICES, dtype=object)[ki[keep]],
            "product_key": keys[pi[keep]],
        }
    )
    frame["views"] = counts["views"].ravel()[keep].astype(np.int64)
    for col, spec in contracts.SCHEMAS["item_daily"].items():
        if col in frame.columns:
            continue
        if col == "view_events":
            frame[col] = frame["views"]
        elif spec == "float":
            frame[col] = np.round(counts[col].ravel()[keep].astype(float), 2)
        else:
            frame[col] = counts[col].ravel()[keep].astype(np.int64)
    frame = frame.loc[:, list(contracts.SCHEMAS["item_daily"])]
    return frame.sort_values(["event_date", "device_category", "product_key"], kind="stable").reset_index(drop=True)


def _sessions(events: np.ndarray, per_session: float, cap: np.ndarray) -> np.ndarray:
    """Sessions with the event: events / per-session rate, at least 1 when there are events."""
    s = np.maximum(np.rint(events / per_session), (events > 0).astype(float))
    return np.minimum(np.minimum(s, events), cap).astype(np.int64)


def _site(
    rng: np.random.Generator, counts: dict[str, np.ndarray], days: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """site_daily, site_funnel and the per-cell view_promotion / select_promotion events."""
    total = {k: counts[k].sum(axis=0).astype(float) for k in ("views", "purchases", "add_to_carts", "checkouts")}
    view_item = np.rint(total["views"] / ITEMS_PER_VIEW_EVENT)
    events: dict[str, np.ndarray] = {
        "view_item": view_item,
        "purchase": np.rint(total["purchases"] / ITEMS_PER_PURCHASE_EVENT),
        "add_to_cart": np.rint(total["add_to_carts"] / ITEMS_PER_VIEW_EVENT),
        "begin_checkout": np.rint(total["checkouts"] / ITEMS_PER_CHECKOUT_EVENT),
        "session_start": rng.poisson(SESSIONS_PER_VIEW_ITEM * view_item).astype(float),
    }
    select_track = np.where(days >= _day(SELECT_ITEM_START), 1.0, 0.004)[:, None]
    for name, (ratio, _) in SITE_EVENTS.items():
        events[name] = rng.poisson(ratio * view_item * (select_track if name == "select_item" else 1.0)).astype(float)
    for name, (ratio, _) in CHECKOUT_STEPS.items():
        events[name] = rng.poisson(ratio * events["begin_checkout"]).astype(float)
    sessions_all = np.rint(events["session_start"] * (1 + EXTRA_SESSIONS))

    per_session = {**{k: v[1] for k, v in SITE_EVENTS.items()}, **{k: v[1] for k, v in CHECKOUT_STEPS.items()}}
    per_session.update(FUNNEL_EVENTS_PER_SESSION)
    per_session["session_start"] = 1.0
    sess: dict[str, np.ndarray] = {}
    frames = []
    for name in sorted(events):
        ev = events[name]
        sess[name] = _sessions(ev, per_session[name], sessions_all)
        if name in ("session_start", "first_visit"):
            share = 0.83 if name == "session_start" else 1.0
        else:
            share = USERS_PER_SESSION
        users = np.minimum(np.maximum(np.rint(sess[name] * share), (sess[name] > 0)), sess[name]).astype(np.int64)
        di, ki = np.nonzero(ev > 0)
        frames.append(
            pd.DataFrame(
                {
                    "event_date": pd.to_datetime(days[di]),
                    "device_category": np.asarray(config.DEVICES, dtype=object)[ki],
                    "event_name": name,
                    "events": ev[di, ki].astype(np.int64),
                    "users": users[di, ki],
                    "sessions": sess[name][di, ki],
                }
            )
        )
    site_daily = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["event_date", "device_category", "event_name"], kind="stable")
        .reset_index(drop=True)
    )
    di, ki = np.indices(view_item.shape)
    di, ki = di.ravel(), ki.ravel()
    purchase_sessions = sess["purchase"].ravel()
    site_funnel = pd.DataFrame(
        {
            "event_date": pd.to_datetime(days[di]),
            "device_category": np.asarray(config.DEVICES, dtype=object)[ki],
            "sessions": sessions_all.ravel().astype(np.int64),
            "view_sessions": sess["view_item"].ravel(),
            "cart_sessions": sess["add_to_cart"].ravel(),
            "checkout_sessions": sess["begin_checkout"].ravel(),
            "purchase_sessions": purchase_sessions,
            "view_purchase_sessions": np.minimum(np.rint(0.97 * purchase_sessions), purchase_sessions).astype(np.int64),
        }
    )
    promo = np.stack([events["view_promotion"], events["select_promotion"]])
    return site_daily, site_funnel, promo


def _promo_daily(rng: np.random.Generator, promo: np.ndarray, days: np.ndarray) -> pd.DataFrame:
    """Banner-level promotion counts: names only, no product (as in the real data)."""
    names = list(PROMOTIONS)
    shares = np.array(list(PROMOTIONS.values()))
    shares = shares / shares.sum()
    views = rng.multinomial(promo[0].astype(np.int64), shares)  # (D, devices, promos)
    clicks = rng.multinomial(promo[1].astype(np.int64), shares)
    di, ki, pi = np.nonzero((views + clicks) > 0)
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(days[di]),
            "device_category": np.asarray(config.DEVICES, dtype=object)[ki],
            "promotion_name": np.asarray(names, dtype=object)[pi],
            "creative_slot": None,
            "product_key": None,
            "promo_views": views[di, ki, pi].astype(np.int64),
            "promo_clicks": clicks[di, ki, pi].astype(np.int64),
        }
    )
    return frame.sort_values(["event_date", "device_category", "promotion_name"], kind="stable").reset_index(drop=True)


def _trends(rng: np.random.Generator) -> pd.DataFrame:
    """Daily Google-Trends-style interest (0-100, max scaled to 100) for one keyword per group."""
    days = np.arange(_day(TRENDS_START), np.datetime64(config.DATA_END, "D") + 1)
    weekend = np.where(_weekday(days) >= 5, 1.04, 1.0)
    frames = []
    for grp, keyword in TREND_KEYWORDS.items():
        kind = "trending" if grp == TRENDING_GROUP else "control" if grp == CONTROL_GROUP else "seasonal"
        raw = _windowed(days, TREND_SEASON[kind]) * weekend * np.exp(rng.normal(0.0, 0.06, len(days)))
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(days),
                    "keyword": keyword,
                    "group_type": "category",
                    "group_value": grp,
                    "geo": "US",
                    "interest": np.clip(np.rint(100 * raw / raw.max()), 0, 100).astype(float),
                    "is_partial": False,
                    "source": SOURCE,
                    "fetched_at": pd.Timestamp(FETCHED_AT),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _keyword_map() -> pd.DataFrame:
    notes = {
        TRENDING_GROUP: (
            f"Synthetic, planted trending group: interest +{TREND_RISE - 1:.0%} on Dec 2-18 vs Nov 1-22 "
            "while on-site views stay flat."
        ),
        CONTROL_GROUP: "Synthetic, planted control group: flat interest and flat on-site views.",
    }
    return pd.DataFrame(
        [
            {
                "group_type": "category",
                "group_value": grp,
                "keyword": kw,
                "notes": notes.get(grp, "Synthetic seasonal series; rises less than on-site views."),
            }
            for grp, kw in TREND_KEYWORDS.items()
        ]
    )


# --------------------------------------------------------------------------- JSON side files


def _tracking_gaps(site_daily: pd.DataFrame, days: np.ndarray) -> dict[str, dict[str, Any]]:
    all_days = pd.to_datetime(days)
    daily = site_daily.groupby(["event_name", "event_date"])["events"].sum()
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(site_daily["event_name"].unique()):
        s = daily.loc[name].reindex(all_days, fill_value=0)
        med = float(s.median())
        gaps = s.index[s.to_numpy() < GAP_SHARE * med] if med > 0 else s.index[:0]
        if len(gaps):
            later = s.index[s.index > gaps.max()]
            first = later.min().date().isoformat() if len(later) else None
        else:
            first = all_days[0].date().isoformat()
        out[str(name)] = {
            "median_daily": med,
            "gap_days": int(len(gaps)),
            "gap_dates": [d.date().isoformat() for d in gaps],
            "first_reliable_date": first,
        }
    return out


def _overview(site_daily: pd.DataFrame, site_funnel: pd.DataFrame, days: np.ndarray, files: dict, now: str) -> dict[str, Any]:
    by_name = site_daily.groupby("event_name")["events"].sum().sort_values(ascending=False, kind="stable")
    by_device = site_daily.groupby("device_category")["events"].sum()
    event_count = int(by_name.sum())
    first_visits = int(site_daily.loc[site_daily["event_name"] == "first_visit", "users"].sum())
    out: dict[str, Any] = {
        "event_count": event_count,
        "user_count": int(round(first_visits * 1.049)),
        "day_count": int(site_daily["event_date"].nunique()),
        "session_count": int(site_funnel["sessions"].sum()),
        "date_min": str(days[0]),
        "date_max": str(days[-1]),
    }
    out["matches_pitch"] = {k: out[k] == v for k, v in config.PROOF_EXPECTED.items()}
    out["events_by_device"] = {d: int(by_device.get(d, 0)) for d in config.DEVICES}
    out["events_by_name"] = {str(k): int(v) for k, v in by_name.items()}
    out["n_countries"] = 109
    out["top_countries"] = [
        {"country": c, "events": int(round(event_count * s)), "users": int(round(out["user_count"] * s))}
        for c, s in TOP_COUNTRIES
    ]
    out["tracking_gaps"] = _tracking_gaps(site_daily, days)
    out["source_csv"] = SOURCE
    out["source"] = SOURCE
    out["files"] = files
    out["generated_at"] = now
    return out


def _summary(item_daily: pd.DataFrame, product_dim: pd.DataFrame) -> dict[str, Any]:
    """Season totals, in the shape of the real manifest's ``summary``."""
    per = item_daily.groupby("product_key")[["views", "purchases"]].sum()
    groups = product_dim.set_index("product_key")["category_group"]
    by_group = per["views"].groupby(groups).sum().reindex(config.CATEGORY_GROUPS, fill_value=0)
    views, purchases = int(per["views"].sum()), int(per["purchases"].sum())
    share = {str(g): float(v / views) if views else 0.0 for g, v in by_group.items()}
    return {
        "products": int(len(product_dim)),
        "products_viewed": int((per["views"] > 0).sum()),
        "products_purchase_only": int(((per["views"] == 0) & (per["purchases"] > 0)).sum()),
        "views": views,
        "purchases": purchases,
        "revenue_usd": round(float(item_daily["revenue_usd"].sum()), 2),
        "pooled_conversion": purchases / views if views else 0.0,
        "view_share_by_group": share,
        "other_share": share["Other"],
        "dates": int(item_daily["event_date"].nunique()),
    }


def _category_colors(view_share: dict[str, float]) -> dict[str, str]:
    """Top 3 groups by season views take the scatter hues in order; every other group is "Other" gray."""
    ranked = sorted(view_share, key=lambda g: (-view_share[g], g))
    colors = {g: config.PALETTE["other"] for g in config.CATEGORY_GROUPS}
    for g, hue in zip(ranked[: len(config.SCATTER_CATEGORICAL)], config.SCATTER_CATEGORICAL):
        colors[g] = hue
    return colors


# --------------------------------------------------------------------------- public API


def make_synthetic_snapshots(out_dir: Path, *, n_products: int = 240, seed: int = 7) -> dict:
    """Write a full synthetic snapshot set into ``out_dir`` and return the planted truth.

    Files: item_daily, product_dim, promo_daily, site_daily, site_funnel and trends_daily
    parquet; overview, profile, manifest, trends_meta and bq_proof json; trends_keywords.csv.

    Truth keys: ``seed``, ``n_products``, ``hidden_gems`` and ``wasted_exposure`` (6 product
    keys each), ``mobile_first``, ``desktop_first``, ``holiday_uplift`` ({group: Cyber Week rate
    multiplier on top of the site seasonality}), ``cannibal_pair`` ({a, b, category_group,
    spike_days, b_dip}), ``null_pair`` ({a, b, category_group, spike_days}), ``trending_group``,
    ``control_group`` and ``trends_keywords_csv``; plus ``base_rate`` (the realised pooled
    purchase / view) and ``purchases_exceed_views`` (product keys bought more often than viewed).
    """
    if n_products < MIN_PRODUCTS:
        raise ValueError(f"n_products must be at least {MIN_PRODUCTS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = np.arange(np.datetime64(config.DATA_START, "D"), np.datetime64(config.DATA_END, "D") + 1)

    # Products: group, brand and expected season views, then the planted roles and names.
    group = rng.permutation(np.asarray(_allocate(n_products, GROUP_PRODUCT_SHARE), dtype=object))
    brand = rng.choice(np.asarray(list(BRAND_SHARE), dtype=object), size=n_products, p=list(BRAND_SHARE.values()))
    # Exact lognormal quantiles (median and p90 hit their targets at any seed), dealt out in a
    # random order that favours the groups with more traffic.
    sigma = float(np.log(VIEWS_P90 / VIEWS_MEDIAN) / _Z90)
    quantiles = np.exp(np.log(VIEWS_MEDIAN) + sigma * ndtri((np.arange(n_products) + 0.5) / n_products))
    score = np.log([GROUP_VIEW_WEIGHT[g] for g in group]) + rng.normal(0.0, 1.0, n_products)
    lam = np.empty(n_products)
    lam[np.argsort(score, kind="stable")] = quantiles
    roles = _plant_roles(rng, group, lam)
    for role in DEVICE_FIRST:
        lam[roles[role]] = max(lam[roles[role]], DEVICE_FIRST_MIN_VIEWS)
    lam[roles["gift_card"]] = GIFT_CARD["views"]
    cat = _catalogue(rng, group, brand, roles)
    keys = cat["product_key"].to_numpy(dtype=object)

    # Expected views per (product, day, device), with the two spike pairs.
    prof = _profiles(rng, days, group)
    dev_share = rng.dirichlet(300 * np.asarray(DEVICE_VIEW_SHARE), size=n_products)
    lam_cell = lam[:, None, None] * prof[:, :, None] * dev_share[:, None, :]
    spikes = {}
    for role, grp in (("cannibal", CANNIBAL_GROUP), ("null", NULL_PAIR_GROUP)):
        spikes[role] = _spike_days(rng, days)
        _apply_spike(lam_cell, group == grp, roles[f"{role}_a"], spikes[role])

    # Conversion: category base x product spread x season x uplift x device, with planted rates.
    spread = rng.beta(RATE_SPREAD_M * BASE_RATE, RATE_SPREAD_M * (1 - BASE_RATE), n_products) / BASE_RATE
    pinned = {i: GEM_RATE for i in roles["gems"]} | {i: WASTED_RATE for i in roles["wasted"]}
    pinned[roles["cannibal_b"]] = pinned[roles["null_b"]] = VICTIM_RATE
    spread[[*pinned, *(roles[k] for k in ("cannibal_a", "null_a", *DEVICE_FIRST))]] = 1.0
    dev_rate = np.tile(np.asarray(DEVICE_RATE), (n_products, 1))
    for role, mult in DEVICE_FIRST.items():
        dev_rate[roles[role]] = mult
    p = _rates(lam_cell, group, spread, dev_rate, pinned, roles["cannibal_b"], spikes["cannibal"], days)

    counts = _draw(rng, lam_cell, p, days, cat["price"].to_numpy(float))
    gc = roles["gift_card"]
    gift_purchases = rng.poisson(GIFT_CARD["purchases"] * prof[gc][:, None] * dev_share[gc][None, :])
    counts["purchases"][gc] = gift_purchases
    counts["purchase_events_raw"][gc] = gift_purchases
    counts["purchases_missing_txn"][gc] = 0
    counts["units"][gc] = gift_purchases
    counts["checkouts"][gc] = gift_purchases
    counts["revenue_usd"][gc] = gift_purchases * float(GIFT_CARD["price"])

    item_daily = _item_daily(counts, days, keys)
    site_daily, site_funnel, promo = _site(rng, counts, days)
    promo_daily = _promo_daily(rng, promo, days)
    trends_daily = _trends(rng)

    totals = item_daily.groupby("product_key").agg(
        total_views=("views", "sum"),
        total_purchases=("purchases", "sum"),
        first_seen=("event_date", "min"),
        last_seen=("event_date", "max"),
    )
    product_dim = cat.drop(columns=["price", "product_type"]).copy()
    product_dim["median_price_usd"] = cat["price"]
    for col in ("total_views", "total_purchases"):
        product_dim[col] = product_dim["product_key"].map(totals[col]).fillna(0).astype(np.int64)
    product_dim.loc[product_dim["total_purchases"] == 0, "median_price_usd"] = np.nan  # prices come from purchase rows
    product_dim["first_seen"] = product_dim["product_key"].map(totals["first_seen"]).fillna(pd.Timestamp(config.DATA_START))
    product_dim["last_seen"] = product_dim["product_key"].map(totals["last_seen"]).fillna(pd.Timestamp(config.DATA_END))
    product_dim = product_dim.loc[:, list(contracts.SCHEMAS["product_dim"])].sort_values("product_key").reset_index(drop=True)

    pooled = float(item_daily["purchases"].sum() / max(int(item_daily["views"].sum()), 1))
    keyword_csv = out_dir / "trends_keywords.csv"
    truth: dict[str, Any] = {
        "seed": int(seed),
        "n_products": int(n_products),
        "hidden_gems": [str(keys[i]) for i in roles["gems"]],
        "wasted_exposure": [str(keys[i]) for i in roles["wasted"]],
        "mobile_first": str(keys[roles["mobile_first"]]),
        "desktop_first": str(keys[roles["desktop_first"]]),
        "holiday_uplift": dict(HOLIDAY_UPLIFT),
        "cannibal_pair": {
            "a": str(keys[roles["cannibal_a"]]),
            "b": str(keys[roles["cannibal_b"]]),
            "category_group": CANNIBAL_GROUP,
            "spike_days": _iso(days[spikes["cannibal"]]),
            "b_dip": B_DIP,
        },
        "null_pair": {
            "a": str(keys[roles["null_a"]]),
            "b": str(keys[roles["null_b"]]),
            "category_group": NULL_PAIR_GROUP,
            "spike_days": _iso(days[spikes["null"]]),
        },
        "trending_group": TRENDING_GROUP,
        "control_group": CONTROL_GROUP,
        "trends_keywords_csv": str(keyword_csv),
        "base_rate": pooled,
        "purchases_exceed_views": [str(keys[gc])],
    }

    _write_all(out_dir, truth, days, item_daily, product_dim, promo_daily, site_daily, site_funnel, trends_daily)
    return truth


def _write_all(
    out_dir: Path,
    truth: dict[str, Any],
    days: np.ndarray,
    item_daily: pd.DataFrame,
    product_dim: pd.DataFrame,
    promo_daily: pd.DataFrame,
    site_daily: pd.DataFrame,
    site_funnel: pd.DataFrame,
    trends_daily: pd.DataFrame,
) -> None:
    now = contracts.utc_now_iso()
    frames = {
        "item_daily": item_daily,
        "product_dim": product_dim,
        "promo_daily": promo_daily,
        "site_daily": site_daily,
        "site_funnel": site_funnel,
        "trends_daily": trends_daily,
    }
    files = {name: {**contracts.write_snapshot(df, name, out_dir), "built_from": BUILT_FROM} for name, df in frames.items()}
    km = _keyword_map()
    km.to_csv(out_dir / "trends_keywords.csv", index=False, lineterminator="\n", encoding="utf-8")

    decisions = {
        "product_key": "name_key",
        "view_metric": "view_events",
        "featured_signal": "views_share",
        "stats": {
            "avg_items_per_view_item": float(
                item_daily["view_rows"].sum()
                / max(int(site_daily.loc[site_daily["event_name"] == "view_item", "events"].sum()), 1)
            ),
        },
        "notes": [
            "SYNTHETIC data generated by ga4app.synthetic for tests and development; not the real store.",
            "product_key = name_key(item_name), as decided for the real data (R1).",
            "view_metric = view_events: a view is an appearance in a view_item event (R2).",
            "featured_signal = views_share: promotions carry no products (R4).",
        ],
    }
    site_by = site_daily.groupby("event_name")["events"].sum()
    purchase_events = int(site_by.get("purchase", 0))
    revenue = float(item_daily["revenue_usd"].sum())
    by_dev = site_daily.pivot_table(index="device_category", columns="event_name", values="events", aggfunc="sum", fill_value=0)
    profile = {
        "generated_at": now,
        "project": SOURCE,
        "source": SOURCE,
        "queries": {},
        "bytes_billed": {},
        "decisions": decisions,
        "purchases": {
            "purchase_events": purchase_events,
            "missing_txn": int(round(purchase_events * 0.159)),
            "distinct_txn": int(round(purchase_events * 0.782)),
            "duplicate_events": int(round(purchase_events * 0.059)),
            "purchase_revenue_usd_raw": round(revenue * 1.067, 2),
            "purchase_revenue_usd_dedup": round(revenue, 2),
        },
        "devices": [
            {
                "device_category": d,
                "events": int(by_dev.loc[d].sum()) if d in by_dev.index else 0,
                "view_item_events": int(by_dev.loc[d].get("view_item", 0)) if d in by_dev.index else 0,
                "purchase_events": int(by_dev.loc[d].get("purchase", 0)) if d in by_dev.index else 0,
            }
            for d in config.DEVICES
        ],
    }
    site_files = {k: {kk: v for kk, v in files[k].items() if kk != "built_from"} for k in ("site_daily", "site_funnel")}
    overview = _overview(site_daily, site_funnel, days, site_files, now)
    proof = {
        "event_count": overview["event_count"],
        "user_count": overview["user_count"],
        "day_count": overview["day_count"],
        "matches_pitch": all(overview["matches_pitch"].values()),
        "expected": dict(config.PROOF_EXPECTED),
        "sql": "-- synthetic snapshot: no BigQuery job was run",
        "project": SOURCE,
        "run_at": now,
        "job_id": None,
        "location": config.BQ_LOCATION,
        "bytes_processed": 0,
        "bytes_billed": 0,
        "cache_hit": False,
        "rows": 1,
        "source": SOURCE,
    }
    keywords = km["keyword"].tolist()
    trends_meta = {
        "keywords": keywords,
        "geo": "US",
        "timeframe": f"{TRENDS_START} {config.DATA_END.isoformat()}",
        "method_used": {kw: SOURCE for kw in keywords},
        "errors": {},
        "status": "ok",
        "source": SOURCE,
        "fetched_at": FETCHED_AT.isoformat(),
    }
    summary = _summary(item_daily, product_dim)
    manifest = {
        "schema_version": contracts.SCHEMA_VERSION,
        "source": SOURCE,
        "generated_at": now,
        "decisions": decisions,
        "product_key": decisions["product_key"],
        "view_metric": decisions["view_metric"],
        "name_aliases": {},
        "category_colors": _category_colors(summary["view_share_by_group"]),
        "files": files,
        "periods": {
            k: {"label": p.label, "start": p.start.isoformat(), "end": p.end.isoformat(), "days": p.days}
            for k, p in config.PERIODS.items()
        },
        "placeholders": list(config.PLACEHOLDERS),
        "devices": {"kept": list(config.DEVICES), "unknown": []},
        "dropped_rows": {"no_product_key": 0, "no_product_key_views": 0, "no_product_key_purchases": 0},
        "summary": summary,
        "seed": truth["seed"],
        "n_products": truth["n_products"],
        "synthetic_truth": truth,
    }
    for name, obj in (("overview", overview), ("profile", profile), ("bq_proof", proof), ("trends_meta", trends_meta)):
        contracts.write_json_atomic(obj, out_dir / contracts.JSON_FILES[name])
    contracts.write_json_atomic(manifest, out_dir / contracts.JSON_FILES["manifest"])  # last: marks the set complete

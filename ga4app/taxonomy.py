"""Product taxonomy: normalised product keys, category paths, category groups and brands.

GA4's ``item_category`` mixes two vocabularies (path-like values on view rows such as
``Home/Apparel/Men's / Unisex/``, flat values on purchase rows such as ``Apparel``) and three
kinds of segment:

* product types (Apparel, Drinkware, Bags, Stationery, ...): the path decides the group;
* departments (Lifestyle, Small Goods, Accessories, Fun, Miscellaneous): too broad to decide on
  their own (``Lifestyle/`` holds bottles, mugs and totes; ``Small Goods`` holds phone stands and
  cable organizers), so the product name decides and "Lifestyle & Gifts" is the fallback;
* collections (Sale, New, Shop by Brand and its brand children, Campus Collection, Eco-Friendly,
  Clearance, Google, the empty value, ...): not product types at all, so the product name decides
  and "Other" is the fallback.

Name rules are keyword patterns matched on the normalised name. When several match, the
right-most wins because the head noun comes last ("Gear Cap Bottle" is a bottle, "Tech Fleece
Vest" is a vest), and a longer phrase beats a shorter one ending at the same place ("Mouse Pad"
beats "Pad", "Pencil Pouch" beats "Pouch"). Every function here is pure.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

from . import config

_WS = re.compile(r"\s+")
_PATH_SPLIT = re.compile(r"(?<! )/(?! )")  # a slash with a space on either side stays: "Men's / Unisex"
_PLACEHOLDERS = frozenset(p.lower() for p in config.PLACEHOLDERS)
OTHER = "Other"

# --------------------------------------------------------------------------- keys


def clean_text(value: object) -> str | None:
    """HTML-unescape, collapse whitespace and trim; ``None`` for missing values and placeholders."""
    if not isinstance(value, str):
        return None
    text = _WS.sub(" ", html.unescape(value)).strip()
    if not text or text.lower() in _PLACEHOLDERS:
        return None
    return text


def name_key(item_name: str | None) -> str | None:
    """Product key: HTML-unescaped, trimmed, whitespace-collapsed and lower-cased item name."""
    text = clean_text(item_name)
    return text.lower() if text else None


# Purchase rows sometimes carry a legacy catalogue name for a product that view rows list under
# its current name. Each pair was checked on the real export: the view-side name has 1k-41k
# views and no purchases, the purchase-side name has purchases and no views, and the legacy
# numeric ids line up. add_to_cart rows use the parent id with the view-side name, purchase rows
# use that parent's size-variant block with the purchase-side name, and the blocks come in the
# same order as the parents (e.g. parent 9195862 -> variants 9195910-14 for the striped L/S).
NAME_ALIASES: dict[str, str] = {
    "womens google striped ls": "google women's striped l/s",
    "google f/c longsleeve charcoal": "google f/c long sleeve tee charcoal",
    "google f/c longsleeve ash": "google f/c long sleeve tee ash",
    "google unisex eco tee black": "google eco tee black",
    "unisex google pocket tee grey": "google pocket tee grey",
    "unisex google jumbo print tee white": "google jumbo print tee white",
    "youth jumbo print tee white": "google youth jumbo print tee white",
    "google summer19 crew grey": "google lightweight crew sweatshirt grey",
}


def canonical_key(item_name: str | None) -> str | None:
    """``name_key`` with legacy purchase-catalogue names folded into their viewed product."""
    key = name_key(item_name)
    return NAME_ALIASES.get(key, key) if key else None


# --------------------------------------------------------------------------- category paths


def parse_category_path(raw: str | None) -> tuple[str, ...]:
    """``"Home/Apparel/Men's / Unisex/"`` -> ``("Apparel", "Men's / Unisex")``."""
    text = clean_text(raw)
    if text is None:
        return ()
    segments = [s.strip() for s in _PATH_SPLIT.split(text)]
    segments = [s for s in segments if s and s.lower() not in _PLACEHOLDERS]
    if segments and segments[0].lower() == "home":
        segments = segments[1:]
    return tuple(segments)


def category_path(raw: str | None) -> str:
    """Display path such as ``"Apparel > Men's / Unisex"``; empty when there is no category."""
    return " > ".join(parse_category_path(raw))


# --------------------------------------------------------------------------- name rules
# Patterns are regex fragments, matched as whole words on the lower-cased name. Group order
# breaks the (rare) tie of two equally long matches ending at the same character.

_NAME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Apparel",
        (
            r"t-?shirts?", r"tees?", r"shirts?", r"tanks?", r"polos?", r"jerseys?",
            r"hood(?:ie|y)s?", r"sweatshirts?", r"sweaters?", r"crew ?necks?", r"crews?", r"pullovers?",
            r"zip-?ups?", r"jackets?", r"fleeces?", r"vests?", r"(?:soft|rain )?shells?", r"softshells?",
            r"raincoats?", r"coats?", r"parkas?", r"windbreakers?", r"anoraks?",
            r"caps?", r"hats?", r"beanies?", r"visors?", r"headbands?",
            r"socks?", r"onesies?", r"bodysuits?", r"bibs?", r"joggers?", r"pants", r"shorts", r"leggings",
            r"l/s", r"ls", r"long ?sleeves?", r"short ?sleeves?", r"scarf", r"scarves", r"gloves?", r"mittens?",
        ),
    ),
    (
        "Drinkware",
        (
            r"bottles?", r"mugs?", r"tumblers?", r"cups?", r"keepcups?", r"straws?", r"(?:mason )?jars?",
            r"flasks?", r"thermos", r"canteens?", r"stojo", r"nalgene", r"drinkware", r"coasters?",
        ),
    ),
    (
        "Bags",
        (
            r"bags?", r"totes?", r"totepaks?", r"backpacks?", r"back packs?", r"packs?", r"techpacks?",
            r"pouch(?:es)?", r"(?:laptop )?sleeves?", r"sacks?", r"duffels?", r"duffles?", r"messengers?",
            r"dopp kits?", r"(?:laptop |travel )?organizers?", r"slings?", r"fanny packs?",
        ),
    ),
    (
        "Office & Stationery",
        (
            r"journals?", r"notebooks?", r"notepads?", r"note ?pads?", r"pads?", r"sticky notes?", r"jotters?",
            r"pens?", r"pencils?", r"pencil (?:pouch(?:es)?|cases?)", r"markers?", r"highlighters?",
            r"stickers?", r"sticker (?:sheets?|packs?)", r"decals?", r"writing sets?", r"stationery", r"planners?",
        ),
    ),
    (
        "Electronics & Accessories",
        (
            r"cables?", r"cable organizers?", r"chargers?", r"charging", r"power ?banks?", r"power",
            r"speakers?", r"phones?", r"phone (?:stands?|holders?|cases?|grips?)", r"earbuds", r"earphones",
            r"headphones", r"usb", r"tech", r"tech tacos?", r"webcams?", r"cameras?", r"mouse ?pads?", r"mouse",
            r"tablets?", r"tablet cases?", r"laptops?", r"lovehandles?", r"see-no hear-no", r"screens?",
            r"electronics?",
        ),
    ),
    (
        "Lifestyle & Gifts",
        (
            r"key ?rings?", r"key ?chains?", r"lanyards?", r"(?:lapel )?pins?", r"magnets?", r"figures?",
            r"figurines?", r"collectibles?", r"plush(?:ies)?", r"toys?", r"games?", r"puzzles?", r"puzzlebots?",
            r"yo-?yos?", r"frisbees?", r"kick ?balls?", r"balls?", r"bots?", r"sunglasses", r"shades",
            r"umbrellas?", r"pets?", r"leash(?:es)?", r"collars?", r"ornaments?", r"sculptures?",
            r"(?:framed )?art", r"posters?", r"blankets?", r"books?", r"gift cards?", r"utensils?",
            r"(?:food )?containers?", r"(?:card |passport )?holders?", r"luggage tags?", r"candles?", r"towels?",
        ),
    ),
)
_NAME_PATTERNS = tuple(
    (rank, group, re.compile(rf"(?<![a-z0-9])(?:{fragment})(?![a-z0-9])"))
    for rank, (group, fragments) in enumerate(_NAME_RULES)
    for fragment in fragments
)
# Multi-pack counts are quantities, not product types: "Pen 4-Pack" is a pen, "Gift Bag 5/PK" a bag.
_QUANTITY = re.compile(r"(?<![a-z0-9])\d+\s*(?:-\s*)?(?:packs?|pk|pcs?|ct)(?![a-z0-9])|\d+\s*/\s*(?:pk|pack)(?![a-z0-9])")


@lru_cache(maxsize=8192)
def _name_group(key: str) -> str | None:
    text = _QUANTITY.sub(" ", key)
    best: tuple[tuple[int, int, int], str] | None = None
    for rank, group, pattern in _NAME_PATTERNS:
        for m in pattern.finditer(text):
            score = (m.end(), m.end() - m.start(), -rank)
            if best is None or score > best[0]:
                best = (score, group)
    return best[1] if best else None


# --------------------------------------------------------------------------- segment kinds

_COLLECTION, _DEPT, _TYPE = 0, 1, 2  # strength order


def _kinds(kind: int, group: str | None, *segments: str) -> dict[str, tuple[int, str | None]]:
    return {s: (kind, group) for s in segments}


_SEGMENTS: dict[str, tuple[int, str | None]] = {
    **_kinds(
        _TYPE, "Apparel",
        "apparel", "hats", "socks", "kids", "kid's", "kids-youth", "kid's-infant", "kid's-toddler",
        "men's / unisex", "men's", "women's", "unisex", "t-shirts", "men's t-shirts", "women's t-shirts",
        "men's warm gear", "women's warm gear",
    ),
    **_kinds(_TYPE, "Drinkware", "drinkware", "mugs & tumblers", "water bottles"),
    **_kinds(_TYPE, "Bags", "bags", "backpacks", "more bags", "shopping and totes"),
    **_kinds(
        _TYPE, "Office & Stationery",
        "stationery", "notebooks", "notebooks & journals", "writing", "writing instruments", "stickers", "office",
    ),
    **_kinds(_TYPE, "Electronics & Accessories", "electronics", "electronic accessories", "electronics accessories"),
    **_kinds(_TYPE, "Lifestyle & Gifts", "gift cards"),
    **_kinds(_DEPT, "Lifestyle & Gifts", "lifestyle", "small goods", "accessories", "fun", "miscellaneous"),
    **_kinds(
        _COLLECTION, None,
        "sale", "new", "clearance", "shop by brand", "campus collection", "eco-friendly", "google", "youtube",
        "android", "google cloud", "#iamremarkable", "black lives matter", "uncategorized items",
    ),
}
_BRAND_ROOT = "shop by brand"  # its children (Google, YouTube, Android, ...) are brand collections


@lru_cache(maxsize=1024)
def _segment_kind(segment: str) -> tuple[int, str | None]:
    norm = _WS.sub(" ", segment.replace("’", "'")).strip().lower()
    if norm in _SEGMENTS:
        return _SEGMENTS[norm]
    if "-" in norm:  # "Clearance-Accessories", "New-Accessories": the strongest part decides
        parts = [_segment_kind(p) for p in norm.split("-") if p.strip()]
        if parts:
            return max(parts, key=lambda kg: kg[0])
    group = _name_group(norm)  # an unseen segment that names a product type ("Drinkware & Gifts")
    return (_TYPE, group) if group else (_COLLECTION, None)


def _path_groups(segments: tuple[str, ...]) -> tuple[str | None, str | None]:
    """(deepest product-type group, deepest department group) along the path."""
    type_group = dept_group = None
    under_brand = False
    for seg in segments:
        if seg.strip().lower() == _BRAND_ROOT:
            under_brand = True
            continue
        kind, group = _segment_kind(seg)
        if kind == _TYPE:
            type_group = group
        elif kind == _DEPT and not under_brand:
            dept_group = group
    return type_group, dept_group


@lru_cache(maxsize=8192)
def _category_group(raw: str | None, key: str | None) -> str:
    type_group, dept_group = _path_groups(parse_category_path(raw))
    if type_group:
        return type_group
    named = _name_group(key) if key else None
    return named or dept_group or OTHER


def category_group(raw_category: str | None, item_name: str | None) -> str:
    """One of ``config.CATEGORY_GROUPS`` for a raw GA4 category and the product's name.

    A product-type segment decides; otherwise the name keywords decide; otherwise a department
    falls back to "Lifestyle & Gifts" and a collection (or no category) to "Other".
    """
    return _category_group(clean_text(raw_category), name_key(item_name))


# --------------------------------------------------------------------------- brands

# The most specific mark wins: a YouTube or Android product is also sold as Google merchandise.
_BRAND_PATTERNS = (
    ("YouTube", re.compile(r"(?<![a-z0-9])youtube(?![a-z0-9])")),
    ("Android", re.compile(r"(?<![a-z0-9])android(?![a-z0-9])")),
    ("Google", re.compile(r"(?<![a-z0-9])(?:google|super g|noogler|chrome)(?![a-z0-9])")),
)


def brand_of(item_name: str | None) -> str:
    """Brand mark named in the product name: "Google", "YouTube", "Android" or "Other".

    Edge cases:
    - Several marks: the most specific sub-brand wins, YouTube then Android then Google
      ("Noogler Android Figure" -> Android).
    - Google marks without the word "Google" still count: "Super G" (the G logo), "Noogler",
      "Chrome" ("Super G Unisex Joggers" -> Google).
    - No mark in the name -> "Other", even though the store sells it: "#IamRemarkable" items
      (their own ``item_brand`` in the export), "Stan and Friends", "Supernatural Paper",
      "Keyboard DOT Sticker", gift cards.
    """
    key = name_key(item_name)
    if key:
        for brand, pattern in _BRAND_PATTERNS:
            if pattern.search(key):
                return brand
    return OTHER

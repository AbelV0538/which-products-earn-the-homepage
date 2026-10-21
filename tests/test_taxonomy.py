"""Taxonomy rules, checked on real product names and category values from the GA4 export."""

from __future__ import annotations

import pytest

from ga4app import config, contracts, taxonomy
from ga4app.taxonomy import brand_of, canonical_key, category_group, category_path, name_key, parse_category_path

# --------------------------------------------------------------------------- name_key


def test_html_entity_names_collapse_to_one_key():
    assert name_key("Android Iconic 4&quot; Decal") == name_key('Android Iconic 4" Decal') == 'android iconic 4" decal'


def test_whitespace_collapses_and_case_folds():
    assert name_key("Google Cloud  Unisex Zip Hoodie") == "google cloud unisex zip hoodie"
    assert name_key("  Google\tTee  Green ") == name_key("GOOGLE TEE GREEN") == "google tee green"


@pytest.mark.parametrize("value", [*config.PLACEHOLDERS, None, "   ", "(NOT SET)", 42])
def test_placeholders_and_empty_values_have_no_key(value):
    assert name_key(value) is None


def test_aliases_fold_legacy_purchase_names_into_the_viewed_product():
    assert canonical_key("Womens Google Striped LS") == name_key("Google Women's Striped L/S")
    assert canonical_key("Google Unisex Eco Tee Black") == "google eco tee black"
    assert canonical_key("Google Tee Green") == name_key("Google Tee Green")
    assert canonical_key("(not set)") is None


def test_alias_table_is_normalised_and_has_no_chains():
    for legacy, current in taxonomy.NAME_ALIASES.items():
        assert name_key(legacy) == legacy
        assert name_key(current) == current
        assert current not in taxonomy.NAME_ALIASES


# --------------------------------------------------------------------------- category paths


def test_parse_keeps_spaced_slash_segment():
    assert parse_category_path("Home/Apparel/Men's / Unisex/") == ("Apparel", "Men's / Unisex")
    assert category_path("Home/Apparel/Men's / Unisex/") == "Apparel > Men's / Unisex"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Lifestyle/Drinkware/", ("Lifestyle", "Drinkware")),
        ("Home/Shop by Brand/Android/", ("Shop by Brand", "Android")),
        ("Men's / Unisex", ("Men's / Unisex",)),
        ("Backpacks/", ("Backpacks",)),
        ("Sale", ("Sale",)),
        ("Home/", ()),
        ("", ()),
        ("(not set)", ()),
        (None, ()),
    ],
)
def test_parse_category_path(raw, expected):
    assert parse_category_path(raw) == expected


# --------------------------------------------------------------------------- category groups


@pytest.mark.parametrize(
    ("raw", "name", "expected"),
    [
        # product-type segments decide
        ("Home/Apparel/Men's / Unisex/", "Google Navy Speckled Tee", "Apparel"),
        ("Lifestyle/Drinkware/", "Google Glass Bottle", "Drinkware"),
        ("Lifestyle/Bags/", "Google Laptop Sleeve", "Bags"),
        ("Home/Stationery/Stickers/", "Google Laptop and Cell Phone Stickers", "Office & Stationery"),
        ("Men's T-Shirts/", "Google Black Tee", "Apparel"),
        ("Mugs & Tumblers/", "Google Felt Mason Jar", "Drinkware"),
        ("Backpacks/", "Google Incognito Zip Pack", "Bags"),
        ("Office", "Google Confetti Pen White", "Office & Stationery"),
        ("Gift Cards", "Gift Card - $25.00", "Lifestyle & Gifts"),
        # collections fall through to the product name
        ("Home/Sale/", "Google Tee Green", "Apparel"),
        ("Home/Shop by Brand/Android/", "Android Small Trace Journal Black", "Office & Stationery"),
        ("Campus Collection", "Google Seattle Campus Mug", "Drinkware"),
        ("Home/New/", "Google Land & Sea Tech Taco", "Electronics & Accessories"),
        ("Home/Sale/", "Android Super Hero 3D Framed Art", "Lifestyle & Gifts"),
        ("Home/Eco-Friendly/", "Google Cork Pencil Pouch", "Office & Stationery"),
        ("Home/Shop by Brand/YouTube/", "YouTube 25 oz Gear Cap Bottle Black", "Drinkware"),
        ("Home/Campus Collection/", "Google NYC Campus Lapel Pin", "Lifestyle & Gifts"),
        ("", "Google 5K Run 2020 Unisex Tee", "Apparel"),
        (None, "Google Clear Pen 4-Pack", "Office & Stationery"),
        # departments: the name decides, "Lifestyle & Gifts" is the fallback
        ("Lifestyle/Small Goods/", "Google Mouse Pad Navy", "Electronics & Accessories"),
        ("Lifestyle/", "Google Thermal Tumbler Navy", "Drinkware"),
        ("Small Goods", "Google Campus Bike", "Lifestyle & Gifts"),
        # nothing matches
        ("Home/Sale/", "Fall Campaign 2020", "Other"),
        (None, None, "Other"),
    ],
)
def test_category_group(raw, name, expected):
    assert category_group(raw, name) == expected


def test_type_segment_beats_name_keywords():
    # Constructed case: a product-type path keeps its type even when the name says otherwise.
    assert category_group("Lifestyle/Drinkware/", "Google Tote") == "Drinkware"
    assert category_group("Home/Sale/", "Google Tote") == "Bags"


def test_groups_are_always_from_config():
    samples = [("Home/Sale/", "Google Tee Green"), ("Lifestyle/", "x"), ("", ""), ("Fun", "Google Bot")]
    assert {category_group(r, n) for r, n in samples} <= set(config.CATEGORY_GROUPS)


# --------------------------------------------------------------------------- brands


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Google Zip Hoodie F/C", "Google"),
        ("YouTube Icon Tee Grey", "YouTube"),
        ("Android Iconic Crewneck Sweatshirt", "Android"),
        ("Noogler Android Figure", "Android"),
        ("Super G Unisex Joggers", "Google"),
        ("Google Chrome Dino Light Up Water Bottle", "Google"),
        ("#IamRemarkable Journal", "Other"),
        ("Stan and Friends Tee Green", "Other"),
        (None, "Other"),
        ("(not set)", "Other"),
    ],
)
def test_brand_of(name, expected):
    assert brand_of(name) == expected
    assert brand_of(name) in config.BRANDS


# --------------------------------------------------------------------------- real data


def test_other_share_on_real_attrs():
    """Every raw (category, name) combination in the real export maps; "Other" stays small."""
    path = config.DEFAULT_DATA_DIR / contracts.PARQUET_FILES["bq_item_attrs"]
    if not path.exists():
        pytest.skip("real BigQuery export not present")
    attrs = contracts.read_snapshot(config.DEFAULT_DATA_DIR, "bq_item_attrs")
    combos = attrs.groupby(["item_category", "item_name"], dropna=False)["view_rows"].sum().reset_index()
    combos["group"] = [
        category_group(None if c is None or c != c else c, None if n is None or n != n else n)
        for c, n in zip(combos["item_category"].astype(object), combos["item_name"].astype(object))
    ]
    share = combos.groupby("group")["view_rows"].sum() / combos["view_rows"].sum()
    other = float(share.get("Other", 0.0))
    assert other <= 0.10, f"'Other' holds {other:.1%} of view rows"

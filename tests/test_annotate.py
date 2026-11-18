"""Chart-note tests: anchors, validation, the number check and the rule-based notes."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ga4app import annotate, config, data, metrics, narrative, phase2
from ga4app.types import ChartNote, Filters, Thresholds

TH = Thresholds(views=1000.0, conversion=0.005, views_rule="median")
F = Filters(start=date(2020, 11, 23), end=date(2020, 11, 30), min_views=200)

# key, name, category, quadrant, code, views, purchases, cr_adj, confidence, lift, forgone, undercount
_ROWS = [
    ("g1", "Google Cloud Sticker", "Office & Stationery", "Hidden gems", "G1", 500, 8, 0.0118, "above", 6.8, 0, False),
    ("g2", "Android Plush", "Lifestyle & Gifts", "Hidden gems", "G2", 450, 5, 0.010, "uncertain", 5.0, 0, False),
    ("w1", "Google Hoodie Navy", "Apparel", "Winners", "W1", 20000, 230, 0.011, "above", 6.0, 0, True),
    ("w2", "Google Tee White", "Apparel", "Winners", "W2", 15000, 110, 0.007, "above", 2.0, 0, False),
    ("x1", "Google Campus Bottle", "Drinkware", "Wasted exposure", "X1", 40000, 20, 0.0006, "below", 0, 176.0, False),
    ("l1", "Google Socks", "Apparel", "Low priority", "", 300, 1, 0.004, "uncertain", 0, 0, False),
    ("n1", "Google Keychain", "Other", "", "", 50, 3, 0.02, "uncertain", 0, 0, False),
]


def make_products() -> pd.DataFrame:
    rows = []
    for key, name, cat, quad, code, v, p, cr_adj, conf, lift, forgone, under in _ROWS:
        rows.append({
            "product_key": key, "item_name": name, "category_group": cat, "brand": "Google",
            "views": v, "purchases": p, "units": p, "revenue_usd": 10.0 * p, "promo_views": 0,
            "cr_raw": p / v, "prior_mean": 0.005, "cr_adj": cr_adj, "ci_low": cr_adj * 0.6, "ci_high": cr_adj * 1.4,
            "confidence": conf, "eligible": v >= 200, "quadrant": quad, "forgone_purchases": float(forgone),
            "lift_per_1k": float(lift), "rpv": 10.0 * p / v, "action_rank": np.nan, "code": code,
            "views_share": v / 76300, "purchases_exceed_views": False, "checkouts": 0, "views_undercount": under,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def products() -> pd.DataFrame:
    return make_products()


@pytest.fixture
def scatter(products) -> annotate.AnchorSet:
    payload = narrative.build_payload(products, TH, F, window_label="Cyber Week")
    return annotate.scatter_anchors(payload, products, {"x1", "zz"})


def _note(anchor: str, detail: str = "Worth a slot.", *, label: str = "Look here", kind: str = "highlight") -> dict:
    return {"anchor": anchor, "kind": kind, "label": label, "detail": detail}


def _reply(*notes: dict) -> str:
    return json.dumps({"notes": list(notes)})


# --------------------------------------------------------------------------- snapshot helpers


def _snapshot_scatter(data_dir: Path, f: Filters = Filters(), y: str = "cr_raw") -> tuple[annotate.AnchorSet, pd.DataFrame]:
    snap = data.load_snapshot(data_dir)
    m = metrics.season_prior_strength(snap.item_daily, snap.product_dim)
    products, th = metrics.compute_products(snap.item_daily, snap.product_dim, f, m, y=y)
    payload = narrative.build_payload(products, th, f, window_label="Full season")
    spikes = phase2.exposure_spikes(snap.item_daily, snap.product_dim, Filters(min_views=0))
    tests = phase2.cannibalization_tests(snap.item_daily, snap.product_dim, spikes, Filters(min_views=0))
    keys = {k for pair in phase2.conflict_pairs(tests) for k in pair}
    return annotate.scatter_anchors(payload, products, keys), products


# --------------------------------------------------------------------------- anchors


def test_scatter_anchors_are_the_coded_eligible_products(scatter, products):
    assert scatter.chart == annotate.CHART_SCATTER
    assert scatter.ids() == ["W1", "W2", "G1", "G2", "X1"]  # the payload's sections, in order
    g1 = scatter.by_id["G1"]
    assert g1["name"] == "Google Cloud Sticker" and g1["quadrant"] == "Hidden gems"
    assert g1["views"] == 500 and g1["conversion_pct"] == 1.6 and g1["adjusted_conversion_pct"] == 1.18
    assert g1["vs_avg"] == 2.36 and g1["lift_per_1k"] == 6.8 and g1["interval_pct"] == [0.71, 1.65]
    assert "revenue_usd" not in g1 and "product_key" not in g1
    assert scatter.by_id["W1"]["views_undercount"] is True
    assert scatter.by_id["X1"]["cannibal_pair"] is True and "cannibal_pair" not in g1
    assert scatter.by_id["X1"]["forgone_purchases"] == 176.0
    assert scatter.scope == {"window": "Cyber Week", "filters": F.label(), "avg_conversion_pct": 0.5,
                             "views_threshold": 1000}
    assert scatter.keys == {"W1": "w1", "W2": "w2", "G1": "g1", "G2": "g2", "X1": "x1"}


def test_scatter_anchors_skip_uncoded_and_ineligible_products(products):
    payload = narrative.build_payload(products, TH, F, window_label="Cyber Week")
    products.loc[products["code"] == "W2", "eligible"] = False
    anchors = annotate.scatter_anchors(payload, products)
    assert "W2" not in anchors.ids() and not {"l1", "n1"} & set(anchors.ids())
    assert set(anchors.ids()) <= narrative.payload_codes(payload)


def test_scatter_anchors_on_the_synthetic_snapshot(synthetic_dir):
    anchors, products = _snapshot_scatter(synthetic_dir)
    coded = products[(products["code"] != "") & products["eligible"].astype(bool)]
    assert 0 < len(anchors.anchors) <= annotate.SCATTER_ANCHORS_MAX
    assert set(anchors.ids()) == set(coded["code"])
    assert all(a["quadrant"] in ("Winners", "Hidden gems", "Wasted exposure") for a in anchors.anchors)


# --------------------------------------------------------------------------- validation


def test_unknown_anchor_is_dropped(scatter):
    result = annotate.validate_notes(_reply(_note("G1"), _note("G9")), scatter)
    assert [n.anchor for n in result.notes] == ["G1"] and result.source == "ai"
    assert result.dropped == [("G9", "unknown anchor")]


def test_duplicate_anchors_keep_the_first(scatter):
    result = annotate.validate_notes(_reply(_note("G1", "First."), _note("G1", "Second.")), scatter)
    assert [n.detail for n in result.notes] == ["First."]
    assert result.dropped == [("G1", "duplicate anchor")]


def test_more_than_four_notes_are_trimmed(scatter):
    result = annotate.validate_notes(_reply(*[_note(a) for a in scatter.ids()]), scatter)
    assert len(result.notes) == config.NOTES_MAX == 4
    assert [n.n for n in result.notes] == [1, 2, 3, 4]
    assert result.dropped == [("X1", "more than 4 notes")]


def test_unknown_kind_becomes_highlight_and_bad_shapes_are_dropped(scatter):
    reply = _reply(_note("G1", kind="warning"), _note("X1", kind="Caution"), {"anchor": "W1", "label": 3, "detail": "x"},
                   "just text")
    result = annotate.validate_notes(reply, scatter)
    assert [(n.anchor, n.kind) for n in result.notes] == [("G1", "highlight"), ("X1", "caution")]
    assert [a for a, _ in result.dropped] == ["W1", ""]


@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```",
    "Here are the notes:\n{}\nHope this helps!",
    "```\n{}\n``` and some trailing prose {{with braces}}",
])
def test_fenced_or_prose_wrapped_json_is_parsed(scatter, wrap):
    result = annotate.validate_notes(wrap.replace("{}", _reply(_note("G1"))), scatter)
    assert result.source == "ai" and [n.anchor for n in result.notes] == ["G1"] and result.error is None


def test_a_bare_list_is_parsed(scatter):
    result = annotate.validate_notes(json.dumps([_note("W1"), _note("G1")]), scatter)
    assert [n.anchor for n in result.notes] == ["W1", "G1"]


@pytest.mark.parametrize("reply", ['{"notes": [{"anchor": "G1", ', "I can't help with that.", "", '{"answer": 1}'])
def test_unreadable_reply_gives_the_rule_based_notes(scatter, reply):
    result = annotate.validate_notes(reply, scatter, model="m")
    assert result.source == "rule" and result.error == "AI notes could not be read" and result.model == "m"
    assert result.notes == annotate.rule_based_notes(scatter).notes


def test_no_surviving_note_gives_the_rule_based_notes_with_reasons(scatter):
    result = annotate.validate_notes(_reply(_note("G9"), _note("G1", "Converts at 9.99%.")), scatter)
    assert result.source == "rule" and result.error == "no AI note passed the checks"
    assert result.dropped == [("G9", "unknown anchor"), ("G1", "9.99 is not a number from its data")]
    assert result.notes and result.notes == annotate.rule_based_notes(scatter).notes
    empty = annotate.validate_notes('{"notes": []}', scatter)
    assert empty.source == "rule" and empty.error == "the model returned no notes"


def test_long_text_is_shortened_at_a_word(scatter):
    label = "A label that is far too long to fit next to a bubble on the chart"
    detail = "word " * 60
    (note,) = annotate.validate_notes(_reply(_note("G1", detail, label=label)), scatter).notes
    assert len(note.label) <= config.NOTE_LABEL_MAX and note.label.endswith("...")
    assert len(note.detail) <= config.NOTE_DETAIL_MAX and note.detail.endswith("word...")


def test_text_is_stored_unescaped(scatter):
    (note,) = annotate.validate_notes(_reply(_note("G1", "<b>Bold</b> & $5 claims", label="A & B")), scatter).notes
    assert note.detail == "<b>Bold</b> & $5 claims" and note.label == "A & B"


# --------------------------------------------------------------------------- number check


@pytest.mark.parametrize("text, ok", [
    ("Converts at 2.4× the average.", True),  # vs_avg 2.36 rounds to 2.4
    ("Converts at 2.36× the average.", True),
    ("Converts at 2.6× the average.", False),
    ("Converts at 2× the average.", True),  # a rounding to 0 decimals
    ("1.1 points above the average.", False),  # a computed difference (1.6 - 0.5)
    ("One of the top 3 gems.", True),  # counting word
    ("Ranked 7th.", True),  # lift_per_1k 6.8 rounds to 7
    ("Ranked 9th.", False),
    ("Google Cloud Sticker [G1] stands out.", True),  # the code is not a number
    ("G1 converts at 1.6% on 500 views.", True),
    ("Average is 0.5%, cutoff 1,000 views.", True),  # scope numbers
    ("In Cyber Week, min 200 views.", True),  # numbers inside the filters string
    ("Interval 0.71% to 1.65%.", True),
    ("Adds 6.8 purchases per 1,000 extra views.", True),  # 1,000 is the scope's views threshold
    ("Adds 68 purchases per 10,000 extra views.", False),  # a computed scaling
])
def test_number_check(scatter, text, ok):
    assert (annotate.unmatched_numbers(text, scatter.by_id["G1"], scatter.scope) == []) is ok


def test_number_check_commas_and_signs():
    facts = {"id": "Apparel·cyber_week", "views": 1234, "lift_pct": -23.4, "name": "Apparel"}
    assert annotate.unmatched_numbers("1,234 views", facts) == []
    assert annotate.unmatched_numbers("Conversion drops 23% in the window.", facts) == []
    assert annotate.unmatched_numbers("Conversion drops 23.4%.", facts) == []
    assert annotate.unmatched_numbers("Conversion drops 24%.", facts) == ["24"]
    assert annotate.unmatched_numbers("1,235 views", facts) == ["1,235"]


def test_number_check_ignores_the_anchor_id():
    facts = {"id": "R12·mobile", "views": 40}
    assert annotate.unmatched_numbers("Row R12·mobile has 40 views.", facts) == []


# --------------------------------------------------------------------------- rule-based notes


def test_rule_based_scatter_notes(scatter):
    result = annotate.rule_based_notes(scatter)
    assert result.source == "rule" and result.dropped == [] and result.error is None
    notes = {n.anchor: n for n in result.notes}
    assert [n.anchor for n in result.notes] == ["G1", "W1", "X1"]
    assert notes["G1"].label == "Best hidden gem" and notes["G1"].kind == "highlight"
    assert notes["G1"].detail == "Google Cloud Sticker [G1] converts at 2.36× the average on 500 views."
    assert notes["X1"].kind == "caution" and "176 purchases forgone" in notes["X1"].detail
    # W1 is the top featured pick and its views look under-counted: its note becomes the caution.
    assert notes["W1"].kind == "caution" and notes["W1"].label == "Views may be under-counted"
    assert [n.n for n in result.notes] == [1, 2, 3]
    assert result == annotate.rule_based_notes(scatter)  # deterministic


def test_rule_based_undercount_note_for_another_pick(products):
    products["views_undercount"] = products["code"] == "W2"
    products.loc[products["code"] == "W1", "confidence"] = "uncertain"
    products.loc[products["code"] == "G2", "confidence"] = "below"
    payload = narrative.build_payload(products, TH, F, window_label="Cyber Week")
    notes = annotate.rule_based_notes(annotate.scatter_anchors(payload, products)).notes
    assert [(n.anchor, n.kind) for n in notes] == [("G1", "highlight"), ("W1", "highlight"), ("X1", "caution"),
                                                   ("W2", "caution")]


def test_rule_based_notes_pass_validation_on_snapshots(synthetic_dir):
    anchors, _ = _snapshot_scatter(synthetic_dir)
    result = annotate.rule_based_notes(anchors)
    assert result.notes and result.dropped == []
    drafts = [{"anchor": n.anchor, "kind": n.kind, "label": n.label, "detail": n.detail} for n in result.notes]
    assert annotate.validate_notes(json.dumps({"notes": drafts}), anchors).notes == result.notes


def test_rule_based_notes_pass_validation_on_the_real_snapshot(real_data_dir):
    anchors, _ = _snapshot_scatter(real_data_dir)
    result = annotate.rule_based_notes(anchors)
    assert result.notes and result.dropped == []


def test_notes_for_an_empty_chart():
    empty = annotate.AnchorSet(annotate.CHART_SCATTER, {})
    assert annotate.rule_based_notes(empty).notes == []
    assert annotate.validate_notes([ChartNote("G1", "highlight", "a", "b")], empty, source="rule").dropped == [
        ("G1", "unknown anchor")]

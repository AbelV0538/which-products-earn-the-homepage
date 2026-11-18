"""Chart-note tests: anchors, validation, the number check and the rule-based notes."""

from __future__ import annotations

import inspect
import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from ga4app import annotate, config, data, llm, metrics, narrative, phase2
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
                             "views_threshold": 1000, "interval_level_pct": 80}
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


def test_number_check_reads_per_1000_as_a_unit():
    facts = {"id": "G1", "lift_per_1k": 57.2}  # seen live: "57.2 extra purchases per 1,000 views"
    assert annotate.unmatched_numbers("Adds 57.2 extra purchases per 1,000 views.", facts) == []
    assert annotate.unmatched_numbers("Adds 57.2 extra purchases per 1000 extra views.", facts) == []
    assert annotate.unmatched_numbers("Drew 1,000 views.", facts) == ["1,000"]


def test_anchor_ids_typed_back_with_other_spaces_still_match():
    anchors = annotate.holiday_anchors(make_groups())
    reply = _reply(_note(" Bags·cyber_week ", "Bags drops 23%."), _note("Apparel\u00b7cyber_week", "Up 55%."),
                   _note("Drinkware ·cyber_week"))  # a space the ID does not have: still a different anchor
    result = annotate.validate_notes(reply, anchors)
    assert [n.anchor for n in result.notes] == ["Bags·cyber_week", "Apparel·cyber_week"]
    assert result.dropped == [("Drinkware ·cyber_week", "unknown anchor")]
    groups = make_groups()
    groups["group"] = groups["group"].replace({"Bags": "Lifestyle & Gifts"})
    gifts = annotate.holiday_anchors(groups)
    nbsp = annotate.validate_notes(_reply(_note("Lifestyle &\u00a0Gifts·cyber_week", "Down 23%.")), gifts)
    assert [n.anchor for n in nbsp.notes] == ["Lifestyle & Gifts·cyber_week"]  # seen in the first live run


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


# --------------------------------------------------------------------------- the model call


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.delenv(config.LLM_FAKE_ENV, raising=False)
    monkeypatch.setattr("ga4app.llm._sleep", lambda s: None)


KEY = "sk-or-v1-testkey0123456789abcdef"


def _status(code: int, message: str = "nope") -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(code, json={"error": {"code": code, "message": message}}))


def test_build_messages_sends_the_anchors_and_nothing_else(scatter):
    system, user = annotate.build_messages(scatter)
    assert system["role"] == "system" and "at most 4 anchors" in system["content"]
    assert user["content"].startswith("Annotate this chart. JSON:\n")
    sent = json.loads(user["content"].split("\n", 1)[1])
    assert sent == {"chart": annotate.CHART_SCATTER, "question": scatter.question, "scope": scatter.scope,
                    "anchors": scatter.anchors}
    assert "w1" not in user["content"]  # product keys stay in the app


def test_cache_key_follows_the_anchors_model_and_prompt(scatter, monkeypatch):
    key = annotate.cache_key(scatter, "m")
    assert key == annotate.cache_key(annotate.AnchorSet(scatter.chart, dict(scatter.scope), list(scatter.anchors)), "m")
    assert key != annotate.cache_key(scatter, "other")
    assert key != annotate.cache_key(annotate.AnchorSet(scatter.chart, scatter.scope, scatter.anchors[:2]), "m")
    monkeypatch.setattr(annotate, "NOTES_PROMPT_VERSION", "notes-test")
    assert key != annotate.cache_key(scatter, "m")


def test_cache_key_ignores_display_controls(synthetic_dir):
    """Raw/Adjusted and Log/Linear do not change the anchors, so the notes are reused. (Raw/Adjusted
    re-classifies the quadrants; the key would change only if that re-coded the products.)"""
    raw, _ = _snapshot_scatter(synthetic_dir, y="cr_raw")
    adjusted, _ = _snapshot_scatter(synthetic_dir, y="cr_adj")
    assert annotate.cache_key(raw, "m") == annotate.cache_key(adjusted, "m")
    # Log/Linear, bubble size and colour are arguments of the chart only, never of the anchors.
    assert set(inspect.signature(annotate.scatter_anchors).parameters) == {"payload", "products", "conflict_keys"}


def test_ai_notes_with_the_fake_responder(scatter):
    result = annotate.ai_notes(scatter, api_key=KEY, model=config.DEFAULT_MODEL,
                               transport=llm.fake_transport(respond=annotate.fake_respond))
    assert result.source == "ai" and result.error is None and result.dropped == []
    assert [(n.anchor, n.kind, n.n) for n in result.notes] == [("W1", "highlight", 1), ("W2", "caution", 2)]
    assert result.model == llm.FAKE_MODEL and result.usage == llm.FAKE_USAGE


def test_ai_notes_uses_the_fake_responder_in_fake_mode(scatter, monkeypatch):
    monkeypatch.setenv(config.LLM_FAKE_ENV, "1")
    result = annotate.ai_notes(scatter, api_key=KEY, model=config.DEFAULT_MODEL)
    assert result.source == "ai" and [n.anchor for n in result.notes] == scatter.ids()[:2]


def test_ai_notes_drops_invented_numbers(scatter):
    reply = _reply(_note("G1", "Google Cloud Sticker [G1] converts at 2.4× the average."),
                   _note("X1", "Retiring it would add 42 purchases.", kind="caution"))
    result = annotate.ai_notes(scatter, api_key=KEY, model="m", transport=llm.fake_transport(lambda body: [reply]))
    assert [n.anchor for n in result.notes] == ["G1"] and result.source == "ai"
    assert result.dropped == [("X1", "42 is not a number from its data")]


@pytest.mark.parametrize("code, reason", [(401, "invalid API key, HTTP 401"), (402, "out of OpenRouter credits, HTTP 402")])
def test_ai_notes_openrouter_error_gives_the_rule_based_notes(scatter, code, reason):
    result = annotate.ai_notes(scatter, api_key=KEY, model="m", transport=_status(code))
    assert result.source == "rule" and result.error == reason
    assert result.notes == annotate.rule_based_notes(scatter).notes


def test_ai_notes_empty_reply_or_no_key_gives_the_rule_based_notes(scatter):
    empty = annotate.ai_notes(scatter, api_key=KEY, model="m", transport=llm.fake_transport(lambda body: [" "]))
    assert empty.source == "rule" and "returned no text" in empty.error
    assert empty.notes == annotate.rule_based_notes(scatter).notes
    no_key = annotate.ai_notes(scatter, api_key=None, model="m", transport=llm.fake_transport(annotate.fake_respond))
    assert no_key.source == "rule" and no_key.error == "no OpenRouter API key"
    no_model = annotate.ai_notes(scatter, api_key=KEY, model=" ")
    assert no_model.source == "rule" and no_model.error == "no model selected"


def test_ai_notes_without_anchors_makes_no_call():
    calls = []
    result = annotate.ai_notes(annotate.AnchorSet(annotate.CHART_SCATTER, {}), api_key=KEY, model="m",
                               transport=llm.fake_transport(lambda body: calls.append(body) or []))
    assert calls == [] and result.notes == [] and result.error is None


# --------------------------------------------------------------------------- Holiday Shift


def make_groups() -> pd.DataFrame:
    """A ``period_comparison``-shaped frame: Apparel rises, Bags drops, Drinkware is flat, Office is thin."""
    spec = {  # group -> (base cr, focus cr, p_value, low_sample)
        "Apparel": (0.004, 0.0062, 0.0004, False),
        "Drinkware": (0.005, 0.0052, 0.40, False),
        "Bags": (0.006, 0.0046, 0.012, False),
        "Office & Stationery": (0.003, 0.009, np.nan, True),
    }
    rows = []
    for g, (cr_b, cr_f, p, low) in spec.items():
        for key, cr, is_base in (("baseline", cr_b, True), ("cyber_week", cr_f, False)):
            rows.append({
                "group": g, "period": key, "period_label": config.PERIODS[key].label, "views": 50_000,
                "purchases": int(50_000 * cr), "cr": cr, "ci_low": cr * 0.9, "ci_high": cr * 1.1,
                "lift": np.nan if is_base else cr_f / cr_b - 1, "p_value": np.nan if is_base else p,
                "low_sample": low, "lift_vs_rest": np.nan if is_base else 0.1234, "p_value_vs_rest": np.nan if is_base else 0.2,
                "is_base": is_base,
            })
    return pd.DataFrame(rows)


def _snapshot_bars(data_dir: Path, base: str = "baseline", focus: str = "cyber_week") -> annotate.AnchorSet:
    snap = data.load_snapshot(data_dir)
    groups = metrics.period_comparison(snap.item_daily, snap.product_dim, [config.PERIODS[base], config.PERIODS[focus]],
                                       Filters(), by="category_group")
    return annotate.holiday_anchors(groups, filters="all devices · all categories · min 200 views")


def test_holiday_anchors_are_the_tested_focus_bars():
    anchors = annotate.holiday_anchors(make_groups(), filters="all devices")
    assert anchors.chart == annotate.CHART_BARS
    assert anchors.ids() == ["Apparel·cyber_week", "Drinkware·cyber_week", "Bags·cyber_week"]  # no base, no low sample
    a = anchors.by_id["Apparel·cyber_week"]
    assert a == {"id": "Apparel·cyber_week", "category": "Apparel", "window": "Cyber Week", "conversion_pct": 0.62,
                 "base_conversion_pct": 0.4, "interval_pct": [0.56, 0.68], "lift_pct": 55.0, "p_value": 0.0004,
                 "significant": True, "lift_vs_rest_pct": 12.3, "p_value_vs_rest": 0.2}
    assert anchors.by_id["Drinkware·cyber_week"]["significant"] is False
    assert anchors.scope == {"base": "Pre-holiday baseline", "focus": "Cyber Week", "base_dates": "Nov 1 – Nov 22",
                             "focus_dates": "Nov 23 – Nov 30", "filters": "all devices", "significance_level": 0.05,
                             "interval_level_pct": 95}
    assert annotate.holiday_anchors(None).anchors == [] and annotate.holiday_anchors(pd.DataFrame()).anchors == []


def test_rule_based_bar_notes_point_at_the_biggest_rise_and_drop():
    anchors = annotate.holiday_anchors(make_groups())
    result = annotate.rule_based_notes(anchors)
    assert result.dropped == []
    assert [(n.anchor, n.kind, n.label) for n in result.notes] == [
        ("Apparel·cyber_week", "highlight", "Biggest rise"), ("Bags·cyber_week", "caution", "Biggest drop")]
    assert result.notes[0].detail == ("Apparel converts 55% better during Cyber Week (0.4% to 0.62%), "
                                      "significant at p < 0.05.")
    assert result.notes[1].detail.startswith("Bags converts 23% worse during Cyber Week")


def test_no_significant_shift_means_no_bar_notes():
    groups = make_groups().assign(p_value=lambda d: np.where(d["is_base"], np.nan, 0.5))
    anchors = annotate.holiday_anchors(groups)
    assert anchors.anchors and annotate.rule_based_notes(anchors).notes == []
    assert anchors.empty == "No significant shift to point out."


def test_bar_notes_number_check_reads_the_lift_label():
    anchors = annotate.holiday_anchors(make_groups())
    bags = anchors.by_id["Bags·cyber_week"]  # lift -23.3%
    assert annotate.unmatched_numbers("Bags drops 23% in Cyber Week.", bags, anchors.scope) == []
    assert annotate.unmatched_numbers("Bags drops 1.4 points.", bags, anchors.scope) == ["1.4"]
    # Seen live: "with a 95% range of 0.22% to 0.36%" names the interval, it does not invent a number.
    assert annotate.unmatched_numbers("Its 95% range is 0.41% to 0.51%.", bags, anchors.scope) == []


@pytest.mark.parametrize("base, focus", config.COMPARISONS)
def test_rule_based_bar_notes_pass_on_the_synthetic_snapshot(synthetic_dir, base, focus):
    anchors = _snapshot_bars(synthetic_dir, base, focus)
    result = annotate.rule_based_notes(anchors)
    assert result.dropped == [] and all(a["id"].endswith(focus) for a in anchors.anchors)


def test_synthetic_holiday_notes_find_the_planted_apparel_uplift(synthetic_dir, synthetic_truth):
    assert synthetic_truth["holiday_uplift"]["Apparel"] > 1
    notes = annotate.rule_based_notes(_snapshot_bars(synthetic_dir)).notes
    assert notes[0].anchor == "Apparel·cyber_week" and notes[0].kind == "highlight"


@pytest.mark.parametrize("base, focus", config.COMPARISONS)
def test_rule_based_bar_notes_pass_on_the_real_snapshot(real_data_dir, base, focus):
    result = annotate.rule_based_notes(_snapshot_bars(real_data_dir, base, focus))
    assert result.notes and result.dropped == []


# --------------------------------------------------------------------------- Device Split


def make_matrix(spec: dict[str, tuple[float, float, float, float]], tablet_views: int = 300) -> pd.DataFrame:
    """A ``device_matrix``-shaped frame: product -> (overall cr_adj, desktop, mobile, tablet index)."""
    rates = {"desktop": 0.006, "mobile": 0.004, "tablet": 0.003}
    rows = []
    for key, (adj, *idx) in spec.items():
        for dev, ix in zip(config.DEVICES, idx):
            views = tablet_views if dev == "tablet" else 4000
            rows.append({"product_key": key, "item_name": key.title(), "device": dev, "views": views,
                         "purchases": int(views * rates[dev] * ix), "cr_raw": rates[dev] * ix, "cr_adj_d": rates[dev] * ix,
                         "device_rate": rates[dev], "index": ix, "cr_adj": adj, "p_bar": 0.005})
    return pd.DataFrame(rows)


SPEC = {  # alpha: strongly desktop; beta: mobile; gamma: as usual everywhere; delta: within +-19% but a strong lean
    "alpha tee": (0.010, 8.0, 0.5, 1.0),
    "beta mug": (0.004, 0.1, 1.6, 0.9),
    "gamma bag": (0.005, 1.0, 1.05, 1.0),
    "delta pen": (0.005, 1.1, 0.9, 1.0),
}
LEAN = pd.DataFrame({"product_key": list(SPEC), "lean": ["Desktop-first", "Mobile-first", "Both", "Desktop-first"]})


def _heat(spec=SPEC, lean=LEAN, codes=None, **kw) -> annotate.AnchorSet:
    from ga4app import charts

    cells = charts.heatmap_cells(make_matrix(spec, **kw), codes=codes if codes is not None else {"alpha tee": "W1"})
    return annotate.heatmap_anchors(cells, lean, scope={"filters": "Nov 01, 2020 – Jan 25, 2021", "candidates": "Pick products"})


def test_heatmap_anchors_follow_the_cells():
    from ga4app import charts

    anchors = _heat()
    assert anchors.chart == annotate.CHART_HEATMAP
    ids = anchors.ids()
    # Rows run by overall adjusted conversion: alpha, delta, gamma (a tie, by key), then beta.
    assert "W1·desktop" in ids and "R4·mobile" in ids  # the code, else the row number
    assert not any(i.startswith("R3") for i in ids)  # gamma bag: as usual everywhere, no lean
    assert {"R2·desktop", "R2·mobile"} <= set(ids)  # delta pen: within +-19% but a strong lean
    cells = charts.heatmap_cells(make_matrix(SPEC), codes={"alpha tee": "W1"}).set_index("anchor")
    w1 = anchors.by_id["W1·desktop"]
    assert w1["x_usual"] == round(cells.loc["W1·desktop", "rel"], 2)  # the number behind the cell's colour
    assert w1["conversion_pct"] == round(100 * cells.loc["W1·desktop", "shown"], 2)  # the cell's text
    assert w1 == {"id": "W1·desktop", "name": "Alpha Tee", "device": "desktop", "conversion_pct": 4.8, "x_usual": 4.0,
                  "vs_usual_pct": 300, "views": 4000, "purchases": 192,
                  "other_devices_pct": {"mobile": 0.2, "tablet": 0.3}, "lean": "Desktop-first"}
    assert anchors.keys["R4·mobile"] == "beta mug"


def test_heatmap_anchors_skip_sparse_tablet_cells():
    assert any(i.endswith("tablet") for i in _heat(tablet_views=300).ids())
    sparse = _heat(tablet_views=10)
    assert not any(i.endswith("tablet") for i in sparse.ids())
    assert "tablet" not in sparse.by_id["W1·desktop"]["other_devices_pct"]  # "–" cells are not quoted either


def test_heatmap_notes_may_compare_devices_and_quote_the_percent_difference():
    """Seen in the first live run: "9.82% on desktop vs 2.15% on mobile" and "24% above its usual
    rate" are true, so each cell carries the product's other devices and vs_usual_pct."""
    anchors = _heat()
    beta = anchors.by_id["R4·mobile"]  # x_usual 2.0
    text = "Beta Mug converts 0.64% on mobile vs 0.06% on desktop, 100% above its usual rate."
    assert annotate.unmatched_numbers(text, beta, anchors.scope) == []
    assert annotate.unmatched_numbers("Beta Mug is 3.1× its desktop rate.", beta, anchors.scope) == ["3.1"]


def test_heatmap_anchors_are_capped_and_keep_strong_leans():
    rng = np.random.default_rng(4)
    spec = {f"p{i:02d}": (0.005, *rng.uniform(0.2, 4.0, 2), 1.0) for i in range(30)}
    spec["lean one"] = (0.005, 1.05, 0.95, 1.0)  # weak cells, but a strong lean
    lean = pd.DataFrame({"product_key": list(spec), "lean": ["Both"] * 30 + ["Mobile-first"]})
    anchors = _heat(spec, lean, codes={})
    assert len(anchors.anchors) == config.HEATMAP_ANCHORS_MAX == 30
    assert sum(a["lean"] == "Mobile-first" for a in anchors.anchors) == 3  # all its readable cells
    rows = [int(i.split("·")[0][1:]) for i in anchors.ids()]
    assert rows == sorted(rows)  # reading order


def test_rule_based_heatmap_notes():
    result = annotate.rule_based_notes(_heat())
    assert result.dropped == []
    assert [(n.anchor, n.kind, n.label) for n in result.notes] == [
        ("W1·desktop", "highlight", "Desktop-first hero"),  # the strongest favoured cell first (4x)
        ("R4·mobile", "highlight", "Mobile-first hero"),  # beta mug (2x); delta pen's 1.1x comes third
        ("R2·mobile", "caution", "Weak on mobile"),  # strongest remaining cell of another product (0.9x)
    ]
    assert result.notes[0].detail == "Alpha Tee converts at 4× its usual rate on desktop: 4.8% on 4,000 views."


def test_rule_based_heatmap_caution_for_a_weak_cell():
    spec = {"a": (0.005, 1.0, 1.0, 1.0), "b": (0.005, 0.3, 1.0, 1.0)}
    lean = pd.DataFrame({"product_key": ["a", "b"], "lean": ["Both", "Both"]})
    (note,) = annotate.rule_based_notes(_heat(spec, lean, codes={})).notes
    assert note.kind == "caution" and note.label == "Weak on desktop"


def _snapshot_heat(data_dir: Path) -> annotate.AnchorSet:
    from ga4app import charts

    snap = data.load_snapshot(data_dir)
    m = metrics.season_prior_strength(snap.item_daily, snap.product_dim)
    products, _ = metrics.compute_products(snap.item_daily, snap.product_dim, Filters(), m)
    eligible = products[products["eligible"].astype(bool)].sort_values(["cr_adj", "views"], ascending=False)
    keys = eligible[eligible["quadrant"].isin(["Winners", "Hidden gems"])]["product_key"].astype(str).head(40).tolist()
    matrix = metrics.device_matrix(snap.item_daily, snap.product_dim, keys, Filters(), m)
    codes = {k: c for k, c in zip(products["product_key"].astype(str), products["code"].astype(str)) if c}
    return annotate.heatmap_anchors(charts.heatmap_cells(matrix, codes=codes), metrics.device_lean(matrix))


def test_rule_based_heatmap_notes_pass_on_the_synthetic_snapshot(synthetic_dir):
    anchors = _snapshot_heat(synthetic_dir)
    result = annotate.rule_based_notes(anchors)
    assert result.notes and result.dropped == []
    assert any(n.label in ("Mobile-first hero", "Desktop-first hero") for n in result.notes)


def test_rule_based_heatmap_notes_pass_on_the_real_snapshot(real_data_dir):
    result = annotate.rule_based_notes(_snapshot_heat(real_data_dir))
    assert result.notes and result.dropped == []

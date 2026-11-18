"""AppTest smoke tests for the shell, Feature Picks and Slot Plan (synthetic and real data)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ga4app import config

pytestmark = pytest.mark.app
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(monkeypatch: pytest.MonkeyPatch, data_dir: Path, **env: str) -> AppTest:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(data_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _scatter_spec(at: AppTest) -> dict:
    """The Feature Picks scatter's Plotly JSON (the chart with the quadrant badges)."""
    import json

    specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    return next(s for s in specs if any(a.get("name") == "badge:Winners" for a in s["layout"].get("annotations", [])))


def _note_names(at: AppTest) -> list[str]:
    return [a["name"] for a in _scatter_spec(at)["layout"]["annotations"] if str(a.get("name", "")).startswith("note:")]


def _note_lines(at: AppTest) -> list[str]:
    import re

    return [m.value for m in at.markdown if re.match(r"^\*\*\d( !)?\*\* ", m.value)]


def test_feature_picks_smoke(monkeypatch, synthetic_dir):
    at = _run(monkeypatch, synthetic_dir)
    assert len(at.metric) >= 5
    assert at.get("plotly_chart")
    assert at.dataframe
    assert "**Verdict:**" in _markdown(at)
    assert any("SYNTHETIC" in w.value for w in at.warning)


def test_slot_plan_page(monkeypatch, synthetic_dir):
    at = _run(monkeypatch, synthetic_dir)
    at.switch_page("views/slot_plan.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(s.value == "Black Friday" for s in at.subheader)
    assert any(s.value == "Retire from the homepage" for s in at.subheader)
    assert "**Verdict:** Run the plan: lead Black Friday with" in _markdown(at)  # the verdict is on the plan


def test_device_pills_and_reset(monkeypatch, synthetic_dir):
    at = _run(monkeypatch, synthetic_dir)
    at.session_state["dev"] = ["mobile"]
    at.run()
    assert not at.exception
    assert "mobile" in at.caption[0].value or any("mobile" in c.value for c in at.caption)
    at.button(key="reset_filters").click().run()
    assert not at.exception
    assert list(at.session_state["dev"]) == []


def test_preset_sets_dates_and_editing_dates_clears_it(monkeypatch, synthetic_dir):
    at = _run(monkeypatch, synthetic_dir)
    p = config.PERIODS["cyber_week"]
    at.button_group(key="preset").set_value("cyber_week").run()
    assert not at.exception
    assert tuple(at.session_state["dates"]) == (p.start, p.end)
    assert any("Nov 23, 2020" in c.value for c in at.caption)
    at.date_input(key="dates").set_value((p.start, config.PERIODS["december"].end)).run()
    assert not at.exception
    assert at.session_state["preset"] is None


def test_inspector_shows_product(monkeypatch, synthetic_dir, synthetic_truth):
    at = _run(monkeypatch, synthetic_dir)
    at.session_state["inspect_key"] = synthetic_truth["hidden_gems"][0]
    at.run()
    assert not at.exception
    assert any(b.label == "Add to slot plan" for b in at.button)
    assert any(m.label == "Adjusted conversion" for m in at.metric)


def test_ai_verdict_fake_stream(monkeypatch, synthetic_dir):
    from ga4app import llm

    at = _run(monkeypatch, synthetic_dir, **{config.LLM_FAKE_ENV: "1"})
    at.secrets["OPENROUTER_API_KEY"] = "test"
    at.run()
    at.button(key="fp_ai_write").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["ai_cache"], "the fake verdict should be cached"
    entry = next(iter(at.session_state["ai_cache"].values()))
    assert entry["model"] in (llm.FAKE_MODEL, config.DEFAULT_MODEL)
    assert entry["text"].strip()
    assert "[G1]" in _markdown(at)


def test_real_data_smoke(monkeypatch, real_data_dir):
    at = _run(monkeypatch, real_data_dir)
    assert "**Verdict:**" in _markdown(at)
    assert not any("SYNTHETIC" in w.value for w in at.warning)
    at.switch_page("views/slot_plan.py").run()
    assert not at.exception, [e.value for e in at.exception]


def test_url_seeds_the_filters_and_follows_changes(monkeypatch, synthetic_dir):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(synthetic_dir))
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params.update({"preset": "cyber_week", "dev": ["mobile", "laptop"], "cat": "Bags", "minv": "500"})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    p = config.PERIODS["cyber_week"]
    assert tuple(at.session_state["dates"]) == (p.start, p.end)
    assert list(at.session_state["dev"]) == ["mobile"]  # unknown values are ignored
    assert list(at.session_state["cat"]) == ["Bags"] and at.session_state["minv"] == 500
    assert at.query_params == {"preset": "cyber_week", "dev": "mobile", "cat": "Bags", "minv": "500"}
    at.button(key="reset_filters").click().run()
    assert not at.exception and at.query_params == {}  # defaults leave the URL clean


def test_url_custom_range_is_clamped_to_the_item_data(monkeypatch, synthetic_dir):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(synthetic_dir))
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params.update({"start": "2021-01-10", "end": "2021-01-31", "minv": "7"})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert tuple(at.session_state["dates"]) == (date(2021, 1, 10), config.ITEM_DATA_END)
    assert at.session_state["preset"] is None and at.session_state["minv"] == config.DEFAULT_MIN_VIEWS


def test_slot_plan_pins_and_exclusions(monkeypatch, synthetic_dir):
    from ga4app import data

    def card_names(at: AppTest) -> list[str]:
        out = []
        for m in at.markdown:
            head, _, rest = m.value.partition(". ")
            if head.startswith("**") and head[2:].isdigit() and rest.endswith("**"):
                out.append(rest[:-2])
        return out

    at = _run(monkeypatch, synthetic_dir)
    at.switch_page("views/slot_plan.py").run()
    assert list(at.session_state["sp_include"]) == []
    snap = data.load_snapshot(synthetic_dir)
    key_of = dict(zip(snap.product_dim["item_name"].astype(str), snap.product_dim["product_key"].astype(str)))
    lead = card_names(at)[0]  # Black Friday slot 1
    totals = snap.item_daily.groupby("product_key", observed=True)[["views", "purchases"]].sum()
    busy = totals[totals["views"] >= 1000]
    weak = str((busy["purchases"] / busy["views"]).idxmin())  # a high-traffic, low-converting product
    assert data.name_of(snap, weak) not in card_names(at)
    at.session_state["sp_excluded"] = [key_of[lead]]
    at.session_state["sp_pinned"] = [weak]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert lead not in card_names(at)  # excluded
    assert data.name_of(snap, weak) in card_names(at)  # pinned into a slot
    assert list(at.session_state["sp_include"]) == [weak] and list(at.session_state["sp_exclude"]) == [key_of[lead]]


def test_chart_notes_toggle_draws_rule_based_notes(monkeypatch, synthetic_dir):
    at = _run(monkeypatch, synthetic_dir)
    assert at.toggle(key="fp_notes_on").value is False  # off by default
    assert _note_names(at) == [] and _note_lines(at) == []
    at.toggle(key="fp_notes_on").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    names = _note_names(at)
    assert names and set(names) == {"note:rule"}
    lines = _note_lines(at)
    assert len(lines) == len(names) and lines[0].startswith("**1")
    assert any("Rule-based notes" in c.value for c in at.caption)


def test_chart_note_inspect_button_sets_the_inspector(monkeypatch, synthetic_dir):
    at = _run(monkeypatch, synthetic_dir)
    at.toggle(key="fp_notes_on").set_value(True).run()
    at.button(key="fp_note_inspect_1").click().run()
    assert not at.exception, [e.value for e in at.exception]
    from ga4app import data

    key = at.session_state["inspect_key"]
    assert key and any(b.label == "Add to slot plan" for b in at.button)
    name = data.name_of(data.load_snapshot(synthetic_dir), key)
    assert name in _note_lines(at)[0]  # the product the first note names

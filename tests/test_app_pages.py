"""AppTest smoke tests for the shell, Feature Picks and Slot Plan (synthetic and real data)."""

from __future__ import annotations

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
    assert any(b.label.startswith("Add") for b in at.button)
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

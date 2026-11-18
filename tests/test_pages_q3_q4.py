"""AppTest checks for the Holiday Shift (Q3) and Device Split (Q4) pages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ga4app import config

pytestmark = pytest.mark.app
APP = str(Path(__file__).resolve().parent.parent / "app.py")
HOLIDAY = "views/holiday_shift.py"
DEVICE = "views/device_split.py"


def _page(monkeypatch: pytest.MonkeyPatch, data_dir: Path, page: str) -> AppTest:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(data_dir))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _charts(at: AppTest) -> int:
    return len(at.get("plotly_chart"))


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_holiday_shift_synthetic(monkeypatch, synthetic_dir):
    at = _page(monkeypatch, synthetic_dir, HOLIDAY)
    assert any(t.value == "Holiday Shift" for t in at.title)
    assert _charts(at) == 3  # daily line, category bars, transition matrix
    assert len(at.metric) >= 4
    box = at.selectbox(key="hs_compare")
    assert tuple(box.value) == config.COMPARISONS[0]
    box.set_value(("january_4wk", "december")).run()
    _ok(at)
    assert _charts(at) == 3
    assert any("December gifting push" in s.value for s in at.subheader)


def test_holiday_shift_finds_apparel_uplift(monkeypatch, synthetic_dir, synthetic_truth):
    """The planted Apparel Cyber Week uplift shows in the plain-English summary."""
    assert synthetic_truth["holiday_uplift"]["Apparel"] > 1
    at = _page(monkeypatch, synthetic_dir, HOLIDAY)
    text = _markdown(at)
    better = next(line for line in text.splitlines() if "Converts better in Cyber Week" in line)
    assert "Apparel" in better


def test_device_split_synthetic(monkeypatch, synthetic_dir):
    at = _page(monkeypatch, synthetic_dir, DEVICE)
    assert any(t.value == "Device Split" for t in at.title)
    assert _charts(at) == 1
    assert len(at.metric) == len(config.DEVICES)
    text = _markdown(at)
    assert "Mobile-first heroes" in text and "Desktop-first heroes" in text
    assert len(at.dataframe) >= 2  # lean table + funnel table (+ hidden chart twin)
    at.button_group(key="ds_mode").set_value("Top adjusted rate").run()
    _ok(at)
    at.slider(key="ds_n").set_value(40).run()
    _ok(at)
    at.button_group(key="ds_mode").set_value("Pick products").run()
    _ok(at)
    assert at.multiselect(key="ds_pick").value


def test_device_split_finds_planted_leans(monkeypatch, synthetic_dir, synthetic_truth):
    at = _page(monkeypatch, synthetic_dir, DEVICE)
    at.button_group(key="ds_mode").set_value("Pick products").run()
    mob, desk = synthetic_truth["mobile_first"], synthetic_truth["desktop_first"]
    at.multiselect(key="ds_pick").set_value([mob, desk]).run()
    _ok(at)
    table = next(d.value for d in at.dataframe if "rr" in d.value.columns)
    lean = table.set_index("product_key")["lean"]
    assert lean[mob] == "Mobile-first"
    assert lean[desk] == "Desktop-first"


@pytest.mark.parametrize("page", [HOLIDAY, DEVICE])
def test_pages_real_data(monkeypatch, real_data_dir, page):
    at = _page(monkeypatch, real_data_dir, page)
    assert _charts(at) >= 1
    assert len(at.metric) >= 3


@pytest.mark.parametrize("focus", ["black_friday", "cyber_monday"])
def test_holiday_shift_short_windows(monkeypatch, synthetic_dir, focus):
    """Black Friday weekend (4 days) and Cyber Monday (2 days): wide intervals, faint low-sample bars, no errors."""
    assert ("baseline", focus) in config.COMPARISONS
    at = _page(monkeypatch, synthetic_dir, HOLIDAY)
    at.selectbox(key="hs_compare").set_value(("baseline", focus)).run()
    _ok(at)
    assert _charts(at) == 3
    assert any(config.PERIODS[focus].label in s.value for s in at.subheader)
    # Every bar carries an opacity; on the 2-day Cyber Monday window some categories have under
    # 10 purchases (low sample) and are drawn faint.
    bars = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    opacities = [o for fig in bars for t in fig["data"] if t.get("type") == "bar"
                 for o in (t.get("marker", {}).get("opacity") or [])]
    assert opacities and set(opacities) <= {0.35, 1.0}
    if focus == "cyber_monday":
        assert 0.35 in opacities


@pytest.mark.parametrize("focus", ["black_friday", "cyber_monday"])
def test_holiday_shift_short_windows_real(monkeypatch, real_data_dir, focus):
    at = _page(monkeypatch, real_data_dir, HOLIDAY)
    at.selectbox(key="hs_compare").set_value(("baseline", focus)).run()
    _ok(at)
    assert _charts(at) == 3



def _specs(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _notes_in(spec: dict) -> list[str]:
    return [a["name"] for a in spec["layout"].get("annotations", []) if str(a.get("name", "")).startswith("note:")]


def test_holiday_shift_chart_notes(monkeypatch, synthetic_dir):
    at = _page(monkeypatch, synthetic_dir, HOLIDAY)
    assert at.toggle(key="hs_notes_on").value is False
    assert not any(_notes_in(s) for s in _specs(at))
    at.toggle(key="hs_notes_on").set_value(True).run()
    _ok(at)
    bars = next(s for s in _specs(at) if any(t.get("type") == "bar" for t in s["data"]))
    assert _notes_in(bars) and set(_notes_in(bars)) == {"note:rule"}
    assert any(m.value.startswith("**1** Biggest rise: Apparel converts") for m in at.markdown)
    # The comparison is a data control: new anchors, still rule-based notes, no errors.
    at.selectbox(key="hs_compare").set_value(("january_4wk", "december")).run()
    _ok(at)
    assert _charts(at) == 3


def test_holiday_shift_ai_notes_fake(monkeypatch, synthetic_dir):
    monkeypatch.setenv(config.LLM_FAKE_ENV, "1")
    at = _page(monkeypatch, synthetic_dir, HOLIDAY)
    at.secrets["OPENROUTER_API_KEY"] = "test"
    at.button(key="hs_notes_ai").click().run()
    _ok(at)
    bars = next(s for s in _specs(at) if any(t.get("type") == "bar" for t in s["data"]))
    assert _notes_in(bars) == ["note:ai", "note:ai"]
    assert at.session_state["notes_cache"]


def test_device_split_chart_notes(monkeypatch, synthetic_dir, synthetic_truth):
    at = _page(monkeypatch, synthetic_dir, DEVICE)
    at.button_group(key="ds_mode").set_value("Pick products").run()
    at.multiselect(key="ds_pick").set_value([synthetic_truth["mobile_first"], synthetic_truth["desktop_first"]]).run()
    at.toggle(key="ds_notes_on").set_value(True).run()
    _ok(at)
    (heat,) = _specs(at)
    shapes = [s for s in heat["layout"].get("shapes", []) if s.get("name") == "note:rule"]
    assert shapes and len(shapes) == len(_notes_in(heat))
    lines = [m.value for m in at.markdown if m.value.startswith("**1")]
    assert lines and ("Mobile-first hero" in lines[0] or "Desktop-first hero" in lines[0])
    # A different candidate list is a data control: new anchors, no errors.
    at.button_group(key="ds_mode").set_value("Top adjusted rate").run()
    _ok(at)

"""AppTest smoke tests for Outside Signal, Cannibalization Check and Data & Method."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ga4app import config

pytestmark = pytest.mark.app
APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ("views/outside_signal.py", "views/cannibalization.py", "views/data_method.py")


def _page(monkeypatch: pytest.MonkeyPatch, data_dir: Path, page: str) -> AppTest:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(data_dir))
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [i.value for i in at.info] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_on_synthetic(monkeypatch, synthetic_dir, page):
    at = _page(monkeypatch, synthetic_dir, page)
    assert at.title


def test_outside_signal_flags_trending_group(monkeypatch, synthetic_dir, synthetic_truth):
    at = _page(monkeypatch, synthetic_dir, PAGES[0])
    text = _text(at)
    assert synthetic_truth["trending_group"] in text
    assert "compared only with itself" in text
    assert at.get("plotly_chart")
    # Benchmarked against the general holiday tide, and the sidebar dates are off on this page.
    assert "vs other categories" in text and "Generic gift keywords" in text
    assert at.date_input(key="dates").disabled
    assert "date filter is off" in text
    assert "The sidebar dates do not apply here" in text


def test_cannibalization_page_has_local_baseline_and_caveat(monkeypatch, synthetic_dir):
    text = _text(_page(monkeypatch, synthetic_dir, PAGES[1]))
    assert "nearby" in text and "p and q are optimistic" in text


def test_outside_signal_no_data_state(monkeypatch, synthetic_dir, tmp_path):
    copy = tmp_path / "snap"
    shutil.copytree(synthetic_dir, copy)
    (copy / "trends_daily.parquet").unlink()
    at = _page(monkeypatch, copy, PAGES[0])
    info = "\n".join(i.value for i in at.info)
    assert "python scripts/fetch_trends.py" in info
    assert "manual_csv" in info
    assert not at.get("plotly_chart")


def test_cannibalization_finds_planted_pair(monkeypatch, synthetic_dir, synthetic_truth):
    at = _page(monkeypatch, synthetic_dir, PAGES[1])
    pair = synthetic_truth["cannibal_pair"]
    at.selectbox(key="cn_a").set_value(pair["a"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Correlation, not causation" in "\n".join(w.value for w in at.warning)
    assert len(at.get("plotly_chart")) == 2
    tables = [d.value for d in at.dataframe]
    assert any((t.astype(str) == "Likely cannibalized").any().any() for t in tables)
    at.button_group(key="cn_ratio").set_value(3.0).run()
    assert not at.exception, [e.value for e in at.exception]


def test_data_method_live_button_and_latex(monkeypatch, synthetic_dir):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("APPDATA", str(synthetic_dir / "no_such_dir"))
    at = _page(monkeypatch, synthetic_dir, PAGES[2])
    assert at.button(key="dm_live").disabled
    assert len(at.latex) >= 3
    assert any(s.value == "SQL" for s in at.subheader)


def test_real_data_smoke(monkeypatch, real_data_dir):
    for page in PAGES:
        at = _page(monkeypatch, real_data_dir, page)
        assert at.title
    at = _page(monkeypatch, real_data_dir, PAGES[0])
    assert "Lifestyle & Gifts" in _text(at)
    at = _page(monkeypatch, real_data_dir, PAGES[2])
    assert "4,295,584" in "\n".join(str(m.value) for m in at.metric)


def test_data_method_shows_raw_m_on_real_data(monkeypatch, real_data_dir):
    text = _text(_page(monkeypatch, real_data_dir, PAGES[2]))
    # About 66 on the season through Jan 25 (70 when the item-less Jan 26-31 days were still counted).
    assert "floored at 100 for stability" in text and "estimated at about 66" in text

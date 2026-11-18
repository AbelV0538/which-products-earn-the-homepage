"""Tests for ``ga4app.charts`` on small hand-made frames (no metrics / phase2 imports)."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ga4app import charts, config
from ga4app.types import PRODUCT_COLUMNS, ChartNote, NotesResult, Thresholds

COLOR_MAP = {
    "Apparel": "#2a78d6",
    "Drinkware": "#eb6834",
    "Bags": "#1baf7a",
    "Office & Stationery": "#8A8F98",
    "Other": "#8A8F98",
}
TH = Thresholds(views=1000.0, conversion=0.005, views_rule="median")
GRAY = config.PALETTE["other"].lower()
CHARTS_PY = Path(charts.__file__)


# --------------------------------------------------------------------------- fixtures


def _products(seed: int = 3) -> pd.DataFrame:
    """A products frame with every PRODUCT_COLUMNS column, quadrants and codes from TH."""
    rng = np.random.default_rng(seed)
    groups = ["Apparel"] * 10 + ["Drinkware"] * 8 + ["Bags"] * 6 + ["Office & Stationery"] * 5 + ["Other"] * 3
    n = len(groups)
    views = np.round(10 ** rng.uniform(2.4, 4.5, n)).astype(int)
    cr = rng.uniform(0.0, 0.011, n)
    cr[0] = 0.09  # far above the 99th percentile -> overflow marker
    views[0] = 600
    views[-2:] = (120, 150)  # below min_views -> not eligible
    purchases = np.round(views * cr).astype(int)
    cr_raw = purchases / views
    cr_adj = 0.6 * cr_raw + 0.4 * 0.005
    eligible = views >= 200
    hi_v, hi_c = views >= TH.views, cr_raw >= TH.conversion
    quadrant = np.select(
        [hi_v & hi_c, ~hi_v & hi_c, hi_v & ~hi_c], ["Winners", "Hidden gems", "Wasted exposure"], default="Low priority"
    ).astype(object)
    quadrant[~eligible] = ""
    confidence = np.where(cr_adj > 0.007, "above", np.where(cr_adj < 0.003, "below", "uncertain"))
    df = pd.DataFrame(
        {
            "product_key": [f"product {i:02d}" for i in range(n)],
            "item_name": [f"Product <{i:02d}> & Co" for i in range(n)],
            "category_group": groups,
            "brand": "Google",
            "views": views,
            "purchases": purchases,
            "units": purchases,
            "revenue_usd": purchases * 21.5,
            "promo_views": 0,
            "cr_raw": cr_raw,
            "prior_mean": 0.005,
            "cr_adj": cr_adj,
            "ci_low": cr_adj * 0.7,
            "ci_high": cr_adj * 1.3,
            "confidence": confidence,
            "eligible": eligible,
            "quadrant": quadrant,
            "forgone_purchases": views * np.maximum(0.005 - cr_adj, 0),
            "lift_per_1k": 1000 * np.maximum(cr_adj - 0.005, 0),
            "rpv": purchases * 21.5 / views,
            "action_rank": np.nan,
            "code": "",
            "views_share": views / views.sum(),
            "purchases_exceed_views": False,
            "checkouts": (views * 0.03).astype(int),
            "views_undercount": False,
        }
    )
    for quad, prefix in config.QUADRANT_CODE_PREFIX.items():
        idx = df.index[df["quadrant"] == quad]
        ranked = df.loc[idx].sort_values("cr_adj", ascending=quad == "Wasted exposure").index[:5]
        for rank, i in enumerate(ranked, start=1):
            df.loc[i, "action_rank"] = rank
            df.loc[i, "code"] = f"{prefix}{rank}"
    assert list(df.columns) == list(PRODUCT_COLUMNS)
    return df


def _device_matrix(*, with_log2: bool = True) -> pd.DataFrame:
    rows = []
    spec = {  # product -> (overall cr_adj, desktop index, mobile index, tablet index)
        "alpha tee": (0.010, 8.0, 0.5, 1.0),
        "beta mug": (0.004, 0.1, 1.6, 0.9),
        "gamma bag": (0.007, 1.0, 1.0, 3.0),
        "delta pen": (0.002, 0.7, 1.3, 1.0),
    }
    rates = {"desktop": 0.006, "mobile": 0.004, "tablet": 0.003}
    for key, (adj, *idx) in spec.items():
        for dev, ix in zip(config.DEVICES, idx):
            views = 300 if dev == "tablet" else 4000
            row = {
                "product_key": key,
                "item_name": key.title(),
                "category_group": "Apparel",
                "device": dev,
                "views": views,
                "purchases": int(views * rates[dev] * ix),
                "cr_raw": rates[dev] * ix,
                "cr_adj_d": rates[dev] * ix,
                "device_rate": rates[dev],
                "index": ix,
                "cr_adj": adj,
                "p_bar": 0.005,
                "device_category": dev,
            }
            if with_log2:
                row["log2_index"] = float(np.log2(ix))  # deliberately unclipped
            rows.append(row)
    return pd.DataFrame(rows)


def _lean() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_key": ["alpha tee", "beta mug", "gamma bag", "delta pen"],
            "lean": ["Desktop-first", "Mobile-first", "Both", "Not enough data"],
            "rr": [0.06, 16.0, 1.0, np.nan],
            "p_value": [0.001, 0.002, 0.9, np.nan],
        }
    )


def _period_df() -> pd.DataFrame:
    rows = []
    for g, (cr_b, cr_f, p) in {"Apparel": (0.004, 0.0062, 0.01), "Drinkware": (0.005, 0.0052, 0.40)}.items():
        for key, cr, is_base in (("baseline", cr_b, True), ("cyber_week", cr_f, False)):
            rows.append(
                {
                    "group": g,
                    "period": key,
                    "period_label": config.PERIODS[key].label,
                    "views": 50_000,
                    "purchases": int(50_000 * cr),
                    "cr": cr,
                    "ci_low": cr * 0.9,
                    "ci_high": cr * 1.1,
                    "lift": np.nan if is_base else cr_f / cr_b - 1,
                    "p_value": np.nan if is_base else p,
                    "low_sample": False,
                    "is_base": is_base,
                }
            )
    return pd.DataFrame(rows)


def _daily(days: int = 40) -> pd.DataFrame:
    d0 = date(2020, 11, 1)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(days):
        for dev in ("desktop", "mobile"):
            v = int(rng.integers(500, 900))
            rows.append({"event_date": d0 + timedelta(days=i), "device_category": dev, "views": v, "purchases": int(v * 0.005)})
    return pd.DataFrame(rows)


def _site_daily_long() -> pd.DataFrame:
    rows = []
    for i in range(30):
        day = date(2020, 11, 1) + timedelta(days=i)
        for dev in ("desktop", "mobile"):
            rows.append({"event_date": day, "device_category": dev, "event_name": "view_item", "events": 1000, "users": 1, "sessions": 1})
            rows.append({"event_date": day, "device_category": dev, "event_name": "purchase", "events": 5 + i % 3, "users": 1, "sessions": 1})
            rows.append({"event_date": day, "device_category": dev, "event_name": "page_view", "events": 9999, "users": 1, "sessions": 1})
    return pd.DataFrame(rows)


def _movers() -> pd.DataFrame:
    pairs = [("Winners", "Winners")] * 3 + [("Hidden gems", "Winners")] * 2 + [("Low priority", "Hidden gems")] + [("Wasted exposure", "")]
    return pd.DataFrame({"product_key": [f"k{i}" for i in range(len(pairs))], "quadrant_base": [a for a, _ in pairs], "quadrant_focus": [b for _, b in pairs]})


def _trend_index() -> pd.DataFrame:
    weeks = pd.date_range("2020-11-08", periods=10, freq="W-SUN")
    rows = []
    for group, kws in {"Apparel": ("hoodie", "t shirt"), "Bags": ("backpack",)}.items():
        for kw in kws:
            for i, w in enumerate(weeks):
                rows.append(
                    {"group_type": "category", "group": group, "keyword": kw, "week": w, "n_days": 7,
                     "interest": 40.0 + i, "site_views": 1000.0, "base_interest": 40.0, "base_site_views": 1000.0,
                     "google_index": 100.0 + 5 * i, "site_views_index": 100.0}
                )
    return pd.DataFrame(rows)


def _spikes() -> pd.DataFrame:
    dates = pd.date_range("2020-11-01", periods=30, freq="D")
    share = np.full(30, 0.05)
    spike_days = [5, 12, 19]
    share[spike_days] = 0.15
    rows = pd.DataFrame(
        {"category_group": "Apparel", "product_key": "alpha tee", "event_date": dates, "views": (share * 2000).astype(int),
         "category_views": 2000, "exposure": (share * 2000).astype(int), "category_exposure": 2000, "share": share,
         "median_share": 0.05, "share_ratio": share / 0.05, "is_spike": share > 0.1, "signal": "views"}
    )
    other = rows.assign(product_key="beta mug", is_spike=False)
    return pd.concat([rows, other], ignore_index=True)


def _tests_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category_group": "Apparel",
            "product_a": "alpha tee",
            "product_b": ["b1", "b2", "b3", "b4", "b5"],
            "item_name_a": "Alpha Tee",
            "item_name_b": ["Beta Mug", "Gamma Bag", "Delta Pen", "Eps Cap", "Zeta Sock"],
            "n_spike_days": 8,
            "O_s": [2, 5, 9, 0, 3],
            "E_s": [6.0, 8.0, 9.0, 4.0, np.nan],
            "rr": [0.35, 0.7, 1.05, 0.0, np.nan],
            "rr_low": [0.15, 0.4, 0.6, 0.0, np.nan],
            "rr_high": [0.7, 1.1, 1.8, np.inf, np.nan],
            "p_value": [0.004, 0.04, 0.6, 0.02, np.nan],
            "q_value": [0.02, 0.2, 0.8, 0.08, np.nan],
            "fisher_p": [0.01, 0.1, 0.7, 0.05, np.nan],
            "tested": [True, True, True, True, False],
            "status": ["Likely cannibalized", "Possible", "No evidence", "Likely cannibalized", "Insufficient data"],
        }
    )


PERIODS = [config.PERIODS["baseline"], config.PERIODS["cyber_week"]]


def _all_figures() -> dict[str, go.Figure]:
    return {
        "scatter": charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP),
        "scatter_highlight": charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP, highlight="Bags"),
        "period_bars": charts.period_bars(_period_df()),
        "daily": charts.daily_conversion(_daily(), PERIODS),
        "transition": charts.transition_matrix(_movers()),
        "heatmap": charts.device_heatmap(_device_matrix(), _lean()),
        "trends": charts.trends_small_multiples(_trend_index(), PERIODS),
        "spike": charts.spike_share(_spikes(), "alpha tee", ["cyber_week"]),
        "scatter_log": charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP, y_scale="log"),
        "scatter_notes": charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP, notes=_scatter_notes()),
        "bars_notes": charts.period_bars(_period_df(), notes=[ChartNote("Apparel·cyber_week", "caution", "a", "b")]),
        "heatmap_notes": charts.device_heatmap(_device_matrix(), _lean(), notes=[ChartNote("R2·mobile", "caution", "a", "b")]),
        "slots": charts.slot_candidates(_slot_scores(), ["k0", "k14"]),
        "device_bars": charts.device_bars(_device_matrix().query("product_key == 'alpha tee'")),
        "daily_gap": charts.daily_conversion(_season_daily(), PERIODS),
        "rr": charts.rr_dotplot(_tests_frame()),
        "timeline": charts.product_timeline(_daily(), PERIODS),
        "empty": charts.empty_figure("nothing"),
    }


def _scatter_notes() -> NotesResult:
    return NotesResult([ChartNote("G1", "highlight", "Best hidden gem", "a", 1),
                        ChartNote("X1", "caution", "Most purchases forgone", "b", 2)], source="rule")


def _slot_scores(n: int = 15) -> pd.DataFrame:
    """A ``metrics.campaign_scores``-shaped frame; k1 is not eligible."""
    score = np.linspace(0.2, 0.05, n)
    views = np.linspace(5, 300, n).astype(int)
    return pd.DataFrame(
        {
            "product_key": [f"k{i}" for i in range(n)], "item_name": [f"Item <{i}>" for i in range(n)],
            "score": score, "score_per_1k": 1000 * score, "ci_low": 0.8 * score, "ci_high": 1.2 * score,
            "views_w": views, "purchases_w": (views * score).astype(int), "window_avg": 0.007,
            "window_weight": views / (views + 100.0), "eligible": [i != 1 for i in range(n)],
        }
    )


def _season_daily() -> pd.DataFrame:
    """Daily totals that stop exactly at config.ITEM_DATA_END (like the trimmed snapshot)."""
    days = pd.date_range(config.DATA_START, config.ITEM_DATA_END)
    return pd.DataFrame({"event_date": days, "views": 1000, "purchases": 5})


def _js(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def _dashed(fig: go.Figure) -> list:
    return [s for s in fig.layout.shapes if s.line is not None and s.line.dash == "dash"]


def _marker_colours(fig: go.Figure) -> set[str]:
    return {t.marker.color.lower() for t in fig.data if isinstance(t.marker.color, str)}


# --------------------------------------------------------------------------- quadrant scatter


def test_scatter_json_contract():
    products = _products()
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP)
    js = _js(fig)
    lay = js["layout"]
    assert lay["xaxis"]["type"] == "log"
    assert lay["xaxis"]["title"]["text"] == "Product views (log scale)"
    assert lay["yaxis"]["title"]["text"] == "Purchases per product view"
    assert lay["height"] == 560
    assert "text" not in lay.get("title", {})
    dashed = [s for s in lay["shapes"] if s.get("line", {}).get("dash") == "dash"]
    assert len(dashed) == 2
    assert {(s["xref"], s["yref"]) for s in dashed} == {("x", "paper"), ("paper", "y")}
    assert any(s["x0"] == TH.views for s in dashed) and any(s["y0"] == TH.conversion for s in dashed)
    badges = [a for a in lay["annotations"] if a.get("xref") == "paper" and a.get("yref") == "paper"]
    assert len(badges) == 4
    assert {re.sub(r"</?b>", "", a["text"]) for a in badges} == {s["badge"] for s in config.QUADRANT_STYLE.values()}
    assert all(a["font"]["color"] == config.PALETTE["ink"] and a["font"]["size"] == 11 for a in badges)
    labels = " ".join(a["text"] for a in lay["annotations"])
    assert "High-traffic cutoff: 1,000 views" in labels and "Average<br>0.50%" in labels
    eligible_keys = set(products.loc[products["eligible"], "product_key"])
    plotted = [row for trace in fig.data for row in trace.customdata]
    assert {row[0] for row in plotted} == eligible_keys  # ineligible products are left out
    assert all(len(row) == 10 for row in plotted)
    first = plotted[0]
    assert first[0] in eligible_keys and first[9] in config.QUADRANTS
    assert "&lt;" in first[1] and "<" not in first[1]  # names are HTML-escaped for the hover


def test_scatter_quadrant_tints_use_data_coordinates():
    fig = charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 4
    assert all(s.xref == "x" and s.yref == "y" and s.opacity == pytest.approx(0.07) for s in rects)
    assert {s.fillcolor for s in rects} == {v["tint"] for v in config.QUADRANT_STYLE.values()}
    # The vertical split is the raw view count (data value), not its log10.
    assert sum(1 for s in rects if s.x1 == TH.views) == 2 and sum(1 for s in rects if s.x0 == TH.views) == 2


def test_scatter_colours_follow_the_map():
    fig = charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP)
    colours = _marker_colours(fig)
    allowed = {c.lower() for c in COLOR_MAP.values()} | {GRAY}
    assert colours <= allowed
    assert len(colours - {GRAY}) <= 3
    by_name = {t.name: t for t in fig.data}
    assert by_name["Apparel"].marker.color == "#2a78d6"
    assert {row[2] for row in by_name["Other"].customdata} == {"Office &amp; Stationery", "Other"}
    # Legend: colour slots in order, "Other" last; "Other" is drawn underneath.
    ranked = sorted(fig.data, key=lambda t: t.legendrank)
    assert [t.name for t in ranked] == ["Apparel", "Drinkware", "Bags", "Other"]
    assert fig.data[0].name == "Other"
    assert fig.layout.showlegend is True


def test_scatter_never_exceeds_three_hues():
    loud = {"Apparel": "#2a78d6", "Drinkware": "#eb6834", "Bags": "#1baf7a", "Office & Stationery": "#aa00aa", "Other": "#00aaaa"}
    fig = charts.quadrant_scatter(_products(), TH, color_map=loud)
    colours = _marker_colours(fig)
    assert len(colours - {GRAY}) == 3
    assert colours - {GRAY} == {"#2a78d6", "#eb6834", "#1baf7a"}


def test_scatter_without_colour_map_is_all_gray():
    fig = charts.quadrant_scatter(_products(), TH, color_map={})
    assert _marker_colours(fig) == {GRAY}
    assert len(fig.data) == 1 and fig.layout.showlegend is False


def test_scatter_highlight_greys_the_rest():
    fig = charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP, highlight="Bags")
    assert _marker_colours(fig) == {config.HIGHLIGHT_COLOR.lower(), GRAY}
    hit = next(t for t in fig.data if t.marker.color == config.HIGHLIGHT_COLOR)
    rest = next(t for t in fig.data if t.marker.color == config.PALETTE["other"])
    assert {row[2] for row in hit.customdata} == {"Bags"}
    assert "Bags" not in {row[2] for row in rest.customdata}
    assert np.allclose(rest.marker.opacity, 0.35)
    assert fig.data[-1] is hit  # drawn on top


def test_scatter_opacity_ring_and_bubble_size():
    products = _products()
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, size="revenue_usd")
    conf = products.set_index("product_key")["confidence"]
    max_rev = products.loc[products["eligible"], "revenue_usd"].max()
    for t in fig.data:
        expected = [0.45 if conf[row[0]] == "uncertain" else 0.9 for row in t.customdata]
        assert np.allclose(t.marker.opacity, expected)
        assert t.marker.line.width == 2 and t.marker.line.color == config.PALETTE["surface"]
        assert t.marker.sizemode == "area"
        assert t.marker.sizeref == pytest.approx(2 * max_rev / 34**2)
        assert t.marker.sizemin == 4  # radius -> 8px minimum diameter


def test_scatter_clips_the_outlier_with_an_overflow_marker():
    products = _products()
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP)
    top = fig.layout.yaxis.range[1]
    assert top < products["cr_raw"].max()
    symbols = [s for t in fig.data for s in t.marker.symbol]
    assert symbols.count("triangle-up") >= 1
    outlier = next(
        (t.hovertext[i], t.y[i]) for t in fig.data for i, row in enumerate(t.customdata) if row[0] == "product 00"
    )
    assert "Off the chart" in outlier[0] and outlier[1] < top
    assert any(a.name == "overflow" for a in fig.layout.annotations)


def test_scatter_labels_only_top_three_codes():
    products = _products()
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP)
    texts = {s for t in fig.data if t.text is not None for s in t.text if s}
    assert texts, "expected some direct labels"
    assert all(re.fullmatch(r"[WGX][1-3]", s) for s in texts)
    assert texts == {c for c in products["code"] if re.fullmatch(r"[WGX][1-3]", c)}
    none = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, label_codes=False)
    assert all(t.text is None for t in none.data)


def test_scatter_adjusted_axis_and_empty_scope():
    products = _products()
    fig = charts.quadrant_scatter(products, TH, y="cr_adj", size="purchases", color_map=COLOR_MAP)
    assert fig.layout.yaxis.title.text.endswith("(adjusted)")
    none = charts.quadrant_scatter(products.assign(eligible=False), TH, color_map=COLOR_MAP)
    assert isinstance(none, go.Figure) and len(none.data) == 0


# --------------------------------------------------------------------------- other builders


def test_heatmap_z_is_clipped_and_rows_ordered():
    for with_log2 in (True, False):
        fig = charts.device_heatmap(_device_matrix(with_log2=with_log2), _lean())
        z = np.concatenate([np.asarray(t.z, dtype=float).ravel() for t in fig.data])
        z = z[np.isfinite(z)]
        assert z.min() >= -1.0 and z.max() <= 1.0
        assert z.min() == -1.0 and z.max() == 1.0
    names = list(fig.layout.yaxis.ticktext)
    assert names == ["Alpha Tee", "Gamma Bag", "Beta Mug", "Delta Pen"]  # overall cr_adj, best first
    tablet = next(t for t in fig.data if list(t.x) == [config.DEVICES.index("tablet")])
    assert tablet.opacity < 1 and tablet.showscale is False
    main = next(t for t in fig.data if 0 in list(t.x))
    assert main.colorscale[0][1] == config.DIVERGING[0] and main.colorscale[-1][1] == config.DIVERGING[2]
    assert main.zmin == -1 and main.zmax == 1
    leans = [a.text for a in fig.layout.annotations if a.name == "lean"]
    assert "<b>Desktop-first</b>" in leans and "<b>Mobile-first</b>" in leans
    cells = [a.text for a in fig.layout.annotations if a.xref == "x"]
    assert "4.80%" in cells  # alpha tee desktop: 0.006 * 8


def test_heatmap_colour_is_device_vs_the_products_usual():
    m = _device_matrix()
    # gamma bag: overall cr_adj 0.007 / p_bar 0.005 = 1.4; give mobile the same index.
    sel = (m["product_key"] == "gamma bag") & (m["device"] == "mobile")
    m.loc[sel, "index"] = 1.4
    fig = charts.device_heatmap(m, _lean())
    names = list(fig.layout.yaxis.ticktext)
    row = names.index("Gamma Bag")
    main = next(t for t in fig.data if 0 in list(t.x))
    col = list(main.x).index(config.DEVICES.index("mobile"))
    assert np.asarray(main.z, dtype=float)[row, col] == pytest.approx(0.0)
    hover = main.text[row][col]
    assert "vs its usual" in hover and "views" in hover and "purchases" in hover
    assert fig.data[0].colorbar.title.text == "Device vs the product's usual (×)"


def test_heatmap_has_one_column_per_device():
    fig = charts.device_heatmap(_device_matrix(), _lean())
    xs = [x for t in fig.data for x in t.x]
    assert sorted(xs) == list(range(len(config.DEVICES)))
    assert list(fig.layout.xaxis.range) == [-0.5, len(config.DEVICES) - 0.5]
    assert all(a.xref == "paper" for a in fig.layout.annotations if a.name == "lean")


def test_scatter_clips_at_the_95th_percentile_with_headroom():
    products = _products()
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP)
    elig = products[products["eligible"].astype(bool) & (products["views"] > 0)]
    cap = float(np.nanquantile(elig["cr_raw"].astype(float), 0.95))
    assert fig.layout.yaxis.range[1] == pytest.approx(cap * charts.Y_HEADROOM)
    ys = [y for t in fig.data for y in t.y]
    assert max(ys) == pytest.approx(cap)


def test_scatter_threshold_labels_sit_outside_the_plot():
    fig = charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP)
    ann = {a.name: a for a in fig.layout.annotations}
    views, conv = ann["threshold:views"], ann["threshold:conversion"]
    assert views.yref == "paper" and views.y >= 1 and views.yanchor == "bottom"
    assert conv.xref == "paper" and conv.x >= 1 and conv.xanchor == "left"
    assert fig.layout.margin.r >= 60
    assert set(fig.layout.xaxis.ticktext) <= {"100", "300", "1k", "3k", "10k", "30k", "100k", "300k", "1M"}


def test_band_labels_are_short_and_staggered():
    fig = charts.daily_conversion(_daily(), ["cyber_week", "december"])
    bands = [a for a in fig.layout.annotations if (a.name or "").startswith("band:")]
    assert [a.text for a in bands] == ["Cyber Week", "Dec push"]
    assert bands[0].yshift != bands[1].yshift


def test_period_bars_lift_labels_only_when_significant():
    fig = charts.period_bars(_period_df())
    assert [t.marker.color for t in fig.data] == [config.PERIOD_COLORS[0], config.PERIOD_COLORS[-1]]
    lifts = [a for a in fig.layout.annotations if a.name == "lift"]
    assert len(lifts) == 1 and lifts[0].text == "<b>+55%</b>"
    focus = fig.data[1]
    assert lifts[0].x == pytest.approx(focus.x[0])  # above Apparel's focus bar
    assert lifts[0].y == pytest.approx(0.0062 * 1.1)
    assert focus.error_y.array is not None and fig.layout.barcornerradius == 4
    assert list(fig.layout.xaxis.ticktext) == ["Apparel", "Drinkware"]


def test_daily_conversion_accepts_item_and_site_frames():
    item = charts.daily_conversion(_daily(), PERIODS)
    assert len(item.data) == 1 and item.data[0].line.width == 2
    day1 = _daily()[_daily()["event_date"] == date(2020, 11, 1)]  # devices are summed per day
    assert item.data[0].y[0] == pytest.approx(day1["purchases"].sum() / day1["views"].sum())
    rects = [s for s in item.layout.shapes if s.type == "rect"]
    assert len(rects) == 2 and all(s.opacity < 0.3 for s in rects)
    assert {a.text for a in item.layout.annotations} >= {charts._SHORT_LABEL[p.key] for p in PERIODS}
    site = charts.daily_conversion(_site_daily_long(), ["baseline"])
    assert site.data[0].y[0] == pytest.approx(10 / 2000)  # page_view rows are ignored


def test_transition_matrix_counts_and_movers_agree():
    from_movers = charts.transition_matrix(_movers())
    quads = list(config.QUADRANTS)
    counts = pd.crosstab(_movers()["quadrant_base"], _movers()["quadrant_focus"]).reindex(index=quads, columns=quads).fillna(0)
    from_counts = charts.transition_matrix(counts)
    assert np.array_equal(np.asarray(from_movers.data[0].z), np.asarray(from_counts.data[0].z))
    z = np.asarray(from_movers.data[0].z)
    assert z.shape == (4, 4) and z.sum() == 6  # the row with no focus quadrant is dropped
    assert z[quads.index("Hidden gems"), quads.index("Winners")] == 2
    assert from_movers.data[0].colorscale[-1][1] == config.PALETTE["navy"]
    assert len(from_movers.layout.annotations) == 16


def test_trends_small_multiples_one_panel_per_group():
    fig = charts.trends_small_multiples(_trend_index(), PERIODS)
    titles = {a.text for a in fig.layout.annotations if a.text in ("Apparel", "Bags")}
    assert titles == {"Apparel", "Bags"}
    assert len(fig.data) == 4
    assert sum(1 for t in fig.data if t.showlegend) == 2
    apparel = next(t for t in fig.data if "Google" in t.name)
    assert apparel.y[0] == pytest.approx(100.0)
    hlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(hlines) == 2 and all(s.y0 == 100 and s.line.dash in (None, "solid") for s in hlines)


def test_spike_share_marks_spike_days():
    fig = charts.spike_share(_spikes(), "alpha tee", ["cyber_week"])
    line, marks = fig.data
    assert len(line.x) == 30 and len(marks.x) == 3
    assert fig.layout.showlegend is True
    assert charts.spike_share(_spikes(), "missing key", None).data == ()


def test_rr_dotplot_status_colours_labels_and_log_axis():
    fig = charts.rr_dotplot(_tests_frame())
    assert fig.layout.xaxis.type == "log"
    by_name = {t.name: t for t in fig.data}
    assert by_name["Likely cannibalized"].marker.color == config.STATUS["critical"]
    assert by_name["Possible"].marker.color == config.STATUS["warning"]
    assert by_name["No evidence"].marker.color == config.PALETTE["other"]
    assert "Insufficient data" not in by_name  # untested pairs are left out
    assert all("Status: <b>" in h for t in fig.data for h in t.hovertext)
    statuses = [a.text for a in fig.layout.annotations if a.name == "status"]
    assert len(statuses) == 4 and "<b>Likely cannibalized</b>" in statuses
    ref = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(ref) == 1 and ref[0].x0 == 1 and ref[0].line.dash is None
    xs = np.concatenate([np.asarray(t.x, dtype=float) for t in fig.data])
    assert xs.min() >= charts.RR_CLIP[0]  # rr = 0 is drawn at the axis edge, not dropped


def test_product_timeline_two_stacked_panels():
    fig = charts.product_timeline(_daily(), PERIODS)
    assert [type(t).__name__ for t in fig.data] == ["Scatter", "Bar"]
    assert fig.data[0].yaxis == "y" and fig.data[1].yaxis == "y2"
    assert fig.layout.xaxis.matches == "x2" or fig.layout.xaxis2.matches == "x"
    assert fig.layout.showlegend is not False


# --------------------------------------------------------------------------- cross-cutting rules


def test_no_dual_axes_anywhere():
    for name, fig in _all_figures().items():
        lay = _js(fig)["layout"]
        for key, axis in lay.items():
            if key.startswith("yaxis"):
                assert "overlaying" not in axis, f"{name}.{key} overlays another axis"


def test_only_the_scatter_thresholds_are_dashed():
    figs = _all_figures()
    for name, fig in figs.items():
        expected = 2 if name.startswith("scatter") else 0
        assert len(_dashed(fig)) == expected, name
        assert all(not (t.line and getattr(t.line, "dash", None) == "dash") for t in fig.data if hasattr(t, "line")), name


def test_template_applied_to_every_figure():
    for name, fig in _all_figures().items():
        tpl = fig.layout.template.layout
        assert tpl.font.family == charts.FONT_FAMILY, name
        assert tpl.paper_bgcolor == "#FFFFFF" and tpl.hoverlabel.bgcolor == "#FFFFFF", name
        assert tpl.hoverlabel.font.color == config.PALETTE["ink"], name
        assert tpl.xaxis.gridcolor == config.PALETTE["grid"], name
        assert not fig.layout.title.text, name


def test_text_never_wears_a_series_colour():
    series = {c.lower() for c in (*config.SCATTER_CATEGORICAL, *config.STATUS.values(), config.PALETTE["coral"])}
    for name, fig in _all_figures().items():
        for a in fig.layout.annotations:
            if a.font is not None and a.font.color:
                assert a.font.color.lower() not in series, (name, a.text)
        for t in fig.data:
            tf = getattr(t, "textfont", None)
            if tf is not None and isinstance(tf.color, str):
                assert tf.color.lower() not in series, name


EMPTY = pd.DataFrame()


@pytest.mark.parametrize(
    "build",
    [
        lambda e: charts.quadrant_scatter(e, TH, color_map=COLOR_MAP),
        lambda e: charts.quadrant_scatter(None, TH, color_map=COLOR_MAP),
        lambda e: charts.period_bars(e),
        lambda e: charts.daily_conversion(e, PERIODS),
        lambda e: charts.daily_conversion(None, None),
        lambda e: charts.transition_matrix(e),
        lambda e: charts.transition_matrix(_movers().iloc[0:0]),
        lambda e: charts.device_heatmap(e, e),
        lambda e: charts.device_heatmap(None, None),
        lambda e: charts.trends_small_multiples(e, PERIODS),
        lambda e: charts.trends_small_multiples(_trend_index().iloc[0:0], PERIODS),
        lambda e: charts.spike_share(e, "x", PERIODS),
        lambda e: charts.rr_dotplot(e),
        lambda e: charts.rr_dotplot(_tests_frame().iloc[4:]),
        lambda e: charts.product_timeline(e, PERIODS),
        lambda e: charts.product_timeline(None, None),
        lambda e: charts.slot_candidates(e),
        lambda e: charts.slot_candidates(None, ["x"]),
        lambda e: charts.device_bars(e),
        lambda e: charts.device_bars(None),
    ],
)
def test_builders_handle_empty_frames(build):
    fig = build(EMPTY)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert any(a.xref == "paper" for a in fig.layout.annotations)
    fig.to_json()


def test_charts_source_rules():
    src = CHARTS_PY.read_text(encoding="utf-8")
    assert "use_container_width" not in src
    assert not re.search(r"^\s*(import|from)\s+streamlit", src, flags=re.M)


# --------------------------------------------------------------------------- improvements


def test_scatter_log_scale_keeps_top_converters_on_the_chart():
    products = _products()
    zero = products.index[products["eligible"] & (products.index > 0)][0]
    products.loc[zero, ["purchases", "cr_raw", "cr_adj"]] = [0, 0.0, 0.002]
    linear = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP)
    log = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, y_scale="log")
    assert any(a.name == "overflow" for a in linear.layout.annotations)  # linear clips product 00 (9%)
    assert log.layout.yaxis.type == "log"
    ann = {a.name: a for a in log.layout.annotations}
    assert "overflow" not in ann and "no purchases" in ann["underflow"].text
    assert ann["threshold:conversion"].y == pytest.approx(np.log10(TH.conversion))  # log10 for annotations
    lo, hi = log.layout.yaxis.range
    ys = np.concatenate([np.asarray(t.y, dtype=float) for t in log.data])
    assert (ys > 0).all() and lo < np.log10(ys.min()) and np.log10(ys.max()) < hi
    assert ys.max() == pytest.approx(products.loc[0, "cr_raw"])  # drawn at its real rate, not clipped
    symbols = [s for t in log.data for s in t.marker.symbol]
    assert "triangle-down" in symbols and "triangle-up" not in symbols
    assert set(log.layout.yaxis.ticktext) <= {t for _, t in charts.LOG_RATE_TICKS}


def test_slot_candidates_marks_picks_and_window_evidence():
    fig = charts.slot_candidates(_slot_scores(), ["k0", "k14"], top=5)
    traces = {t.name: t for t in fig.data}
    assert set(traces) == {"Other candidates", "In the plan"}
    # Top 5 eligible (k1 is not), plus k14: in the plan but outside the top, so it is still shown.
    ticks = list(fig.layout.yaxis.ticktext)
    assert ticks == ["Item &lt;0&gt;", "Item &lt;2&gt;", "Item &lt;3&gt;", "Item &lt;4&gt;", "Item &lt;5&gt;", "Item &lt;14&gt;"]
    assert list(traces["In the plan"].y) == [0, 5]
    assert np.asarray(traces["In the plan"].error_x.array) == pytest.approx(0.2 * np.asarray(traces["In the plan"].x))
    views = [a.text for a in fig.layout.annotations if a.name == "window_views"]
    assert len(views) == 6 and views[0] == "5 (mostly season)" and views[-1] == "300"
    assert any(a.name == "window_avg" and a.text == "Window average 7.0 (off the axis, to the left)" for a in fig.layout.annotations)
    assert fig.layout.xaxis.range[0] > 7  # zoomed to the candidates, not stretched down to the average


def test_device_bars_one_bar_per_device_with_averages():
    one = _device_matrix().query("product_key == 'alpha tee'")
    bars, avg = charts.device_bars(one).data
    assert list(bars.y) == ["Desktop", "Mobile", "Tablet"]
    assert np.asarray(bars.x, dtype=float) == pytest.approx(one["cr_raw"].to_numpy())
    assert np.asarray(avg.x, dtype=float) == pytest.approx([0.006, 0.004, 0.003])
    assert bars.marker.opacity[-1] < 1 == bars.marker.opacity[0]  # tablet faded
    sparse = one.assign(views=np.where(one["device"] == "tablet", 5, one["views"]))
    assert charts.device_bars(sparse).data[0].text[-1] == "too few views"


def test_heatmap_hides_sparse_tablet_cells_and_an_empty_lean_column():
    m = _device_matrix()
    m.loc[(m["device"] == "tablet") & (m["product_key"] == "alpha tee"), "views"] = 5
    fig = charts.device_heatmap(m, _lean())
    assert "–" in [a.text for a in fig.layout.annotations]
    assert any(a.name == "lean" for a in fig.layout.annotations)  # strong leans keep the column
    calm = charts.device_heatmap(_device_matrix(), _lean().assign(lean="Both"))
    assert not any(a.name == "lean" for a in calm.layout.annotations) and calm.layout.margin.r < 100


def test_item_gap_band_only_when_a_series_stops_at_the_cut():
    fig = charts.daily_conversion(_season_daily(), ["cyber_week"])
    gap = [s for s in fig.layout.shapes if s.name == "item_gap"]
    assert len(gap) == 1 and pd.Timestamp(gap[0].x1) > pd.Timestamp(config.DATA_END)
    assert pd.Timestamp(fig.layout.xaxis.range[1]) > pd.Timestamp(config.DATA_END)
    assert not [s for s in charts.daily_conversion(_daily(), ["cyber_week"]).layout.shapes if s.name == "item_gap"]
    timeline = charts.product_timeline(_season_daily(), PERIODS)
    assert len([s for s in timeline.layout.shapes if s.name == "item_gap"]) == 2  # both panels


def test_short_label():
    assert charts.short_label("december") == "Dec push" and charts.short_label("january_4wk") == "Jan baseline"
    assert charts.short_label("post_cutoff") == "After cutoff" and charts.short_label("nope") == "nope"


# --------------------------------------------------------------------------- chart notes


def _note_ann(fig: go.Figure, source: str = "rule") -> list:
    return [a for a in fig.layout.annotations if a.name == f"note:{source}"]


def _code_row(products: pd.DataFrame, code: str) -> pd.Series:
    return products.loc[products["code"] == code].iloc[0]


# Annotation names per figure before chart notes existed: notes=None must keep them.
BASELINE_ANNOTATIONS = {
    "scatter": {"badge:Winners": 1, "badge:Hidden gems": 1, "badge:Wasted exposure": 1, "badge:Low priority": 1,
                "threshold:views": 1, "threshold:conversion": 1, "overflow": 1},
    "scatter_log": {"badge:Winners": 1, "badge:Hidden gems": 1, "badge:Wasted exposure": 1, "badge:Low priority": 1,
                    "threshold:views": 1, "threshold:conversion": 1},
    "period_bars": {"lift": 1},
    "heatmap": {None: 13, "lean": 4},
}


def test_figures_without_notes_are_unchanged():
    from collections import Counter

    figs = _all_figures()
    for name, expected in BASELINE_ANNOTATIONS.items():
        assert dict(Counter(a.name for a in figs[name].layout.annotations)) == expected, name
    for empty in (None, [], NotesResult()):
        assert charts.period_bars(_period_df(), notes=empty).to_json() == figs["period_bars"].to_json()
    products = _products()
    for scale in ("linear", "log"):
        plain = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, y_scale=scale)
        for empty in (None, [], NotesResult()):
            same = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, y_scale=scale, notes=empty)
            assert same.to_json() == plain.to_json()


def test_scatter_note_sits_on_its_bubble_on_log_axes():
    products = _products()
    gem = _code_row(products, "G1")
    notes = NotesResult([ChartNote("G1", "highlight", "Best hidden gem", "Converts well.", 1)], source="ai")
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, y_scale="log", notes=notes)
    (ann,) = _note_ann(fig, "ai")
    assert ann.x == pytest.approx(np.log10(gem["views"]))  # log axes take log10 coordinates
    assert ann.y == pytest.approx(np.log10(gem["cr_raw"]))
    assert ann.showarrow and ann.text == "<b>1</b> Best hidden gem" and ann.hovertext == "Converts well."
    assert ann.bordercolor == config.PALETTE["ink"] and ann.font.color == config.PALETTE["ink"]
    # The box is pushed toward the interior: left of a bubble in the right half, below one in the top third.
    lo, hi = fig.layout.xaxis.range
    assert (ann.ax < 0) == ((ann.x - lo) / (hi - lo) > 0.5)
    y_lo, y_hi = fig.layout.yaxis.range
    assert (ann.ay > 0) == ((ann.y - y_lo) / (y_hi - y_lo) > 2 / 3)
    assert ann.standoff >= charts.BUBBLE_MIN_PX / 2


def test_scatter_note_on_a_clipped_bubble_points_at_the_drawn_cap():
    products = _products()
    products.loc[0, "code"] = "G9"  # product 00 is clipped to the top edge on the linear axis
    notes = [ChartNote("G9", "highlight", "Off the chart", "Clipped.")]
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, y_scale="linear", notes=notes)
    (ann,) = _note_ann(fig)
    drawn = next(t.y[i] for t in fig.data for i, row in enumerate(t.customdata) if row[0] == "product 00")
    assert ann.y == pytest.approx(drawn) and ann.y < products.loc[0, "cr_raw"]
    assert ann.x == pytest.approx(np.log10(products.loc[0, "views"]))
    assert ann.ay > 0  # top third: the box hangs below the bubble


def test_scatter_notes_skip_undrawn_points_and_keep_their_numbers():
    products = _products()
    notes = NotesResult([
        ChartNote("Z9", "highlight", "Not a bubble", "Skipped.", 1),
        ChartNote("X1", "caution", "Most purchases forgone", "Retire it.", 2),
    ], source="rule")
    fig = charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, notes=notes)
    (ann,) = _note_ann(fig)
    assert ann.text == "<b>2</b> <b>!</b> Most purchases forgone"
    assert ann.bordercolor == config.PALETTE["coral"] and ann.font.color == config.PALETTE["ink"]
    unnumbered = [ChartNote("Z9", "highlight", "a", "b"), ChartNote("X1", "caution", "c", "d")]
    (ann,) = _note_ann(charts.quadrant_scatter(products, TH, color_map=COLOR_MAP, notes=unnumbered))
    assert ann.text.startswith("<b>2</b>")  # its place in the list


def test_bar_note_sits_above_its_bar_and_the_y_range_grows():
    plain = charts.period_bars(_period_df())
    notes = NotesResult([ChartNote("Apparel·cyber_week", "highlight", "Biggest rise", "Up 55%.", 1),
                         ChartNote("Drinkware·baseline", "highlight", "Base bar", "Not offered.", 2),
                         ChartNote("Bags·cyber_week", "caution", "Not drawn", "No such bar.", 3)], source="ai")
    fig = charts.period_bars(_period_df(), notes=notes)
    (ann,) = _note_ann(fig, "ai")[:1]
    focus = fig.data[1]
    assert ann.x == pytest.approx(focus.x[0]) and ann.y == pytest.approx(0.0062 * 1.1)  # Apparel's focus whisker top
    assert ann.ax == 0 and ann.ay == -charts.NOTE_BAR_DY and ann.standoff > 17  # clear of the "+55%" label
    lift = next(a for a in fig.layout.annotations if a.name == "lift")
    assert (lift.x, lift.y) == (ann.x, ann.y)
    assert [a.text for a in _note_ann(fig, "ai")] == ["<b>1</b> Biggest rise"]  # base bars and missing bars: skipped
    top = 0.0062 * 1.1
    assert plain.layout.yaxis.range[1] == pytest.approx(top * charts.BAR_HEADROOM)
    assert fig.layout.yaxis.range[1] == pytest.approx(top * charts.NOTE_BAR_HEADROOM)
    unplaced = charts.period_bars(_period_df(), notes=[ChartNote("Nope·x", "highlight", "a", "b")])
    assert unplaced.to_json() == plain.to_json()  # no note drawn: the figure is unchanged


def test_heatmap_cells_match_the_drawn_heatmap():
    m = _device_matrix()
    m.loc[(m["device"] == "tablet") & (m["product_key"] == "alpha tee"), "views"] = 5
    cells = charts.heatmap_cells(m, codes={"beta mug": "G2"})
    fig = charts.device_heatmap(m, _lean())
    assert len(cells) == 4 * len(config.DEVICES)
    assert list(dict.fromkeys(cells["item_name"])) == list(fig.layout.yaxis.ticktext)
    for t in fig.data:
        for i, row in enumerate(np.asarray(t.z, dtype=float)):
            for x, z in zip(t.x, row):
                cell = cells[(cells["row"] == i) & (cells["col"] == x)].iloc[0]
                assert (np.isnan(z) and np.isnan(cell["z"])) or z == pytest.approx(cell["z"])
    sparse = cells[cells["sparse"]]
    assert list(sparse["anchor"]) == ["R1·tablet"]
    assert "G2·mobile" in set(cells["anchor"])
    assert charts.heatmap_cells(None).empty and charts.heatmap_cells(pd.DataFrame()).empty


def test_heatmap_note_adds_a_cell_rectangle_and_a_badge():
    plain = charts.device_heatmap(_device_matrix(), _lean())
    notes = NotesResult([ChartNote("W1·desktop", "highlight", "Desktop-first hero", "Up.", 1),
                         ChartNote("R3·mobile", "caution", "Weak", "Down & <out>.", 2),
                         ChartNote("Z9·mobile", "highlight", "Missing", "No such cell.", 3)], source="rule")
    fig = charts.device_heatmap(_device_matrix(), _lean(), notes=notes, codes={"alpha tee": "W1"})
    rects = [s for s in fig.layout.shapes if s.name == "note:rule"]
    badges = _note_ann(fig)
    assert len(rects) == len(badges) == 2
    names = list(fig.layout.yaxis.ticktext)
    alpha, beta = names.index("Alpha Tee"), names.index("Beta Mug")
    desktop, mobile = config.DEVICES.index("desktop"), config.DEVICES.index("mobile")
    assert rects[0].x0 < desktop < rects[0].x1 and rects[0].y0 < alpha < rects[0].y1
    assert rects[0].line.color == config.PALETTE["ink"] and rects[1].line.color == config.PALETTE["coral"]
    assert rects[1].x0 < mobile < rects[1].x1 and rects[1].y0 < beta < rects[1].y1
    assert badges[0].text == "<b>1</b>" and badges[1].text == "<b>2</b> <b>!</b>"  # no label on the grid
    assert (badges[0].x, badges[0].y) == (desktop - 0.5, alpha - 0.5)
    assert badges[1].hovertext == "Weak: Down &amp; &lt;out&gt;."
    # Without notes the heatmap is unchanged.
    for empty in (None, [], NotesResult()):
        assert charts.device_heatmap(_device_matrix(), _lean(), notes=empty).to_json() == plain.to_json()


def test_note_text_is_escaped():
    notes = [ChartNote("W1", "highlight", "<b>Tee</b> & Co", "Bold <i>claim</i> & more")]
    (ann,) = _note_ann(charts.quadrant_scatter(_products(), TH, color_map=COLOR_MAP, notes=notes))
    assert ann.text == "<b>1</b> &lt;b&gt;Tee&lt;/b&gt; &amp; Co"
    assert ann.hovertext == "Bold &lt;i&gt;claim&lt;/i&gt; &amp; more"

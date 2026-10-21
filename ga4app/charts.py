"""Plotly figure builders for every page, plus the app's light template.

No Streamlit here: builders take plain frames (the ``ga4app.types.PRODUCT_COLUMNS`` contract and
the ``metrics`` / ``phase2`` outputs) and return ``plotly.graph_objects.Figure``. Pages render
them with ``st.plotly_chart(fig, theme=None, width="stretch", key=..., config={"displaylogo": False})``.

Design rules every builder follows: one y-axis per chart (indexing or subplots instead of a
second scale), at most three categorical hues plus "Other" gray on the scatter, colour fixed per
entity, text in ink tokens only, solid hairline grids, and dashed lines only for the two quadrant
thresholds. Empty or missing inputs give ``empty_figure`` instead of raising.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from . import config
from .config import Period
from .types import Thresholds

__all__ = [
    "FONT_FAMILY",
    "TEMPLATE",
    "apply_template",
    "daily_conversion",
    "device_heatmap",
    "empty_figure",
    "period_bars",
    "product_timeline",
    "quadrant_scatter",
    "rr_dotplot",
    "spike_share",
    "transition_matrix",
    "trends_small_multiples",
]

# --------------------------------------------------------------------------- tokens

INK = config.PALETTE["ink"]
INK_2 = config.PALETTE["ink_secondary"]
MUTED = config.PALETTE["muted"]
GRID = config.PALETTE["grid"]
AXIS = config.PALETTE["axis"]
SURFACE = config.PALETTE["surface"]
OTHER = config.PALETTE["other"]
FONT_FAMILY = "Inter, Segoe UI, system-ui, sans-serif"
WHITE_BOX = "rgba(255,255,255,0.85)"
# Two-series line charts reuse the first two categorical slots, always in this order.
SERIES = config.SCATTER_CATEGORICAL

BUBBLE_MAX_PX = 34  # diameter of the largest bubble
BUBBLE_MIN_PX = 8
Y_CLIP_QUANTILE = 0.95
Y_HEADROOM = 1.18  # y range top / clip value: room for overflow markers below the corner badges
Y_CLIP_SLACK = 1.15  # clip only when the max is this far above the quantile (small samples)
LABEL_TOP_CODES = 3  # direct labels for W1-W3, G1-G3, X1-X3
SHADE_OPACITY = 0.22
# Two-window bands: light lavender for the base, a stronger tint for the focus.
SHADE_PAIR = (config.PERIOD_COLORS[0], config.PERIOD_COLORS[1])
WHITE = "#FFFFFF"
_SHORT_LABEL = {
    "full": "Full season", "baseline": "Baseline", "cyber_week": "Cyber Week", "black_friday": "BF weekend",
    "cyber_monday": "Cyber Mon", "december": "Dec push", "post_cutoff": "After cutoff", "january": "January",
    "january_4wk": "Jan baseline",
}
LOG_TICKS = ((100, "100"), (300, "300"), (1_000, "1k"), (3_000, "3k"), (10_000, "10k"), (30_000, "30k"),
             (100_000, "100k"), (300_000, "300k"), (1_000_000, "1M"))
QUADRANT_TINT_OPACITY = 0.07
RR_CLIP = (0.05, 20.0)

_SEQ_NAVY = [[0.0, config.PALETTE["lavender"]], [0.35, "#9AA6D9"], [0.7, "#6270BD"], [1.0, config.PALETTE["navy"]]]
_DIVERGING_SCALE = [[0.0, config.DIVERGING[0]], [0.5, config.DIVERGING[1]], [1.0, config.DIVERGING[2]]]
_STATUS_STYLE = {  # phase2 status -> (colour, marker symbol); the label always travels with it
    "Likely cannibalized": (config.STATUS["critical"], "diamond"),
    "Possible": (config.STATUS["warning"], "square"),
    "No evidence": (OTHER, "circle"),
    "Insufficient data": (OTHER, "circle-open"),
}
_CONFIDENCE_TEXT = {
    "above": "Reliably above average",
    "below": "Reliably below average",
    "uncertain": "Not clearly different from average",
}
_LEAN_STRONG = ("Mobile-first", "Desktop-first")
_CODE_RE = re.compile(r"^([A-Z])(\d+)$")


def _axis_style() -> dict[str, Any]:
    return {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": AXIS,
        "ticklen": 4,
        "tickfont": {"color": MUTED, "size": 12},
        "title": {"font": {"color": INK_2, "size": 12}, "standoff": 8},
        "automargin": True,
    }


TEMPLATE = go.layout.Template(
    layout={
        "font": {"family": FONT_FAMILY, "size": 13, "color": INK},
        "paper_bgcolor": WHITE,
        "plot_bgcolor": WHITE,
        "colorway": list(SERIES),
        "margin": {"l": 56, "r": 24, "t": 40, "b": 48},
        "hoverlabel": {
            "bgcolor": SURFACE,
            "bordercolor": AXIS,
            "font": {"family": FONT_FAMILY, "size": 12, "color": INK},
            "align": "left",
        },
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "font": {"color": INK_2, "size": 12},
            "bgcolor": "rgba(255,255,255,0)",
        },
        "xaxis": _axis_style(),
        "yaxis": _axis_style(),
        "annotationdefaults": {"showarrow": False, "font": {"color": INK_2, "size": 11}},
        "bargap": 0.3,
        "barcornerradius": 4,
    }
)


def apply_template(fig: go.Figure, *, height: int | None = None, **layout: Any) -> go.Figure:
    """Apply the app template (plus optional layout overrides) and return ``fig``."""
    # Backgrounds go on the layout too: the template alone is not always honoured by the renderer.
    layout = {"plot_bgcolor": WHITE, "paper_bgcolor": WHITE, **layout}
    fig.update_layout(template=TEMPLATE, **({"height": height} if height else {}), **layout)
    return fig


def empty_figure(message: str = "No data in this scope.", *, height: int = 320) -> go.Figure:
    """A blank figure with a centred note, for no-data states."""
    fig = go.Figure()
    apply_template(fig, height=height, showlegend=False, margin={"l": 24, "r": 24, "t": 24, "b": 24})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=_esc(message), x=0.5, y=0.5, xref="paper", yref="paper", font={"size": 13, "color": INK_2}
    )
    return fig


# --------------------------------------------------------------------------- helpers


def _has(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and len(df) > 0


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    """A column as float (``default`` when absent; unparseable values become NaN)."""
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _txt(df: pd.DataFrame, col: str, default: str = "") -> np.ndarray:
    """A column as an object array of str (``default`` for missing values)."""
    if col not in df.columns:
        return np.full(len(df), default, dtype=object)
    s = df[col].astype(object)
    return s.where(s.notna(), default).astype(str).to_numpy(dtype=object)


def _esc(text: Any) -> str:
    """Plotly renders a subset of HTML in labels and hovers, so escape data strings."""
    return html.escape(str(text), quote=False)


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x: Any) -> float | None:
    """A JSON-safe float (None for NaN / inf)."""
    return float(x) if _finite(x) else None


def _pct(x: Any, digits: int = 2) -> str:
    return f"{float(x):.{digits}%}" if _finite(x) else "–"


def _dates(values: Any) -> pd.Series:
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _period(p: Any) -> Period | None:
    if isinstance(p, Period):
        return p
    if isinstance(p, str):
        if p in config.PERIODS:
            return config.PERIODS[p]
        if p in config.CAMPAIGNS:
            return config.PERIODS[config.CAMPAIGNS[p]]
        return None
    if all(hasattr(p, a) for a in ("key", "label", "start", "end")):
        return p
    return None


def _periods(periods: Any) -> list[Period]:
    """Periods from a Period, a key, a sequence of either, or a mapping of them (None -> [])."""
    if periods is None:
        return []
    if isinstance(periods, Mapping):
        periods = list(periods.values())
    elif isinstance(periods, (Period, str)) or not isinstance(periods, Iterable):
        periods = [periods]
    return [q for q in (_period(p) for p in periods) if q is not None]


def _ordinal(n: int, colors: Sequence[str] = config.PERIOD_COLORS) -> list[str]:
    """``n`` steps of an ordinal ramp, light to dark (two windows take the ends)."""
    colors = list(colors)
    if n <= 0 or not colors:
        return []
    if n == 1:
        return [colors[-1]]
    if n <= len(colors):
        return [colors[int(i)] for i in np.round(np.linspace(0, len(colors) - 1, n))]
    scale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    return list(sample_colorscale(scale, list(np.linspace(0.0, 1.0, n))))


def _shade(fig: go.Figure, periods: Any, *, row: Any = None, col: Any = None, label: bool = True) -> None:
    """Low-opacity bands for comparison windows (half a day of padding so daily points sit inside)."""
    plist = _periods(periods)
    colours = list(SHADE_PAIR) if len(plist) == 2 else _ordinal(len(plist))
    for i, (p, colour) in enumerate(zip(plist, colours)):
        kw: dict[str, Any] = {
            "x0": (pd.Timestamp(p.start) - pd.Timedelta(hours=12)).isoformat(),
            "x1": (pd.Timestamp(p.end) + pd.Timedelta(hours=12)).isoformat(),
            "fillcolor": colour,
            "opacity": SHADE_OPACITY,
            "line_width": 0,
            "layer": "below",
        }
        if label:
            kw.update(
                annotation_text=_esc(_SHORT_LABEL.get(p.key, p.label)),
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color=INK_2,
                # Stagger heights so labels of adjacent windows never collide.
                annotation={"name": f"band:{p.key}", "yshift": -16 * i, "bgcolor": WHITE_BOX},
            )
        if row is not None:
            kw.update(row=row, col=col)
        fig.add_vrect(**kw)


def _daily_counts(daily: pd.DataFrame | None) -> pd.DataFrame:
    """``event_date, views, purchases`` per calendar day (missing days filled with 0).

    Accepts item rows (``views``, ``purchases``), wide site rows (``view_item``, ``purchase``)
    or long ``site_daily`` rows (``event_name``, ``events``); devices are summed.
    """
    empty = pd.DataFrame({"event_date": pd.Series(dtype="datetime64[ns]"), "views": [], "purchases": []})
    if not _has(daily) or "event_date" not in daily.columns:
        return empty
    cols = set(daily.columns)
    if {"event_name", "events"} <= cols:
        name = _txt(daily, "event_name")
        events = np.nan_to_num(_num(daily, "events", 0.0))
        views, purchases = np.where(name == "view_item", events, 0.0), np.where(name == "purchase", events, 0.0)
    elif "view_item" in cols:
        views, purchases = _num(daily, "view_item", 0.0), _num(daily, "purchase", 0.0)
    else:
        views, purchases = _num(daily, "views", 0.0), _num(daily, "purchases", 0.0)
    frame = pd.DataFrame(
        {
            "event_date": _dates(daily["event_date"]).to_numpy(),
            "views": np.nan_to_num(views),
            "purchases": np.nan_to_num(purchases),
        }
    ).dropna(subset=["event_date"])
    if frame.empty:
        return empty
    g = frame.groupby("event_date").sum().sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D", name="event_date"), fill_value=0.0)
    return g.reset_index()


def _rate(df: pd.DataFrame, col: str) -> np.ndarray:
    """``cr_raw`` (falls back to purchases / views), ``cr_adj`` (falls back to cr_raw) or any column."""
    v, p = _num(df, "views", 0.0), _num(df, "purchases", 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fallback = np.where(v > 0, p / v, np.nan)
    raw = _num(df, "cr_raw")
    raw = np.where(np.isfinite(raw), raw, fallback)
    if col == "cr_raw":
        return raw
    vals = _num(df, col)
    return np.where(np.isfinite(vals), vals, raw) if col == "cr_adj" else vals


def _no_grid(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=False, showline=False, ticks="")


# --------------------------------------------------------------------------- Feature Picks


def _eligible(products: pd.DataFrame | None) -> pd.DataFrame:
    """Eligible products with positive views (log x axis)."""
    if not _has(products):
        return pd.DataFrame()
    df = products
    if "eligible" in df.columns:
        df = df[df["eligible"].astype(object).where(df["eligible"].notna(), False).astype(bool).to_numpy()]
    v = _num(df, "views", 0.0)
    return df[np.isfinite(v) & (v > 0)].reset_index(drop=True)


def _is_gray(colour: str) -> bool:
    return colour.strip().lower() in {OTHER.lower(), MUTED.lower(), "gray", "grey"}


def _slot(colour: str) -> int:
    hues = [c.lower() for c in config.SCATTER_CATEGORICAL]
    return hues.index(colour.lower()) if colour.lower() in hues else len(hues)


def _colour_series(
    groups: np.ndarray, color_map: Mapping[str, str] | None, highlight: str | None
) -> list[tuple[str, str, np.ndarray, int, bool]]:
    """(legend name, colour, row mask, legend rank, dimmed) per trace, in drawing order.

    Default mode keeps at most three non-gray hues (ordered by their categorical slot) and folds
    every other group into one gray "Other" trace drawn underneath. Highlight mode shows the one
    group in ``HIGHLIGHT_COLOR`` over everything else in dimmed gray.
    """
    if highlight:
        hit = groups == highlight
        series = [
            ("Other categories", OTHER, ~hit, 1000, True),
            (str(highlight), config.HIGHLIGHT_COLOR, hit, 0, False),
        ]
        return [s for s in series if s[2].any()]
    cmap = {str(k): str(v).strip() for k, v in (color_map or {}).items() if isinstance(v, str) and v.strip()}
    original = {}
    for g, c in cmap.items():
        if g != "Other" and not _is_gray(c):
            original.setdefault(c.lower(), c)
    hues = sorted(original, key=lambda c: (_slot(c), c))[:3]
    colour_of = np.array([cmap.get(str(g), "").lower() for g in groups], dtype=object)
    coloured = []
    for rank, hue in enumerate(hues):
        mask = colour_of == hue
        if mask.any():
            coloured.append((" / ".join(dict.fromkeys(groups[mask])), original[hue], mask, rank, False))
    other = ~np.isin(colour_of, hues) if hues else np.ones(len(groups), dtype=bool)
    return ([("Other", OTHER, other, 1000, False)] if other.any() else []) + coloured


def _top_code(code: str, top: int = LABEL_TOP_CODES) -> bool:
    m = _CODE_RE.match(code or "")
    return bool(m) and m.group(1) in config.QUADRANT_CODE_PREFIX.values() and int(m.group(2)) <= top


def _y_cap(values: np.ndarray, floor: float) -> float:
    """Clip value for y: the 95th percentile when an outlier sits well above it, else the max."""
    vmax = float(np.nanmax(values))
    cap = float(np.nanquantile(values, Y_CLIP_QUANTILE))
    if not vmax > cap * Y_CLIP_SLACK:
        cap = vmax
    return max(cap, floor * 1.25 if _finite(floor) else 0.0, 1e-6)


_SCATTER_HOVER = (
    "<b>%{customdata[1]}</b><br>"
    "%{customdata[2]} · %{customdata[9]}<br>"
    "Views: %{customdata[3]:,.0f} · Purchases: %{customdata[4]:,.0f}<br>"
    "Conversion: %{customdata[5]:.2%} (adjusted %{customdata[6]:.2%})<br>"
    "Revenue: $%{customdata[7]:,.0f}<br>"
    "%{hovertext}<extra></extra>"
)


def quadrant_scatter(
    products: pd.DataFrame | None,
    th: Thresholds,
    *,
    y: str = "cr_raw",
    size: str = "revenue_usd",
    color_map: Mapping[str, str] | None = None,
    highlight: str | None = None,
    label_codes: bool = True,
    height: int = 560,
) -> go.Figure:
    """The Feature Picks bubble scatter: log views (x) vs conversion (y), eligible products only.

    ``customdata`` per point is ``[product_key, item_name, category_group, views, purchases,
    cr_raw, cr_adj, revenue_usd, code, quadrant]`` (item_name / category_group HTML-escaped), so
    a selection's ``customdata[0]`` is the product key. ``color_map`` is the manifest's
    ``category_colors``; groups it does not colour fall into gray "Other".
    """
    df = _eligible(products)
    if not df.empty:
        df = df[np.isfinite(_rate(df, y))].reset_index(drop=True)
    if df.empty:
        return empty_figure("No products meet the minimum-views rule in this scope.", height=height)

    n = len(df)
    views = _num(df, "views", 0.0)
    purchases = np.nan_to_num(_num(df, "purchases", 0.0))
    cr_raw, cr_adj, yv = _rate(df, "cr_raw"), _rate(df, "cr_adj"), _rate(df, y)
    groups = _txt(df, "category_group", "Other")
    groups[groups == ""] = "Other"
    tv = float(th.views) if th is not None and _finite(th.views) and th.views > 0 else float(np.median(views))
    tc = float(th.conversion) if th is not None and _finite(th.conversion) else float(purchases.sum() / views.sum())

    quads = _txt(df, "quadrant", "")
    missing = quads == ""
    if missing.any():
        guess = np.select(
            [(views >= tv) & (yv >= tc), (views < tv) & (yv >= tc), (views >= tv) & (yv < tc)],
            ["Winners", "Hidden gems", "Wasted exposure"],
            default="Low priority",
        )
        quads = np.where(missing, guess, quads).astype(object)

    cap = _y_cap(yv, tc)
    over = yv > cap
    y_plot = np.minimum(yv, cap)
    y_range = [-0.05 * cap, cap * Y_HEADROOM]
    lo, hi = math.log10(views.min()), math.log10(views.max())
    lo, hi = min(lo, math.log10(tv)), max(hi, math.log10(tv))
    pad = max(0.12, 0.06 * (hi - lo))
    x_range = [lo - pad, hi + pad]

    raw_size = np.nan_to_num(_num(df, size, 0.0) if size in df.columns else purchases)
    smax = float(raw_size.max()) if n else 0.0
    if smax > 0:
        # A zero size would hide the marker (plotly draws nothing), so lift it to the minimum.
        sizes = np.where(raw_size > 0, raw_size, smax * 1e-6)
        marker_size: dict[str, Any] = {
            "size": sizes,
            "sizemode": "area",
            "sizeref": 2.0 * smax / BUBBLE_MAX_PX**2,
            "sizemin": BUBBLE_MIN_PX / 2,  # plotly compares sizemin with the bubble radius
        }
    else:
        marker_size = {"size": 12}

    conf = _txt(df, "confidence", "")
    opacity = np.where(conf == "uncertain", 0.45, 0.9)
    symbols = np.where(over, "triangle-up", "circle")
    keys = _txt(df, "product_key", "")
    names = _txt(df, "item_name", "")
    names = np.where(names == "", keys, names)
    codes = _txt(df, "code", "")
    revenue = np.nan_to_num(_num(df, "revenue_usd", 0.0))
    ci_lo, ci_hi = _num(df, "ci_low"), _num(df, "ci_high")
    custom = [
        [
            str(keys[i]),
            _esc(names[i]),
            _esc(groups[i]),
            int(round(views[i])),
            int(round(purchases[i])),
            _f(cr_raw[i]),
            _f(cr_adj[i]),
            float(revenue[i]),
            str(codes[i]),
            str(quads[i]),
        ]
        for i in range(n)
    ]
    notes = []
    for i in range(n):
        lines = []
        conf_line = _CONFIDENCE_TEXT.get(str(conf[i]), "")
        if _finite(ci_lo[i]) and _finite(ci_hi[i]):
            conf_line = f"{conf_line} (80% interval {_pct(ci_lo[i])}–{_pct(ci_hi[i])})".strip()
        if codes[i]:
            lines.append(f"Code {codes[i]}")
        if conf_line:
            lines.append(conf_line)
        if over[i]:
            lines.append(f"▲ Off the chart: drawn at {_pct(cap)}, actual {_pct(yv[i])}")
        notes.append("<br>".join(lines) or " ")

    fig = go.Figure()
    series = _colour_series(groups, color_map, highlight)
    for name, colour, mask, rank, dimmed in series:
        idx = np.flatnonzero(mask)
        labels = [str(codes[i]) if label_codes and _top_code(str(codes[i])) else "" for i in idx]
        marker = {
            **{k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in marker_size.items()},
            "color": colour,
            "opacity": np.full(len(idx), 0.35) if dimmed else opacity[idx],
            "symbol": symbols[idx].tolist(),
            "line": {"width": 2, "color": SURFACE},
        }
        fig.add_trace(
            go.Scatter(
                x=views[idx],
                y=y_plot[idx],
                mode="markers+text" if any(labels) else "markers",
                name=_esc(name),
                legendrank=rank,
                marker=marker,
                text=labels if any(labels) else None,
                textposition="top center",
                textfont={"color": INK, "size": 11},
                customdata=[custom[i] for i in idx],
                hovertext=[notes[i] for i in idx],
                hovertemplate=_SCATTER_HOVER,
            )
        )

    x_lo, x_hi = 10 ** x_range[0], 10 ** x_range[1]
    rects = {
        "Hidden gems": (x_lo, tv, tc, y_range[1]),
        "Winners": (tv, x_hi, tc, y_range[1]),
        "Low priority": (x_lo, tv, y_range[0], tc),
        "Wasted exposure": (tv, x_hi, y_range[0], tc),
    }
    for quad, (x0, x1, y0, y1) in rects.items():
        # Shapes take data values on a log axis (annotations would need log10 values).
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
            fillcolor=config.QUADRANT_STYLE[quad]["tint"], opacity=QUADRANT_TINT_OPACITY, line={"width": 0},
        )
    threshold = {"color": INK, "width": 1, "dash": "dash"}
    fig.add_shape(type="line", xref="x", yref="paper", x0=tv, x1=tv, y0=0, y1=1, line=threshold)
    fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=tc, y1=tc, line=threshold)

    corners = {
        "top-left": (0, 1, "left", "top", 6, -6),
        "top-right": (1, 1, "right", "top", -6, -6),
        "bottom-left": (0, 0, "left", "bottom", 6, 6),
        "bottom-right": (1, 0, "right", "bottom", -6, 6),
    }
    for quad in config.QUADRANTS:
        style = config.QUADRANT_STYLE[quad]
        x, yy, xa, ya, dx, dy = corners[style["corner"]]
        fig.add_annotation(
            name=f"badge:{quad}", text=f"<b>{style['badge']}</b>", xref="paper", yref="paper", x=x, y=yy,
            xanchor=xa, yanchor=ya, xshift=dx, yshift=dy, bgcolor=WHITE_BOX, borderpad=3,
            font={"color": INK, "size": 11},
        )
    named_rule = th is not None and th.views_rule in config.VIEWS_RULES and th.views_rule != "median"
    rule = f" ({th.views_rule})" if named_rule else ""
    fig.add_annotation(
        name="threshold:views", text=f"High-traffic cutoff{rule}: {tv:,.0f} views", xref="x", yref="paper",
        x=math.log10(tv), y=1.0, xanchor="center", yanchor="bottom", yshift=2,
    )
    fig.add_annotation(
        name="threshold:conversion", text=f"Average<br>{_pct(tc)}", xref="paper", yref="y", x=1.0, y=tc,
        xanchor="left", yanchor="middle", xshift=6, align="left",
    )
    if over.any():
        k = int(over.sum())
        fig.add_annotation(
            name="overflow", text=f"▲ {k} product{'s' if k != 1 else ''} above {_pct(cap)}, drawn at the top edge",
            xref="x domain", yref="y domain", x=0.5, y=1, xanchor="center", yanchor="top", yshift=-6,
            bgcolor=WHITE_BOX, font={"color": MUTED, "size": 11},
        )

    y_title = "Purchases per product view" + (" (adjusted)" if y == "cr_adj" else "")
    apply_template(
        fig, height=height, showlegend=len(series) >= 2, legend={"itemsizing": "constant", "y": 1.09},
        margin={"t": 84, "r": 84},
    )
    ticks = [(v, t) for v, t in LOG_TICKS if x_range[0] <= math.log10(v) <= x_range[1]]
    fig.update_xaxes(
        type="log", range=x_range, title_text="Product views (log scale)",
        tickvals=[v for v, _ in ticks], ticktext=[t for _, t in ticks], minor={"showgrid": False},
    )
    fig.update_yaxes(range=y_range, tickformat=".2~%", title_text=y_title)
    return fig


def product_timeline(daily_one: pd.DataFrame | None, periods: Any = None, *, height: int = 380) -> go.Figure:
    """One product's daily views (top) and purchases (bottom), sharing x; windows shaded."""
    d = _daily_counts(daily_one)
    if d.empty:
        return empty_figure("No daily activity for this product in scope.", height=height)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    fig.add_trace(
        go.Scatter(
            x=d["event_date"], y=d["views"], mode="lines", name="Product views", line={"color": SERIES[0], "width": 2},
            hovertemplate="%{y:,.0f} views<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=d["event_date"], y=d["purchases"], name="Purchases", marker={"color": SERIES[1], "line": {"width": 0}},
            hovertemplate="%{y:,.0f} purchases<extra></extra>",
        ),
        row=2, col=1,
    )
    for r in (1, 2):
        _shade(fig, periods, row=r, col=1, label=r == 1)
    apply_template(fig, height=height, hovermode="x unified", bargap=0.15, barcornerradius=2, margin={"t": 40})
    fig.update_xaxes(hoverformat="%a %b %d, %Y")
    fig.update_yaxes(title_text="Views", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Purchases", rangemode="tozero", row=2, col=1)
    return fig


# --------------------------------------------------------------------------- Holiday Shift


def period_bars(
    period_df: pd.DataFrame | None, *, colors: Sequence[str] = config.PERIOD_COLORS, height: int = 420
) -> go.Figure:
    """Grouped bars of conversion per group (x) and window, with 95% intervals.

    Takes ``metrics.period_comparison`` output. Bars sit at explicit numeric positions so the
    lift labels (only where ``p_value < 0.05``) land exactly above their bar.
    """
    if not _has(period_df) or "period" not in period_df.columns:
        return empty_figure("No window comparison for this scope.", height=height)
    df = period_df.reset_index(drop=True)
    gcol = "group" if "group" in df.columns else ("category_group" if "category_group" in df.columns else None)
    df = df.assign(_g=_txt(df, gcol, "All products") if gcol else "All products", _p=_txt(df, "period"))
    periods = list(dict.fromkeys(df["_p"]))
    groups = list(dict.fromkeys(df["_g"]))
    labels = {}
    for p in periods:
        lab = _txt(df[df["_p"] == p], "period_label", "")
        labels[p] = lab[0] if len(lab) and lab[0] else (config.PERIODS[p].label if p in config.PERIODS else p)
    base = periods[0]
    if "is_base" in df.columns:
        flagged = df.loc[df["is_base"].astype(object).where(df["is_base"].notna(), False).astype(bool), "_p"]
        base = flagged.iloc[0] if len(flagged) else base

    n = len(periods)
    band = 0.62
    slot = band / n
    width = slot * 0.84  # leaves a small surface gap between neighbouring bars
    fig = go.Figure()
    y_top = 0.0
    for i, (p, colour) in enumerate(zip(periods, _ordinal(n, colors))):
        sub = df[df["_p"] == p].drop_duplicates("_g").set_index("_g").reindex(groups)
        cr, lo, hi = _num(sub, "cr"), _num(sub, "ci_low"), _num(sub, "ci_high")
        views, purch = _num(sub, "views", 0.0), _num(sub, "purchases", 0.0)
        lift, pval = _num(sub, "lift"), _num(sub, "p_value")
        low = _txt(sub, "low_sample", "False")
        x = np.arange(len(groups)) - band / 2 + (i + 0.5) * slot
        short = f" · {config.PERIODS[p].short()}" if p in config.PERIODS else ""
        hover = []
        for j, g in enumerate(groups):
            lines = [
                f"<b>{_esc(g)}</b>",
                _esc(labels[p]),
                f"Conversion: {_pct(cr[j])} (95% interval {_pct(lo[j])}–{_pct(hi[j])})",
            ]
            if _finite(views[j]):
                lines.append(f"{views[j]:,.0f} views · {np.nan_to_num(purch[j]):,.0f} purchases")
            if p == base:
                lines.append("Base window")
            elif low[j].lower() == "true":
                lines.append(f"Lift vs base: {lift[j]:+.0%} (low sample, not tested)" if _finite(lift[j]) else "Low sample")
            elif _finite(lift[j]):
                lines.append(f"Lift vs base: {lift[j]:+.0%}" + (f" · p = {pval[j]:.3f}" if _finite(pval[j]) else ""))
            hover.append("<br>".join(lines))
        err_hi = np.nan_to_num(np.where(np.isfinite(hi) & np.isfinite(cr), hi - cr, 0.0)).clip(min=0)
        err_lo = np.nan_to_num(np.where(np.isfinite(lo) & np.isfinite(cr), cr - lo, 0.0)).clip(min=0)
        fig.add_trace(
            go.Bar(
                x=x, y=cr, width=width, name=_esc(labels[p] + short), marker={"color": colour, "line": {"width": 0}},
                error_y={"type": "data", "symmetric": False, "array": err_hi, "arrayminus": err_lo,
                         "color": INK_2, "thickness": 1, "width": 3},
                hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
            )
        )
        tops = np.where(np.isfinite(hi), hi, cr)
        if np.isfinite(tops).any():
            y_top = max(y_top, float(np.nanmax(tops)))
        if p == base:
            continue
        for j in range(len(groups)):
            if _finite(pval[j]) and pval[j] < 0.05 and _finite(lift[j]) and _finite(tops[j]):
                fig.add_annotation(
                    name="lift", text=f"<b>{lift[j]:+.0%}</b>", x=x[j], y=tops[j], xref="x", yref="y",
                    yanchor="bottom", yshift=3, font={"color": INK, "size": 11},
                )
    apply_template(fig, height=height, barmode="overlay", barcornerradius=4, showlegend=n >= 2, hovermode="closest")
    fig.update_xaxes(
        tickvals=list(range(len(groups))), ticktext=[_esc(g) for g in groups], range=[-0.5, len(groups) - 0.5],
        showgrid=False, ticks="",
    )
    fig.update_yaxes(tickformat=".2~%", title_text="Purchases per product view", range=[0, max(y_top, 1e-6) * 1.18])
    return fig


def daily_conversion(
    daily: pd.DataFrame | None,
    periods: Any = None,
    *,
    y_title: str = "Purchases per product view",
    height: int = 380,
) -> go.Figure:
    """Daily conversion line (one axis) with the comparison windows shaded and labelled.

    ``daily`` may be item rows (``event_date, views, purchases``), ``metrics.daily_totals``
    output, or ``site_daily`` (``view_item`` / ``purchase`` events, long or wide).
    """
    d = _daily_counts(daily)
    if d.empty or not (d["views"] > 0).any():
        return empty_figure("No daily traffic in this scope.", height=height)
    v, p = d["views"].to_numpy(dtype=float), d["purchases"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = np.where(v > 0, p / v, np.nan)
    fig = go.Figure(
        go.Scatter(
            x=d["event_date"], y=cr, mode="lines", name="Daily conversion", connectgaps=False,
            line={"color": SERIES[0], "width": 2}, customdata=np.column_stack([v, p]),
            hovertemplate="Conversion: %{y:.2%}<br>%{customdata[0]:,.0f} views · %{customdata[1]:,.0f} purchases<extra></extra>",
        )
    )
    _shade(fig, periods)
    apply_template(fig, height=height, showlegend=False, hovermode="x unified")
    fig.update_xaxes(hoverformat="%a %b %d, %Y")
    fig.update_yaxes(tickformat=".2~%", rangemode="tozero", title_text=_esc(y_title))
    return fig


def _transition_counts(counts: pd.DataFrame) -> pd.DataFrame:
    quads = list(config.QUADRANTS)
    if {"quadrant_base", "quadrant_focus"} <= set(counts.columns):
        both = counts[counts["quadrant_base"].isin(quads) & counts["quadrant_focus"].isin(quads)]
        m = pd.crosstab(both["quadrant_base"].astype(str), both["quadrant_focus"].astype(str))
    else:
        m = counts
    return m.reindex(index=quads, columns=quads).fillna(0).astype("int64")


def transition_matrix(
    counts: pd.DataFrame | None,
    *,
    base_label: str = "base window",
    focus_label: str = "focus window",
    height: int = 380,
) -> go.Figure:
    """4x4 heatmap of quadrant transitions (rows = base quadrant, columns = focus quadrant).

    Takes ``metrics.transition_counts`` output, or a ``quadrant_movers`` frame (counted here).
    """
    if not _has(counts):
        return empty_figure("No products were eligible in both windows.", height=height)
    m = _transition_counts(counts)
    z = m.to_numpy(dtype=float)
    if z.sum() <= 0:
        return empty_figure("No products were eligible in both windows.", height=height)
    quads = list(config.QUADRANTS)
    zmax = max(float(z.max()), 1.0)
    hover = [
        [
            (f"Stayed {qb}" if qb == qf else f"{qb} → {qf}") + f": {int(z[i, j]):,} product{'s' if z[i, j] != 1 else ''}"
            for j, qf in enumerate(quads)
        ]
        for i, qb in enumerate(quads)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=quads, y=quads, colorscale=_SEQ_NAVY, zmin=0, zmax=zmax, showscale=False, xgap=2, ygap=2,
            text=hover, hovertemplate="%{text}<extra></extra>",
        )
    )
    for i, qb in enumerate(quads):
        for j, qf in enumerate(quads):
            fig.add_annotation(
                text=f"{int(z[i, j]):,}", x=qf, y=qb, xref="x", yref="y",
                font={"size": 12, "color": SURFACE if z[i, j] / zmax > 0.5 else INK},
            )
    apply_template(fig, height=height, showlegend=False)
    _no_grid(fig)
    fig.update_xaxes(title_text=f"Quadrant in the {_esc(focus_label)}", side="bottom")
    fig.update_yaxes(title_text=f"Quadrant in the {_esc(base_label)}", autorange="reversed")
    return fig


# --------------------------------------------------------------------------- Device Split


def device_heatmap(
    matrix: pd.DataFrame | None,
    lean: pd.DataFrame | None = None,
    *,
    text_col: str = "cr_raw",
    max_label: int = 38,
    height: int | None = None,
) -> go.Figure:
    """Products (rows, best overall ``cr_adj`` first) x devices.

    Takes ``metrics.device_matrix`` and optional ``metrics.device_lean`` output. Colour answers
    "does it differ by device?": ``log2((cr_adj_d / device_rate) / (cr_adj / site_rate))``, the
    product's device index relative to its own all-device index, clipped to +-1 (gray = as usual,
    navy = over-performs on this device, coral = under-performs); cell text is
    ``text_col`` as a percentage; the low-volume tablet column is faded; the lean is listed on
    the right.
    """
    dev_col = None
    if _has(matrix) and "product_key" in matrix.columns:
        dev_col = next((c for c in ("device", "device_category") if c in matrix.columns), None)
    if dev_col is None:
        return empty_figure("Pick products to compare across devices.", height=height or 320)
    df = matrix.assign(_k=_txt(matrix, "product_key"), _d=_txt(matrix, dev_col))
    present = list(dict.fromkeys(df["_d"]))
    devices = [d for d in config.DEVICES if d in present] + sorted(d for d in present if d not in config.DEVICES)
    keys = list(dict.fromkeys(df["_k"]))

    totals = pd.DataFrame(
        {"k": df["_k"], "v": np.nan_to_num(_num(df, "views", 0.0)), "p": np.nan_to_num(_num(df, "purchases", 0.0))}
    ).groupby("k", sort=False)[["v", "p"]].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = (totals["p"] / totals["v"]).where(totals["v"] > 0)
    overall = pd.Series(_num(df, "cr_adj"), index=df["_k"]).groupby(level=0, sort=False).first()
    score = overall.reindex(keys).where(lambda s: s.notna(), pooled.reindex(keys))
    order = sorted(keys, key=lambda k: (-score[k] if _finite(score[k]) else math.inf, k))
    n = len(order)
    cells = df.drop_duplicates(["_k", "_d"]).set_index(["_k", "_d"])

    def grid(col: str) -> np.ndarray:
        if col not in cells.columns:
            return np.full((n, len(devices)), np.nan)
        wide = pd.to_numeric(cells[col], errors="coerce").unstack("_d")
        return wide.reindex(index=order, columns=devices).to_numpy(dtype=float)

    index, views, purch = grid("index"), grid("views"), grid("purchases")
    raw, adj, rate = grid("cr_raw"), grid("cr_adj_d"), grid("device_rate")
    v_all, p_all = np.nan_to_num(_num(df, "views", 0.0)), np.nan_to_num(_num(df, "purchases", 0.0))
    site_rate = float(p_all.sum() / v_all.sum()) if v_all.sum() > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        # Device index: the product's adjusted device rate over the device average.
        dev_tot = pd.DataFrame({"d": df["_d"], "v": v_all, "p": p_all}).groupby("d")[["v", "p"]].sum()
        dev_rate = (dev_tot["p"] / dev_tot["v"]).where(dev_tot["v"] > 0).reindex(devices).to_numpy(dtype=float)
        dev_index = np.where(np.isfinite(index), index, adj / rate)
        dev_index = np.where(np.isfinite(dev_index), dev_index, raw / dev_rate[None, :])
        # The product's usual (all-device) index: cr_adj over the site rate, else pooled counts.
        p_bar = _num(df, "p_bar")
        p_bar = float(np.nanmedian(p_bar)) if np.isfinite(p_bar).any() else site_rate
        usual = (overall.reindex(order) / p_bar).to_numpy(dtype=float)
        usual = np.where(np.isfinite(usual), usual, (pooled.reindex(order) / site_rate).to_numpy(dtype=float))
        rel = dev_index / usual[:, None]
        rel = np.where(rel > 0, rel, np.nan)
        z = np.clip(np.log2(rel), -1.0, 1.0)
    shown = grid(text_col)
    name_of = dict(zip(df["_k"], _txt(df, "item_name", "")))
    names = [_short(name_of.get(k) or k, max_label) for k in order]

    hover = np.empty((n, len(devices)), dtype=object)
    for i in range(n):
        for j, dev in enumerate(devices):
            lines = [f"<b>{_esc(names[i])}</b> · {_esc(dev)}"]
            if _finite(views[i, j]):
                lines.append(f"{views[i, j]:,.0f} views · {np.nan_to_num(purch[i, j]):,.0f} purchases")
            lines.append(f"Device conversion: {_pct(raw[i, j])}")
            lines.append(f"{rel[i, j]:.2f}× vs its usual" if _finite(rel[i, j]) else "vs its usual: –")
            hover[i, j] = "<br>".join(lines)

    fig = go.Figure()
    colorbar = {
        "orientation": "h", "x": 0.5, "xanchor": "center", "yref": "container", "y": 0, "yanchor": "bottom",
        "ypad": 8, "len": 0.5, "thickness": 10, "outlinewidth": 0,
        "tickvals": [-1, -0.5, 0, 0.5, 1], "ticktext": ["0.5×", "0.7×", "1×", "1.4×", "2×"],
        "tickfont": {"color": MUTED, "size": 11},
        "title": {"text": "Device vs the product's usual (×)", "side": "top", "font": {"color": INK_2, "size": 11}},
    }
    # Numeric x positions: one heatmap column per device, nothing else.
    parts = [[j for j, d in enumerate(devices) if d != "tablet"], [j for j, d in enumerate(devices) if d == "tablet"]]
    first = True
    for cols_idx, faded in zip(parts, (False, True)):
        if not cols_idx:
            continue
        fig.add_trace(
            go.Heatmap(
                z=z[:, cols_idx], x=cols_idx, y=list(range(n)),
                text=hover[:, cols_idx].tolist(), hovertemplate="%{text}<extra></extra>",
                colorscale=_DIVERGING_SCALE, zmin=-1, zmax=1, xgap=2, ygap=2,
                opacity=0.5 if faded else 1.0, showscale=first, colorbar=colorbar if first else None,
                name="low volume" if faded else "devices",
            )
        )
        first = False
    for i in range(n):
        for j, dev in enumerate(devices):
            if dev == "tablet":
                colour = INK_2
            else:
                colour = SURFACE if _finite(z[i, j]) and z[i, j] >= 0.55 else INK
            fig.add_annotation(
                text=_pct(shown[i, j]), x=j, y=i, xref="x", yref="y", font={"size": 11, "color": colour}
            )

    has_lean = _has(lean) and {"product_key", "lean"} <= set(lean.columns)
    if has_lean:
        lean_of = dict(zip(_txt(lean, "product_key"), _txt(lean, "lean")))
        fig.add_annotation(
            text="<b>Lean</b> (mobile vs desktop)", xref="paper", yref="paper", x=1, y=1, xanchor="left",
            yanchor="bottom", xshift=10, font={"color": INK_2, "size": 11},
        )
        for i, k in enumerate(order):
            call = lean_of.get(k, "")
            if call:
                fig.add_annotation(
                    name="lean", text=f"<b>{_esc(call)}</b>" if call in _LEAN_STRONG else _esc(call),
                    xref="paper", yref="y", x=1, y=i, xanchor="left", xshift=10,
                    font={"color": INK if call in _LEAN_STRONG else INK_2, "size": 11},
                )
    low_volume = "<br><span style='font-size:10px'>low volume</span>"
    ticktext = [_esc(d) + (low_volume if d == "tablet" else "") for d in devices]
    apply_template(
        fig, height=height or max(320, 30 * n + 170), showlegend=False,
        margin={"t": 64, "b": 84, "r": 170 if has_lean else 24},
    )
    _no_grid(fig)
    fig.update_xaxes(side="top", type="linear", range=[-0.5, len(devices) - 0.5], tickvals=list(range(len(devices))),
                     ticktext=ticktext, tickfont={"color": INK_2})
    fig.update_yaxes(tickvals=list(range(n)), ticktext=[_esc(s) for s in names], autorange="reversed",
                     tickfont={"color": INK_2})
    return fig


# --------------------------------------------------------------------------- Outside Signal


def trends_small_multiples(
    index_df: pd.DataFrame | None, periods: Any = None, *, cols: int = 3, height: int | None = None
) -> go.Figure:
    """One panel per category group: Google interest and on-site views, both indexed to
    baseline = 100 on the panel's single y-axis (``phase2.trend_index`` output).

    Groups with several keywords show the mean of their keyword indices.
    """
    if not _has(index_df) or not {"group", "week"} <= set(index_df.columns):
        return empty_figure("No Google Trends data for this scope.", height=height or 320)
    df = index_df
    if "group_type" in df.columns and (_txt(df, "group_type") == "category").any():
        df = df[_txt(df, "group_type") == "category"]
    frame = pd.DataFrame(
        {
            "group": _txt(df, "group"),
            "week": _dates(df["week"]).to_numpy(),
            "g": _num(df, "google_index"),
            "s": _num(df, "site_views_index"),
        }
    ).dropna(subset=["week"])
    agg = frame.groupby(["group", "week"], as_index=False).agg(g=("g", "mean"), s=("s", "mean"), n=("g", "count"))
    agg = agg[agg["g"].notna() | agg["s"].notna()]
    if agg.empty:
        return empty_figure("No Google Trends data for this scope.", height=height or 320)
    present = list(dict.fromkeys(agg["group"]))
    known = config.CATEGORY_GROUPS
    groups = [g for g in known if g in present] + sorted(g for g in present if g not in known)
    ncols = max(1, min(cols, len(groups)))
    nrows = math.ceil(len(groups) / ncols)
    fig = make_subplots(
        rows=nrows, cols=ncols, shared_xaxes="all", subplot_titles=[_esc(g) for g in groups],
        horizontal_spacing=0.06, vertical_spacing=min(0.16, 0.9 / (nrows - 1)) if nrows > 1 else 0.0,
    )
    fig.update_annotations(font={"size": 12, "color": INK})  # subplot titles
    for k, group in enumerate(groups):
        r, c = k // ncols + 1, k % ncols + 1
        g = agg[agg["group"] == group].sort_values("week")
        n_kw = int(g["n"].max()) if len(g) else 1
        google_name = "Google interest index" + (f" (mean of {n_kw} keywords)" if n_kw > 1 and k == 0 else "")
        lines = (
            ("g", google_name, "Google interest", SERIES[0], "google"),
            ("s", "On-site views index", "On-site views", SERIES[1], "site"),
        )
        for col, name, short, colour, lg in lines:
            fig.add_trace(
                go.Scatter(
                    x=g["week"], y=g[col], mode="lines+markers", name=name, legendgroup=lg, showlegend=k == 0,
                    line={"color": colour, "width": 2},
                    marker={"size": 8, "color": colour, "line": {"width": 2, "color": SURFACE}},
                    hovertemplate=f"{_esc(group)} · {short}: %{{y:.0f}}<extra></extra>",
                ),
                row=r, col=c,
            )
        fig.add_hline(y=100, line={"color": AXIS, "width": 1}, row=r, col=c)
        _shade(fig, periods, row=r, col=c, label=k == 0)
    # Container-anchored legend: a paper-anchored one would sit on the first row's panel titles.
    apply_template(
        fig, height=height or 240 * nrows + 100, hovermode="x unified", margin={"t": 76},
        legend={"yref": "container", "y": 0.995, "yanchor": "top"},
    )
    fig.update_xaxes(showticklabels=True, hoverformat="week ending %b %d")
    fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(title_text="Index (baseline = 100)", col=1)
    return fig


# --------------------------------------------------------------------------- Cannibalization


def spike_share(spikes: pd.DataFrame | None, key: str | None, periods: Any = None, *, height: int = 360) -> go.Figure:
    """Product A's daily share of its category's views, with spike days marked
    (``phase2.exposure_spikes`` output)."""
    if not _has(spikes) or "product_key" not in spikes.columns or key is None:
        return empty_figure("No daily share data for this product.", height=height)
    d = spikes[_txt(spikes, "product_key") == str(key)]
    if d.empty or "event_date" not in d.columns:
        return empty_figure("No daily share data for this product.", height=height)
    d = d.assign(_date=_dates(d["event_date"]).to_numpy()).sort_values("_date")
    share = _num(d, "share")
    if not np.isfinite(share).any():
        return empty_figure("No daily share data for this product.", height=height)
    expo = _num(d, "exposure") if "exposure" in d.columns else _num(d, "views")
    cat = _num(d, "category_exposure") if "category_exposure" in d.columns else _num(d, "category_views")
    ratio = _num(d, "share_ratio")
    custom = np.column_stack([np.nan_to_num(expo), np.nan_to_num(cat), ratio])
    hover = (
        "%{y:.1%} of category views<br>"
        "%{customdata[0]:,.0f} of %{customdata[1]:,.0f} views · %{customdata[2]:.1f}× its median<extra></extra>"
    )
    spike = _txt(d, "is_spike", "False") == "True"
    fig = go.Figure(
        go.Scatter(
            x=d["_date"], y=share, mode="lines", name="Share of category views", line={"color": SERIES[0], "width": 2},
            customdata=custom, hovertemplate=hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["_date"][spike], y=share[spike], mode="markers", name="Spike day",
            marker={"size": 9, "color": SERIES[1], "line": {"width": 2, "color": SURFACE}},
            customdata=custom[spike], hovertemplate="<b>Spike day</b><br>" + hover,
        )
    )
    med = _num(d, "median_share")
    if np.isfinite(med).any():
        m = float(np.nanmedian(med))
        fig.add_hline(
            y=m, line={"color": MUTED, "width": 1}, annotation_text=f"Typical day (median share): {_pct(m, 1)}",
            annotation_position="top right", annotation_font_size=11, annotation_font_color=INK_2,
        )
    _shade(fig, periods)
    apply_template(fig, height=height, hovermode="x unified", showlegend=True)
    fig.update_xaxes(hoverformat="%a %b %d, %Y")
    fig.update_yaxes(tickformat=".1~%", rangemode="tozero", title_text="Share of category views")
    return fig


def rr_dotplot(tests: pd.DataFrame | None, *, max_label: int = 40, height: int | None = None) -> go.Figure:
    """Relative risk of B's purchases on A's spike days, with 95% whiskers, on a log axis
    (``phase2.cannibalization_tests`` output). Status sets colour and marker and is always
    written out, so colour never carries it alone."""
    if not _has(tests) or "rr" not in tests.columns:
        return empty_figure("No product pairs had enough data to test.", height=height or 320)
    t = tests
    if "tested" in t.columns:
        t = t[t["tested"].astype(object).where(t["tested"].notna(), False).astype(bool).to_numpy()]
    t = t[np.isfinite(_num(t, "rr"))]
    if t.empty:
        return empty_figure("No product pairs had enough data to test.", height=height or 320)
    t = t.assign(_rr=_num(t, "rr")).sort_values(["_rr"], kind="mergesort").reset_index(drop=True)
    n = len(t)
    rr = t["_rr"].to_numpy(dtype=float)
    floor, ceil = RR_CLIP
    rr_c = np.clip(rr, floor, ceil)
    rl, rh = _num(t, "rr_low"), _num(t, "rr_high")  # rr_high is inf when B sold nothing on other days
    lo = np.clip(np.where(np.isnan(rl), rr_c, rl), floor, rr_c)
    hi = np.clip(np.where(np.isnan(rh), rr_c, rh), rr_c, ceil)
    a_names, b_names = _txt(t, "item_name_a"), _txt(t, "item_name_b")
    a_names = np.where(a_names == "", _txt(t, "product_a"), a_names)
    b_names = np.where(b_names == "", _txt(t, "product_b"), b_names)
    status = _txt(t, "status", "No evidence")
    status[status == ""] = "No evidence"
    many_a = len(set(a_names)) > 1
    labels = [_short(f"{a} → {b}" if many_a else b, max_label) for a, b in zip(a_names, b_names)]

    spike_days, o_s, e_s = _num(t, "n_spike_days"), _num(t, "O_s"), _num(t, "E_s")
    tests_p = [(lab, _num(t, col)) for col, lab in (("p_value", "p"), ("q_value", "q"), ("fisher_p", "Fisher p"))]
    hover = []
    for i in range(n):
        lines = [f"<b>{_esc(b_names[i])}</b>", f"When {_esc(a_names[i])} spikes"]
        if _finite(spike_days[i]):
            lines[-1] += f" ({spike_days[i]:,.0f} days)"
        lines.append(f"Status: <b>{_esc(status[i])}</b>")
        hi_txt = "∞" if np.isposinf(rh[i]) else (f"{rh[i]:.2f}" if _finite(rh[i]) else "–")
        lo_txt = f"{rl[i]:.2f}" if _finite(rl[i]) else "–"
        lines.append(f"Relative risk {rr[i]:.2f} (95% interval {lo_txt}–{hi_txt})")
        if _finite(o_s[i]) and _finite(e_s[i]):
            lines.append(f"Spike days: {o_s[i]:,.0f} purchases vs {e_s[i]:,.1f} expected")
        stats_txt = [f"{lab} = {vals[i]:.3f}" for lab, vals in tests_p if _finite(vals[i])]
        if stats_txt:
            lines.append(" · ".join(stats_txt))
        if rr[i] != rr_c[i]:
            lines.append(f"Drawn at the axis edge ({rr_c[i]:g}×)")
        hover.append("<br>".join(lines))

    fig = go.Figure()
    order = list(_STATUS_STYLE) + sorted(set(status) - set(_STATUS_STYLE))
    for rank, st in enumerate(order):
        idx = np.flatnonzero(status == st)
        if not len(idx):
            continue
        colour, symbol = _STATUS_STYLE.get(st, (OTHER, "circle"))
        fig.add_trace(
            go.Scatter(
                x=rr_c[idx], y=idx, mode="markers", name=_esc(st), legendrank=rank,
                marker={"size": 11, "color": colour, "symbol": symbol, "line": {"width": 2, "color": SURFACE}},
                error_x={"type": "data", "symmetric": False, "array": hi[idx] - rr_c[idx], "arrayminus": rr_c[idx] - lo[idx],
                         "color": colour, "thickness": 1.5, "width": 0},
                hovertext=[hover[i] for i in idx], hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    fig.add_shape(type="line", xref="x", yref="paper", x0=1, x1=1, y0=0, y1=1, line={"color": INK, "width": 1})
    fig.add_annotation(
        text="<b>Status</b>", xref="paper", yref="paper", x=1, y=1, xanchor="left", yanchor="bottom", xshift=10
    )
    for i in range(n):
        strong = status[i] in ("Likely cannibalized", "Possible")
        fig.add_annotation(
            name="status", text=f"<b>{_esc(status[i])}</b>" if strong else _esc(status[i]), xref="paper", yref="y",
            x=1, y=i, xanchor="left", xshift=10, font={"color": INK if strong else INK_2, "size": 11},
        )
    x_lo = math.log10(min(float(lo.min()), 0.8) / 1.25)
    x_hi = math.log10(max(float(hi.max()), 1.25) * 1.25)
    ticks = [v for v in (0.05, 0.1, 0.25, 0.5, 0.8, 1, 1.25, 2, 4, 10, 20) if x_lo <= math.log10(v) <= x_hi]
    apply_template(
        fig, height=height or max(300, 30 * n + 150), showlegend=True, hovermode="closest",
        margin={"r": 170, "t": 56},
    )
    fig.update_xaxes(
        type="log", range=[x_lo, x_hi], tickvals=ticks,
        ticktext=[f"{v:g}×" + (" (no change)" if v == 1 else "") for v in ticks],
        title_text="Relative risk of B's purchases on A's spike days (log scale)",
    )
    fig.update_yaxes(
        tickvals=list(range(n)), ticktext=[_esc(s) for s in labels], autorange="reversed", showgrid=False,
        tickfont={"color": INK_2},
    )
    return fig

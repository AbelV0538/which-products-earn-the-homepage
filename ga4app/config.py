"""Central configuration shared by the pipeline scripts, the analysis modules and the app.

Everything that more than one module needs to agree on lives here: paths, the BigQuery
source, season windows, placeholder tokens, thresholds, the validated colour palette and
the OpenRouter model list. Values that depend on the real data (product key, view metric,
category mapping) are decided by profiling and recorded in ``data/profile.json`` /
``data/manifest.json`` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_ENV = "GA4APP_DATA_DIR"
LLM_FAKE_ENV = "GA4APP_LLM_FAKE"
GCP_PROJECT_ENV = "GA4APP_GCP_PROJECT"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
RAW_SUBDIR = "raw"
TRENDS_RAW_SUBDIR = "raw/trends"
TRENDS_MANUAL_SUBDIR = "trends_manual"
SQL_DIR = PROJECT_ROOT / "sql"
CONFIG_DIR = PROJECT_ROOT / "config"
LOCAL_CONFIG = CONFIG_DIR / "local.json"  # git-ignored: remembers the GCP project id
TRENDS_KEYWORDS_CSV = CONFIG_DIR / "trends_keywords.csv"
CSV_DEFAULT = PROJECT_ROOT / "ga4_events_export.csv"
SECRETS_TOML = PROJECT_ROOT / ".streamlit" / "secrets.toml"


def resolve_data_dir() -> Path:
    """Snapshot directory: ``$GA4APP_DATA_DIR`` if set (tests, synthetic runs), else ``data/``."""
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    return Path(override).resolve() if override else DEFAULT_DATA_DIR


# --------------------------------------------------------------------------- BigQuery source

BQ_TABLE = "bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*"
BQ_SUFFIX_START = "20201101"
BQ_SUFFIX_END = "20210131"
BQ_LOCATION = "US"
BQ_SCOPES = ("https://www.googleapis.com/auth/bigquery",)
BQ_SHARDS = 92
BQ_MIN_BILL_PER_TABLE = 10 * 1024**2  # on-demand billing floor: 10 MiB per referenced table
BQ_MIN_BILL_PER_QUERY = 10 * 1024**2  # ... plus 10 MiB per query
BQ_JOB_LABELS = {"app": "itcs5122-team9"}

# The pitch's proof query must reproduce these numbers (slide 5).
PROOF_EXPECTED = {"event_count": 4_295_584, "user_count": 270_154, "day_count": 92}

# --------------------------------------------------------------------------- data shape

DATA_START = date(2020, 11, 1)
DATA_END = date(2021, 1, 31)  # raw extent of the export (the 92 days of the proof query)
# Last day with item-level purchases. From Jan 26, 2021 the sample's purchase events stop listing
# their items: site purchase events stay normal (19-69 a day) while item-level purchases fall to
# 53, 12, 10, 16, 11, 0. Through Jan 25 every day has at least 1.13 item rows per purchase event;
# after it, under 1.0 (scripts/verify_snapshots.py checks this). The app drops item rows after
# this date, so those days are never read as days nobody bought anything.
ITEM_DATA_END = date(2021, 1, 25)
DEVICES = ("desktop", "mobile", "tablet")
FUNNEL_EVENTS = ("view_item", "add_to_cart", "begin_checkout", "purchase")
# GA4 obfuscation placeholders. The SQL files spell this list out literally; a unit test
# asserts both stay identical.
PLACEHOLDERS = ("", "(not set)", "<Other>", "(data deleted)", "Not available in demo dataset")
# add_to_cart / begin_checkout are not tracked consistently before this date (verified in the
# CSV: ~0 add_to_cart events on Nov 1-15 and Nov 21-24). Never compare cart metrics across it.
CART_TRACKING_START = date(2020, 11, 26)


@dataclass(frozen=True)
class Period:
    """An inclusive date window."""

    key: str
    label: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def short(self) -> str:
        return f"{self.start:%b %d} – {self.end:%b %d}".replace(" 0", " ")


# Windows come from the data: purchases surge on Mon Nov 23 (before Thanksgiving), peak
# Dec 9-16 and collapse after the Dec 18 shipping cutoff. Only ~3 pre-holiday weeks exist
# (data starts Nov 1), so a January window (Jan 4-25, the same 22 days) is offered as the
# alternative baseline; item-level data ends Jan 25 (ITEM_DATA_END). "full" is the analysis
# season; "export" is the raw 92-day extent, used only for site-level counts.
PERIODS: dict[str, Period] = {
    p.key: p
    for p in (
        Period("full", "Full season", DATA_START, ITEM_DATA_END),
        Period("baseline", "Pre-holiday baseline", date(2020, 11, 1), date(2020, 11, 22)),
        Period("cyber_week", "Cyber Week", date(2020, 11, 23), date(2020, 11, 30)),
        Period("black_friday", "Black Friday weekend", date(2020, 11, 26), date(2020, 11, 29)),
        Period("cyber_monday", "Cyber Monday", date(2020, 11, 30), date(2020, 12, 1)),
        Period("december", "December gifting push", date(2020, 12, 2), date(2020, 12, 18)),
        Period("post_cutoff", "After shipping cutoff", date(2020, 12, 19), date(2020, 12, 31)),
        Period("january", "January", date(2021, 1, 1), ITEM_DATA_END),
        # Key kept from the earlier 4-week version (Jan 4-31) so saved comparisons still resolve.
        Period("january_4wk", "January baseline", date(2021, 1, 4), ITEM_DATA_END),
        Period("export", "Whole export (site counts only)", DATA_START, DATA_END),
    )
}
# Sidebar season presets (order = display order).
PRESETS = ("full", "baseline", "cyber_week", "black_friday", "cyber_monday", "december", "january")
# Holiday Shift comparisons: (base, focus); the first is the default.
COMPARISONS = (
    ("baseline", "cyber_week"),
    ("baseline", "black_friday"),
    ("baseline", "cyber_monday"),
    ("baseline", "december"),
    ("cyber_week", "december"),
    ("january_4wk", "cyber_week"),
    ("january_4wk", "december"),
)
# The decision's three slot campaigns -> evidence window.
CAMPAIGNS = {
    "Black Friday": "black_friday",
    "Cyber Monday": "cyber_monday",
    "December gifting": "december",
}
# Google Trends window read for each campaign (weekly Trends data cannot isolate a 2-4 day window).
CAMPAIGN_TREND_FOCUS = {"Black Friday": "cyber_week", "Cyber Monday": "cyber_week", "December gifting": "december"}

# --------------------------------------------------------------------------- thresholds

MIN_VIEWS_OPTIONS = (0, 25, 50, 100, 200, 300, 500, 1000, 2000, 5000)
DEFAULT_MIN_VIEWS = 200
VIEWS_RULES = ("median", "p75", "p90")  # high-traffic cutoff rule for the quadrant x-line
SHRINK_M_BOUNDS = (100.0, 20_000.0)  # clamp for the empirical-Bayes prior strength m
SHRINK_MIN_VIEWS = 100  # products used to estimate m
CONF_INTERVAL = 0.80  # posterior interval used for the above/below/uncertain call
TOP_N_CODES = 5  # W1..W5, G1..G5, X1..X5 citation codes
DEFAULT_SLOTS = 4  # homepage slots per campaign
# A product whose checkout events number at least this share of its product views is flagged:
# shoppers reached it some other way than a product page (the site average is ~1 checkout per
# 35 views), so its views are probably under-counted and its conversion overstated.
UNDERCOUNT_RATIO = 0.5
CATEGORY_UPLIFT_MIN_VIEWS = 2000  # below this a category borrows the site-wide uplift
DEVICE_LEAN = {"rr_hi": 1.25, "rr_lo": 0.80, "alpha": 0.10, "min_purchases": 10}
CANNIBAL = {
    "top_k": 12,
    "spike_ratio": 2.0,
    "spike_ratio_options": (1.5, 2.0, 3.0),
    "min_daily_views": 30,
    "min_spike_days": 3,
    "min_other_days": 10,
    "min_purchases": 10,
    "min_expected": 3.0,
    "min_spike_views": 200,
    "rr_flag": 0.80,
    "q_flag": 0.10,
    "p_possible": 0.05,
}
TRENDING = {"ext_min": 1.20, "gap_min": 1.15, "min_base_views": 1000, "min_base_interest": 5.0}

# --------------------------------------------------------------------------- categories

CATEGORY_GROUPS = (
    "Apparel",
    "Drinkware",
    "Bags",
    "Office & Stationery",
    "Electronics & Accessories",
    "Lifestyle & Gifts",
    "Other",
)
BRANDS = ("Google", "YouTube", "Android", "Other")

# --------------------------------------------------------------------------- palette
# Validated with the dataviz skill's validator against a #FFFFFF surface.

PALETTE = {
    "navy": "#2F3B7D",  # brand primary: text, badges, ramp ends (too dark for markers)
    "coral": "#F7646A",  # brand accent: tints and diverging pole only, never text (3.0:1)
    "yellow": "#F8E79A",  # brand pale yellow: Hidden-gems tint
    "lavender": "#ECEEF8",  # cards / sidebar
    "ink": "#1F2433",
    "ink_secondary": "#52514E",
    "muted": "#898781",
    "grid": "#E1E0D9",
    "axis": "#C3C2B7",
    "surface": "#FFFFFF",
    "other": "#8A8F98",  # neutral "Other" fold for categorical charts
}
# Scatter / bubble charts: all-pairs CVD check caps the series at three hues.
SCATTER_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
HIGHLIGHT_COLOR = "#2a78d6"
PERIOD_COLORS = ("#9AA6D9", "#6270BD", "#2F3B7D")  # ordinal ramp (passes --ordinal)
DIVERGING = ("#F7646A", "#F0EFEC", "#2F3B7D")  # under-index (coral) · neutral · over-index (navy)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

QUADRANTS = ("Winners", "Hidden gems", "Wasted exposure", "Low priority")
QUADRANT_STYLE = {
    "Hidden gems": {"tint": "#F8E79A", "badge": "HIDDEN GEMS", "corner": "top-left"},
    "Winners": {"tint": "#2F3B7D", "badge": "WINNERS", "corner": "top-right"},
    "Low priority": {"tint": "#8A8F98", "badge": "LOW PRIORITY", "corner": "bottom-left"},
    "Wasted exposure": {"tint": "#F7646A", "badge": "WASTED EXPOSURE", "corner": "bottom-right"},
}
QUADRANT_CODE_PREFIX = {"Winners": "W", "Hidden gems": "G", "Wasted exposure": "X"}

# --------------------------------------------------------------------------- AI narrator

APP_TITLE = "Which Products Earn the Homepage?"
APP_SUBTITLE = "Team 9 · Account D · Google Merchandise Store"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_REFERER = "https://github.com/itcs5122-team9/which-products-earn-the-homepage"
LLM_MODELS = (
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-haiku-5.5",
    "anthropic/claude-fable-5.1",
    "openrouter/free",
    "google/gemma-4-31b-it:free",
)
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
DEFAULT_FALLBACKS = ("anthropic/claude-sonnet-5.5",)
FREE_MODELS = ("openrouter/free", "google/gemma-4-31b-it:free")
LLM_MAX_TOKENS = 4000  # covers reasoning + visible output on OpenRouter
LLM_REASONING_EFFORT = "low"
LLM_TIMEOUT_S = 90.0

# AI chart notes (ga4app.annotate): numbered callouts on the Feature Picks scatter, the Holiday
# Shift bars and the Device Split heatmap.
NOTES_MAX = 4  # notes per chart
NOTE_LABEL_MAX = 40  # characters of the short label drawn on the chart
NOTE_DETAIL_MAX = 200  # characters of the sentence in the list and the hover
HEATMAP_NOTE_MIN_LOG2 = 0.25  # heatmap cells within +-19% of the product's usual are not offered...
HEATMAP_ANCHORS_MAX = 30  # ... and at most this many cells are (the strongest first)
NOTE_ANCHOR_SEP = "·"  # anchor IDs of bars and cells: "Apparel·cyber_week", "G1·mobile"


def is_free_model(model: str) -> bool:
    return model in FREE_MODELS or model.endswith(":free") or model == "openrouter/free"

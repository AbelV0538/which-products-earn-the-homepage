"""Small shared value types, kept separate so every module can import them without cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from . import config


@dataclass(frozen=True)
class Filters:
    """The sidebar scope. Empty ``devices`` / ``categories`` mean "all". The default dates are the
    analysis season (``config.PERIODS["full"]``, ending at ``config.ITEM_DATA_END``)."""

    start: date = config.DATA_START
    end: date = config.ITEM_DATA_END
    devices: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    min_views: int = config.DEFAULT_MIN_VIEWS

    def device_set(self) -> tuple[str, ...]:
        return self.devices or config.DEVICES

    def label(self) -> str:
        dev = ", ".join(self.devices) if self.devices else "all devices"
        cat = ", ".join(self.categories) if self.categories else "all categories"
        return f"{self.start:%b %d, %Y} – {self.end:%b %d, %Y} · {dev} · {cat} · min {self.min_views:,} views"


@dataclass(frozen=True)
class Thresholds:
    """Quadrant split lines: x = views cutoff, y = pooled conversion."""

    views: float
    conversion: float
    views_rule: str = "median"


# Column contract of the per-product frame produced by ``metrics`` and consumed by
# charts, narrative, phase2 and the pages. Extra columns are allowed; these must exist.
PRODUCT_COLUMNS: tuple[str, ...] = (
    "product_key",
    "item_name",
    "category_group",
    "brand",
    "views",
    "purchases",
    "units",
    "revenue_usd",
    "promo_views",
    "cr_raw",  # purchases / views (NaN when views == 0)
    "prior_mean",  # category pooled rate used as the shrinkage prior
    "cr_adj",  # empirical-Bayes rate (the efficiency score)
    "ci_low",  # 80% posterior interval
    "ci_high",
    "confidence",  # "above" | "below" | "uncertain"
    "eligible",  # views >= min_views
    "quadrant",  # one of config.QUADRANTS ("" when not eligible)
    "forgone_purchases",  # views * max(p_bar - cr_adj, 0)
    "lift_per_1k",  # 1000 * max(cr_adj - p_bar, 0)
    "rpv",  # revenue per view
    "action_rank",  # rank within its quadrant's action list (1 = top), NaN otherwise
    "code",  # W1.., G1.., X1.. for the top-N per actionable quadrant, "" otherwise
    "views_share",  # share of in-scope views
    "purchases_exceed_views",  # data-quality flag
    "checkouts",  # checkout events that list the product
    "views_undercount",  # data-quality flag: checkouts >= config.UNDERCOUNT_RATIO * views
)


@dataclass
class Caveats:
    """Plain-language limits repeated on pages and sent to the narrator."""

    items: list[str] = field(
        default_factory=lambda: [
            "No historical homepage-placement data: these are candidates, not proof that featuring causes sales.",
            "No inventory or margin data: we optimise conversion, not profit, and a top converter may have been out of stock.",
            "Add-to-cart and checkout were not tracked consistently before Nov 26, 2020, so only purchase ÷ view is compared across the season.",
            "The GA4 sample is obfuscated by Google; read relative patterns, not exact business totals.",
        ]
    )

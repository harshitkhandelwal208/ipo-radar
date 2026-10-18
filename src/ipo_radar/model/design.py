"""The choices the model package makes, and the rules that map an IPO to a segment and a model set.

Each choice is picked on the pre-2025 walk-forward blocks (model/experiments.py) and written into the saved bundle, so
a bundle says what it was built with. Segments: IN_MAIN, IN_SME, and GLOBAL (US, HK and other markets together,
because GMP exists only for India and the global tables are small).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ipo_radar.schema import MARKETS

SEGMENTS = ("IN_MAIN", "IN_SME", "GLOBAL")
SEGMENT_OF = {"IN_MAIN": "IN_MAIN", "IN_SME": "IN_SME", "US": "GLOBAL", "HK": "GLOBAL", "OTHER": "GLOBAL"}
# Which markets a model set is trained on. 'pooled' is every market; 'india' is both Indian segments.
SET_MARKETS: dict[str, tuple[str, ...]] = {
    "pooled": tuple(MARKETS), "india": ("IN_MAIN", "IN_SME"), "IN_MAIN": ("IN_MAIN",), "IN_SME": ("IN_SME",),
    "GLOBAL": ("US", "HK", "OTHER"),
}
# Model sets a segment may be routed to.
ROUTE_OPTIONS: dict[str, tuple[str, ...]] = {
    "IN_MAIN": ("pooled", "india", "IN_MAIN"), "IN_SME": ("pooled", "india", "IN_SME"),
    "GLOBAL": ("pooled", "GLOBAL"),
}
GMP_MODELS = ("gbm", "linear", "blend")
SELECTION_YEARS = (2022, 2023, 2024)   # walk-forward blocks used to make and check every choice
LOCKED_YEARS = (2025, 2026)            # evaluated once, for the chosen design only
PROTOCOL = ("Choices (conformal, monotone, GMP model, segment routing) were made on walk-forward blocks 2022-2024 "
            "only. Blocks 2025-2026 are locked: evaluated once for the chosen design. Skill flags use 2022-2024. "
            "Serving caveats use the latest blocks (2025-2026) for the 'worse than the historical median' warning only. "
            "This warns more, never less; it does not change the model or any choice.")


def segment_of(market: str | None) -> str | None:
    """IN_MAIN, IN_SME or GLOBAL; None for a market the model does not know."""
    return SEGMENT_OF.get(market or "")


@dataclass
class Design:
    conformal: bool = False            # p10/p90 widened or narrowed by out-of-fold conformal residuals
    monotone: bool = False             # LightGBM monotone increase in subscription and GMP features
    gmp_model: str = "gbm"             # pre_listing only: gbm | linear (quantile regression) | blend (average)
    routing: dict[str, str | None] = field(default_factory=lambda: {s: "pooled" for s in SEGMENTS})

    def __post_init__(self) -> None:
        if self.gmp_model not in GMP_MODELS:
            raise ValueError(f"gmp_model must be one of {GMP_MODELS}, got {self.gmp_model!r}")
        for segment, name in self.routing.items():
            if segment not in SEGMENTS:
                raise ValueError(f"unknown segment {segment!r}")
            if name is not None and name not in ROUTE_OPTIONS[segment]:
                raise ValueError(f"segment {segment} cannot be routed to {name!r}")

    def sets_used(self) -> list[str]:
        """Model sets to train: the pooled set (always kept for the Kaggle contract) plus every routed set."""
        used = {"pooled"} | {name for name in self.routing.values() if name}
        return sorted(used)

    def as_dict(self) -> dict:
        return {"conformal": self.conformal, "monotone": self.monotone, "gmp_model": self.gmp_model,
                "routing": dict(self.routing)}


# The choices made on the 2022-2024 blocks (see models/experiments.json and docs/MODEL_REPORT.md). Fixed before the
# locked 2025-2026 evaluation.
DEFAULT_DESIGN = Design(conformal=True, monotone=True, gmp_model="blend",
                        routing={"IN_MAIN": "IN_MAIN", "IN_SME": "india", "GLOBAL": "pooled"})

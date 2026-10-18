"""What training saves and serving loads.

Every new field has a default, so older callers (the Kaggle job stubs, tests) that build a Bundle or StageModels with
the original keyword arguments still work. An Ensemble wraps one model per training seed and answers the same
predict/predict_proba calls as a single LightGBM model, so `models.q50.predict(X)` keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression

from ipo_radar.model.features import Vocab


class Ensemble:
    """Average of models trained with different seeds. Predictions and contributions are averaged; for a single
    member this is exactly that model."""

    def __init__(self, members: list) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)

    def predict(self, X, pred_contrib: bool = False, **kwargs):
        return np.mean([m.predict(X, pred_contrib=pred_contrib, **kwargs) for m in self.members], axis=0)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    @property
    def booster_(self):
        return self.members[0].booster_

    @property
    def feature_name_(self):
        return self.members[0].feature_name_


@dataclass
class StageModels:
    stage: str
    features: list[str]
    q10: Ensemble | LGBMRegressor                 # open gain, transformed scale (features.to_model_scale)
    q50: Ensemble | LGBMRegressor
    q90: Ensemble | LGBMRegressor
    close_q50: Ensemble | LGBMRegressor | None    # None when the stage had too few close-gain targets
    clf: Ensemble | LGBMClassifier | None         # P(open gain > 0), before calibration
    iso: IsotonicRegression | None                # calibrator for clf; None when too few out-of-fold points
    n_open: int
    n_close: int
    n_by_market: dict[str, int]                   # training IPOs per market with an open-gain target
    core: dict[str, tuple[str, ...]]              # market -> inputs that must be present (coverage >= 50%)
    ranges: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)  # market -> feature -> (p1, p99)
    conf: dict[str, float] = field(default_factory=dict)      # segment (or ALL) -> conformal adjustment in pct points
    conf_n: dict[str, int] = field(default_factory=dict)      # out-of-fold residuals behind each adjustment
    gmp_linear: list | None = None                # three linear quantile regressions (p10, p50, p90), pre_listing only
    gmp_counts: dict[str, dict[int, int]] = field(default_factory=dict)  # segment -> listing year -> GMP rows


@dataclass
class Bundle:
    version: str
    trained_at: str
    n_rows: int
    stages: dict[str, StageModels]                # the pooled model set (kept under this name for compatibility)
    vocab: Vocab
    seed: int
    seeds: list[int] = field(default_factory=list)
    design: dict = field(default_factory=dict)    # the Design the bundle was built with
    routing: dict[str, str | None] = field(default_factory=dict)  # segment -> model set name
    alt_sets: dict[str, dict[str, StageModels]] = field(default_factory=dict)  # every set except 'pooled'
    skill: dict[str, dict[str, dict]] = field(default_factory=dict)  # stage -> segment -> pre-2025 skill flags
    protocol: str = ""

    def set_models(self, name: str | None) -> dict[str, StageModels]:
        """Stage models of a named set; the pooled set when the name is missing or not trained."""
        if name is None:
            return self.stages
        if name == "pooled":
            return self.stages
        return self.alt_sets.get(name, self.stages)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _to_numpy(a: Iterable) -> np.ndarray:
    if isinstance(a, (pd.Series, pd.Index)):
        return a.to_numpy()
    arr = np.asarray(a)
    return arr


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_time_decay_weights(
    timestamps: Iterable,
    *,
    half_life_days: float,
    ref_timestamp: Optional[pd.Timestamp] = None,
    w_min: float = 0.0,
) -> np.ndarray:
    """Compute exponential time-decay weights based on age in days.

    w = max(w_min, 0.5 ** (age_days / half_life_days))
    where age_days = (ref_timestamp - ts) in days. ref defaults to max(timestamps).
    Robust to Series/Index/ndarray inputs.
    """
    # Normalize to pandas Series[datetime64]
    ts_ser = pd.to_datetime(pd.Series(_to_numpy(timestamps)), errors="coerce")
    ref = pd.to_datetime(ref_timestamp) if ref_timestamp is not None else pd.to_datetime(ts_ser.max())
    # Compute age in days without relying on .dt on Index/TimedeltaIndex
    age_days = ((ref - ts_ser) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    raw = np.power(0.5, age_days / float(half_life_days))
    if w_min and w_min > 0:
        raw = np.maximum(raw, float(w_min))
    return raw.astype(float)


@dataclass
class PlattCalibrator:
    C: float = 1.0
    use_logit: bool = True
    max_iter: int = 1000
    random_state: int = 42

    def __post_init__(self) -> None:
        self._model = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=self.max_iter, random_state=self.random_state
        )

    def fit(
        self,
        y_pred: Iterable,
        y_true: Iterable,
        *,
        sample_weight: Optional[Iterable] = None,
    ) -> "PlattCalibrator":
        p = _to_numpy(y_pred).astype(float)
        y = _to_numpy(y_true).astype(float)
        x = logit(p) if self.use_logit else np.clip(p, 1e-12, 1 - 1e-12)
        X = x.reshape(-1, 1)
        sw = None if sample_weight is None else _to_numpy(sample_weight).astype(float)
        self._model.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, y_pred: Iterable) -> np.ndarray:
        p = _to_numpy(y_pred).astype(float)
        x = logit(p) if self.use_logit else np.clip(p, 1e-12, 1 - 1e-12)
        X = x.reshape(-1, 1)
        prob = self._model.predict_proba(X)[:, 1]
        return np.clip(prob, 1e-12, 1 - 1e-12)


@dataclass
class IsotonicCalibrator:
    y_min: float = 0.0
    y_max: float = 1.0
    out_of_bounds: str = "clip"

    def __post_init__(self) -> None:
        self._iso = IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds=self.out_of_bounds)

    def fit(
        self,
        y_pred: Iterable,
        y_true: Iterable,
        *,
        sample_weight: Optional[Iterable] = None,
    ) -> "IsotonicCalibrator":
        p = _to_numpy(y_pred).astype(float)
        y = _to_numpy(y_true).astype(float)
        sw = None if sample_weight is None else _to_numpy(sample_weight).astype(float)
        self._iso.fit(p, y, sample_weight=sw)
        return self

    def predict_proba(self, y_pred: Iterable) -> np.ndarray:
        p = _to_numpy(y_pred).astype(float)
        prob = self._iso.transform(p)
        return np.clip(prob, 1e-12, 1 - 1e-12)

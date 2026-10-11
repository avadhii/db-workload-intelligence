"""
anomaly.py — Anomaly detection engine (Phase 3).

Three complementary methods:
A. Forecast residuals outside prediction interval (uses Phase 2 model)
B. STL decomposition + robust z-score
C. Isolation Forest on multivariate metric vector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    ts: datetime
    metric: str
    value: float
    score: float        # 0–1 (higher = more anomalous)
    method: str
    is_anomaly: bool
    context: dict = field(default_factory=dict)


def detect_residual_anomalies(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    timestamps: pd.Series,
    metric: str,
    persistence_n: int = 3,
    persistence_m: int = 5,
) -> list[AnomalyScore]:
    """
    Method A: Flag points where actual is outside [lower, upper] prediction interval.
    Requires N of M consecutive windows to fire (reduces false positives).
    """
    anomalies = []
    outside = ((actual < lower) | (actual > upper)).values
    n = len(actual)

    for i in range(n):
        window = outside[max(0, i - persistence_m + 1): i + 1]
        triggered = window.sum() >= persistence_n

        if triggered:
            val = float(actual.iloc[i])
            lo  = float(lower.iloc[i])
            hi  = float(upper.iloc[i])
            score = min(1.0, max(
                (lo - val) / max(abs(lo), 1e-9) if val < lo else 0,
                (val - hi) / max(abs(hi), 1e-9) if val > hi else 0,
            ))
            anomalies.append(AnomalyScore(
                ts=timestamps.iloc[i],
                metric=metric,
                value=val,
                score=score,
                method="residual",
                is_anomaly=True,
                context={"lower": lo, "upper": hi},
            ))

    return anomalies


def detect_stl_anomalies(
    series: pd.Series,
    timestamps: pd.Series,
    metric: str,
    period: int = 288,   # 5-min buckets → 1 day
    z_threshold: float = 3.5,
) -> list[AnomalyScore]:
    """
    Method B: STL decomposition; flag residuals by robust z-score (median/MAD).
    """
    from statsmodels.tsa.seasonal import STL

    if len(series) < period * 2:
        logger.warning("Not enough data for STL (need %d, have %d)", period * 2, len(series))
        return []

    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        residuals = result.resid.values
    except Exception as exc:
        logger.error("STL decomposition failed: %s", exc)
        return []

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    robust_z = np.abs(residuals - median) / max(mad * 1.4826, 1e-9)

    anomalies = []
    for i, (rz, val, ts) in enumerate(zip(robust_z, series.values, timestamps)):
        if rz > z_threshold:
            anomalies.append(AnomalyScore(
                ts=ts, metric=metric, value=float(val),
                score=min(1.0, float(rz) / (z_threshold * 2)),
                method="stl_zscore", is_anomaly=True,
                context={"robust_z": float(rz), "residual": float(residuals[i])},
            ))

    return anomalies


def detect_isolation_forest(
    df: pd.DataFrame,
    timestamps: pd.Series,
    contamination: float = 0.05,
) -> list[AnomalyScore]:
    """
    Method C: Isolation Forest on multivariate metric vector.
    df columns should be the metric columns to consider jointly.
    """
    X = df.fillna(df.median()).values
    clf = IsolationForest(contamination=contamination, random_state=42)
    labels = clf.fit_predict(X)   # -1 = anomaly, 1 = normal
    scores = clf.score_samples(X)

    anomalies = []
    for i, (label, score, ts) in enumerate(zip(labels, scores, timestamps)):
        if label == -1:
            anomalies.append(AnomalyScore(
                ts=ts, metric="multivariate",
                value=float(score),
                score=float(-score / max(abs(scores.min()), 1e-9)),
                method="isolation_forest", is_anomaly=True,
                context={"raw_score": float(score)},
            ))

    return anomalies


def merge_and_deduplicate(
    anomaly_lists: list[list[AnomalyScore]],
    cooldown_seconds: int = 300,
) -> list[AnomalyScore]:
    """
    Merge anomalies from multiple detectors.
    Apply cooldown window to suppress overlapping detections.
    """
    all_anomalies = [a for lst in anomaly_lists for a in lst]
    all_anomalies.sort(key=lambda a: a.ts)

    deduped = []
    last_fired: dict[str, datetime] = {}

    for a in all_anomalies:
        key = f"{a.metric}_{a.method}"
        if key in last_fired:
            delta = (a.ts - last_fired[key]).total_seconds()
            if delta < cooldown_seconds:
                continue
        deduped.append(a)
        last_fired[key] = a.ts

    return deduped

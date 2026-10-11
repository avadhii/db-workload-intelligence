"""
features.py — Feature engineering for the forecasting model (Phase 2).

Pulls raw metric_snapshot rows from the metrics DB and builds
a feature matrix with lag features, rolling statistics, and
calendar features suitable for LightGBM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_calendar_features(df: pd.DataFrame, ts_col: str = "bucket") -> pd.DataFrame:
    """
    Add time-based features: hour, day-of-week, month, is_weekend, etc.
    """
    df = df.copy()
    df["hour"]        = df[ts_col].dt.hour
    df["dayofweek"]   = df[ts_col].dt.dayofweek
    df["month"]       = df[ts_col].dt.month
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lag and rolling statistics features for time-series modeling.

    Args:
        df: DataFrame sorted by time with `target_col` present.
        lags: Number of steps back to include as lag features.
        rolling_windows: Window sizes for rolling mean/std.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48]   # in units of your bucket interval
    if rolling_windows is None:
        rolling_windows = [3, 6, 12, 24]

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df[target_col].shift(1).rolling(w).std()

    return df


def make_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    ts_col: str = "bucket",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline.

    Returns:
        X: Feature DataFrame (rows with NaN from lags dropped)
        y: Target series
    """
    df = build_calendar_features(df, ts_col)
    df = build_lag_features(df, target_col)
    df = df.dropna()

    feature_cols = [c for c in df.columns if c not in [ts_col, target_col]]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

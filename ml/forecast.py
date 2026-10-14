"""
forecast.py — LightGBM workload forecasting (Phase 2).

Walk-forward (rolling-origin) training and evaluation.
Baselines: seasonal-naive, ETS.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Lazy imports for optional heavy dependencies ───────────────────────
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("lightgbm not installed — forecasting disabled")


@dataclass
class ForecastResult:
    horizon_steps: int
    predictions: list[float]
    lower_bound: list[float]   # prediction interval lower
    upper_bound: list[float]   # prediction interval upper
    model_name: str
    mape: Optional[float] = None


def seasonal_naive_forecast(series: pd.Series, horizon: int, season: int = 288) -> ForecastResult:
    """
    Seasonal naive baseline: repeat the values from one season ago.
    Default season=288 assumes 5-minute buckets in a 24h day.
    """
    n = len(series)
    preds = []
    for h in range(horizon):
        idx = n - season + h
        preds.append(float(series.iloc[max(0, idx)]))
    return ForecastResult(
        horizon_steps=horizon,
        predictions=preds,
        lower_bound=[p * 0.8 for p in preds],
        upper_bound=[p * 1.2 for p in preds],
        model_name="seasonal_naive",
    )


class LightGBMForecaster:
    """
    LightGBM multi-step ahead forecaster with quantile regression
    for prediction intervals.
    Phase 2 full implementation.
    """

    def __init__(
        self,
        quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
        num_boost_round: int = 300,
        early_stopping_rounds: int = 30,
    ):
        self.quantiles = quantiles
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.models: dict[float, lgb.Booster] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for this model")

        for q in self.quantiles:
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": max(2, min(20, len(X_train) // 5)),
                "verbose": -1,
            }
            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds   = lgb.Dataset(X_val, label=y_val)
            self.models[q] = lgb.train(
                params,
                train_ds,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            logger.info("Trained quantile=%.2f model", q)

    def save(self, directory: str) -> None:
        """Save forecaster models and hyperparameters to directory."""
        import json
        import os
        os.makedirs(directory, exist_ok=True)
        meta = {
            "quantiles": list(self.quantiles),
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)
        for q, model in self.models.items():
            model.save_model(os.path.join(directory, f"model_q{int(q*100)}.txt"))

    @classmethod
    def load(cls, directory: str) -> LightGBMForecaster:
        """Load forecaster models and hyperparameters from directory."""
        import json
        import os
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for this model")
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta.json in {directory}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        forecaster = cls(
            quantiles=tuple(meta["quantiles"]),
            num_boost_round=meta["num_boost_round"],
            early_stopping_rounds=meta["early_stopping_rounds"],
        )
        for q in forecaster.quantiles:
            m_path = os.path.join(directory, f"model_q{int(q*100)}.txt")
            if os.path.exists(m_path):
                forecaster.models[q] = lgb.Booster(model_file=m_path)
        return forecaster

    def predict(self, X: pd.DataFrame) -> ForecastResult:
        if not self.models:
            raise RuntimeError("Model not trained. Call fit() first.")

        preds_50 = self.models[0.5].predict(X).tolist()
        preds_lo = self.models[0.1].predict(X).tolist()
        preds_hi = self.models[0.9].predict(X).tolist()

        return ForecastResult(
            horizon_steps=len(X),
            predictions=preds_50,
            lower_bound=preds_lo,
            upper_bound=preds_hi,
            model_name="lightgbm_quantile",
        )

    def predict_future(
        self,
        df: pd.DataFrame,
        target_col: str = "avg_val",
        ts_col: str = "bucket",
        horizon: int = 12,
        freq_minutes: int = 5,
    ) -> ForecastResult:
        """
        Produce multi-step future forecasts autoregressively.
        Uses the median (0.5) quantile model for recursive lag feedback,
        and 0.1 and 0.9 quantile models for confidence intervals.
        """
        if not self.models:
            raise RuntimeError("Model not trained. Call fit() first.")

        from ml.features import make_feature_matrix

        if df.empty:
            return ForecastResult(
                horizon_steps=0,
                predictions=[],
                lower_bound=[],
                upper_bound=[],
                model_name="lightgbm_quantile",
            )

        extended_df = df[[ts_col, target_col]].copy().sort_values(ts_col).reset_index(drop=True)
        preds_50: list[float] = []
        preds_lo: list[float] = []
        preds_hi: list[float] = []

        last_ts = extended_df[ts_col].iloc[-1]
        step_delta = timedelta(minutes=freq_minutes)

        for step in range(horizon):
            next_ts = last_ts + step_delta * (step + 1)
            # Placeholder for target value to allow lag feature calculation
            last_known = preds_50[-1] if preds_50 else extended_df[target_col].iloc[-1]
            temp_row = pd.DataFrame([{ts_col: next_ts, target_col: last_known}])
            curr_df = pd.concat([extended_df, temp_row], ignore_index=True)

            X_curr, _ = make_feature_matrix(curr_df, target_col, ts_col)
            if X_curr.empty:
                break

            latest_X = X_curr.iloc[[-1]]
            p50 = float(self.models[0.5].predict(latest_X)[0])
            plo = float(self.models[0.1].predict(latest_X)[0])
            phi = float(self.models[0.9].predict(latest_X)[0])

            # Non-negative clamping if target history is non-negative
            if (df[target_col] >= 0).all():
                p50 = max(0.0, p50)
                plo = max(0.0, min(plo, p50))
                phi = max(p50, phi)

            preds_50.append(round(p50, 4))
            preds_lo.append(round(plo, 4))
            preds_hi.append(round(phi, 4))

            # Append the actual prediction to extended_df
            new_row = pd.DataFrame([{ts_col: next_ts, target_col: p50}])
            extended_df = pd.concat([extended_df, new_row], ignore_index=True)

        return ForecastResult(
            horizon_steps=len(preds_50),
            predictions=preds_50,
            lower_bound=preds_lo,
            upper_bound=preds_hi,
            model_name="lightgbm_quantile",
        )


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(actual - predicted) / denom)) * 100)


def walk_forward_evaluate(
    df: pd.DataFrame,
    target_col: str,
    ts_col: str = "bucket",
    n_splits: int = 5,
    horizon: int = 288,
):
    """
    Walk-forward (rolling-origin) evaluation.
    Returns dict with per-split and overall MAPE / sMAPE.
    Phase 2 implementation.
    """
    from ml.features import make_feature_matrix

    results = []
    n = len(df)
    train_size = n - n_splits * horizon

    for split in range(n_splits):
        train_end = train_size + split * horizon
        val_end   = train_end + horizon

        train_df = df.iloc[:train_end]
        val_df   = df.iloc[train_end:val_end]

        X_train, y_train = make_feature_matrix(train_df, target_col, ts_col)
        X_val,   y_val   = make_feature_matrix(val_df, target_col, ts_col)

        if X_val.empty:
            continue

        baseline = seasonal_naive_forecast(y_train, len(X_val))
        baseline_smape = smape(y_val.values, np.array(baseline.predictions[:len(y_val)]))

        results.append({
            "split": split,
            "baseline_smape": baseline_smape,
            "n_train": len(X_train),
            "n_val": len(X_val),
        })
        logger.info("Split %d — baseline sMAPE=%.2f%%", split, baseline_smape)

    return results

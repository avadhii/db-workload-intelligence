"""
Forecast router — LightGBM-powered workload forecasting.

Endpoints:
  GET  /forecast/               → fetch LightGBM forecast for a metric
  POST /forecast/train          → trigger model (re)training
  GET  /forecast/model-status   → check if model is trained + metadata
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.database import get_db
from ml.features import build_calendar_features, build_lag_features, make_feature_matrix
from ml.forecast import (
    LightGBMForecaster,
    ForecastResult,
    seasonal_naive_forecast,
    smape,
    HAS_LIGHTGBM,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Module-level model cache ───────────────────────────────────────────
_forecaster: Optional[LightGBMForecaster] = None
_train_meta: dict = {}       # stores training metadata
_FORECAST_MODEL_CACHE_DIR = os.environ.get(
    "FORECAST_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models_cache", "forecast"),
)


def _try_load_cached_forecaster():
    global _forecaster, _train_meta
    meta_path = os.path.join(_FORECAST_MODEL_CACHE_DIR, "train_meta.json")
    if os.path.exists(meta_path) and HAS_LIGHTGBM:
        try:
            with open(meta_path, "r") as f:
                _train_meta = json.load(f)
            _forecaster = LightGBMForecaster.load(_FORECAST_MODEL_CACHE_DIR)
            logger.info("Loaded cached LightGBM forecaster from %s", _FORECAST_MODEL_CACHE_DIR)
        except Exception as exc:
            logger.warning("Failed to load cached forecaster: %s", exc)


_try_load_cached_forecaster()


class TrainRequest(BaseModel):
    metric: str = "db.qps"
    hours: int = 48          # how many hours of history to train on
    bucket_minutes: int = 5


class TrainResponse(BaseModel):
    status: str
    metric: str
    training_rows: int
    message: str


# ── Helpers ─────────────────────────────────────────────────────────────

async def _fetch_metric_timeseries(
    db, metric: str, hours: int, bucket_minutes: int = 5,
) -> pd.DataFrame:
    """Fetch bucketed metric timeseries from the metrics DB."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                date_bin($1::interval, ts, TIMESTAMP '2000-01-01') AS bucket,
                AVG(value) AS avg_val
            FROM metric_snapshot
            WHERE ts >= $2 AND metric = $3
            GROUP BY bucket
            ORDER BY bucket
            """,
            timedelta(minutes=bucket_minutes), since, metric,
        )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    df = df.sort_values("bucket").reset_index(drop=True)
    return df


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
async def train_forecast_model(
    req: TrainRequest = TrainRequest(),
    db=Depends(get_db),
):
    """
    Train (or retrain) the LightGBM forecaster on historical metric data.
    The model is held in memory for subsequent /forecast/ predictions.
    """
    global _forecaster, _train_meta

    if not HAS_LIGHTGBM:
        raise HTTPException(
            status_code=503,
            detail="LightGBM is not installed. Install with: pip install lightgbm",
        )

    df = await _fetch_metric_timeseries(db, req.metric, req.hours, req.bucket_minutes)
    if df.empty or len(df) < 60:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least 60 data points, got {len(df)}. "
                   f"Let the collector run longer or increase the 'hours' parameter.",
        )

    target_col = "avg_val"
    ts_col = "bucket"

    try:
        X, y = make_feature_matrix(df, target_col, ts_col)

        if len(X) < 50:
            raise HTTPException(
                status_code=422,
                detail=f"After feature engineering only {len(X)} rows remain (need ≥50). "
                       f"Increase the training window.",
            )

        # 80/20 train-validation split
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

        forecaster = LightGBMForecaster()
        forecaster.fit(X_train, y_train, X_val, y_val)

        # Evaluate on validation set
        result = forecaster.predict(X_val)
        val_smape = smape(y_val.values, np.array(result.predictions[:len(y_val)]))

        _forecaster = forecaster
        _train_meta = {
            "metric": req.metric,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": len(X),
            "validation_smape": round(val_smape, 2),
            "bucket_minutes": req.bucket_minutes,
            "hours_used": req.hours,
        }

        try:
            _forecaster.save(_FORECAST_MODEL_CACHE_DIR)
            with open(os.path.join(_FORECAST_MODEL_CACHE_DIR, "train_meta.json"), "w") as f:
                json.dump(_train_meta, f)
        except Exception as save_exc:
            logger.warning("Failed to save forecast model to disk: %s", save_exc)

        logger.info(
            "LightGBM forecast model trained: %d rows, val sMAPE=%.2f%%",
            len(X), val_smape,
        )

        return TrainResponse(
            status="trained",
            metric=req.metric,
            training_rows=len(X),
            message=f"Model trained successfully. Validation sMAPE: {val_smape:.2f}%",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Forecast training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@router.get("/")
async def get_forecast(
    metric: str = Query(default="db.qps", description="Metric to forecast"),
    hours: int = Query(default=24, ge=1, le=168, description="Historical look-back window"),
    horizon: int = Query(default=12, ge=1, le=96, description="Number of future steps to predict"),
    bucket_minutes: int = Query(default=5, ge=1, le=60),
    db=Depends(get_db),
):
    """
    Return LightGBM forecast predictions with confidence intervals.

    If the model is not trained, falls back to the seasonal-naive baseline
    and includes a hint to call POST /forecast/train first.
    """
    # Fetch historical data (at least 24 hours to ensure enough lag features)
    fetch_hours = max(hours, 24)
    df = await _fetch_metric_timeseries(db, metric, fetch_hours, bucket_minutes)

    if df.empty or len(df) < 10:
        return {
            "metric": metric,
            "model": "none",
            "message": "Insufficient historical data for forecasting.",
            "forecast": [],
        }

    target_col = "avg_val"
    ts_col = "bucket"

    # Always compute baseline
    baseline = seasonal_naive_forecast(
        pd.Series(df[target_col].values),
        horizon=horizon,
        season=min(288, len(df)),
    )

    last_bucket = df["bucket"].iloc[-1]
    interval = timedelta(minutes=bucket_minutes)

    baseline_points = []
    for i, (pred, lo, hi) in enumerate(zip(
        baseline.predictions, baseline.lower_bound, baseline.upper_bound
    )):
        ts = last_bucket + interval * (i + 1)
        baseline_points.append({
            "ts": ts.isoformat(),
            "predicted": round(pred, 4),
            "lower": round(lo, 4),
            "upper": round(hi, 4),
        })

    result = {
        "metric": metric,
        "historical_points": len(df),
        "horizon": horizon,
        "bucket_minutes": bucket_minutes,
        "baseline_forecast": baseline_points,
    }

    # LightGBM forecast
    if _forecaster is not None and _train_meta.get("metric") == metric:
        try:
            lgbm_result = _forecaster.predict_future(
                df,
                target_col=target_col,
                ts_col=ts_col,
                horizon=horizon,
                freq_minutes=bucket_minutes,
            )
            if lgbm_result and lgbm_result.predictions:
                lgbm_points = []
                for i, (pred, lo, hi) in enumerate(zip(
                    lgbm_result.predictions, lgbm_result.lower_bound, lgbm_result.upper_bound
                )):
                    ts = last_bucket + interval * (i + 1)
                    lgbm_points.append({
                        "ts": ts.isoformat(),
                        "predicted": round(pred, 4),
                        "lower": round(lo, 4),
                        "upper": round(hi, 4),
                    })
                result["lightgbm_forecast"] = lgbm_points
                result["model"] = "lightgbm_quantile"
                result["model_meta"] = _train_meta
            else:
                result["model"] = "baseline_only"
                result["model_hint"] = "Not enough lag history to compute LightGBM features; baseline used."
        except Exception as exc:
            logger.warning("LightGBM prediction failed, falling back to baseline: %s", exc)
            result["model"] = "baseline_only"
            result["model_error"] = str(exc)
    else:
        result["model"] = "baseline_only"
        result["model_hint"] = (
            "LightGBM model not trained for this metric. "
            "Call POST /forecast/train to train."
        )

    return result


@router.get("/model-status")
async def model_status():
    """Return the current state of the in-memory forecast model."""
    if _forecaster is None:
        return {"trained": False, "message": "No model trained. POST /forecast/train to train."}
    return {"trained": True, **_train_meta}

"""
Anomalies router — Isolation Forest ML-powered anomaly detection + DB incidents.

Endpoints:
  GET  /anomalies/                → list DB incidents (existing)
  GET  /anomalies/detect          → run Isolation Forest on recent metrics
  POST /anomalies/train           → train the Isolation Forest model
  GET  /anomalies/model-status    → check if anomaly model is trained
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
from ml.anomaly import (
    detect_isolation_forest,
    detect_residual_anomalies,
    merge_and_deduplicate,
    AnomalyScore,
    IsolationForestDetector,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Module-level model cache ───────────────────────────────────────────
_detector: Optional[IsolationForestDetector] = None
_anomaly_train_meta: dict = {}
_ANOMALY_MODEL_CACHE_DIR = os.environ.get(
    "ANOMALY_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models_cache", "anomaly"),
)


def _try_load_cached_detector():
    global _detector, _anomaly_train_meta
    meta_path = os.path.join(_ANOMALY_MODEL_CACHE_DIR, "train_meta.json")
    model_path = os.path.join(_ANOMALY_MODEL_CACHE_DIR, "detector.joblib")
    if os.path.exists(meta_path) and os.path.exists(model_path):
        try:
            with open(meta_path, "r") as f:
                _anomaly_train_meta = json.load(f)
            _detector = IsolationForestDetector.load(model_path)
            logger.info("Loaded cached Isolation Forest detector from %s", model_path)
        except Exception as exc:
            logger.warning("Failed to load cached anomaly detector: %s", exc)


_try_load_cached_detector()

# Metrics to use for multivariate anomaly detection
_ANOMALY_METRICS = [
    "db.qps",
    "db.connections",
    "db.cache_hit_ratio",
    "db.deadlocks",
    "db.temp_files",
    "db.rollback_ratio",
]


# ── Helpers ─────────────────────────────────────────────────────────────

async def _fetch_pivot_metrics(
    db, hours: int, bucket_minutes: int = 5,
) -> pd.DataFrame:
    """
    Fetch multiple metrics from metric_snapshot, pivot into a wide DataFrame
    with one column per metric and timestamp as index.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    metrics_list = _ANOMALY_METRICS

    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                date_bin($1::interval, ts, TIMESTAMP '2000-01-01') AS bucket,
                metric,
                AVG(value) AS avg_val
            FROM metric_snapshot
            WHERE ts >= $2 AND metric = ANY($3)
            GROUP BY bucket, metric
            ORDER BY bucket
            """,
            timedelta(minutes=bucket_minutes), since, metrics_list,
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)

    # Pivot to wide format: each metric is a column
    pivoted = df.pivot_table(
        index="bucket", columns="metric", values="avg_val", aggfunc="last",
    ).reset_index()
    pivoted = pivoted.sort_values("bucket")

    # Rename columns for cleaner access
    pivoted.columns.name = None

    return pivoted


class TrainAnomalyRequest(BaseModel):
    hours: int = 48
    bucket_minutes: int = 5
    contamination: float = 0.05


class TrainAnomalyResponse(BaseModel):
    status: str
    samples_used: int
    contamination: float
    message: str


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("/")
async def list_anomalies(
    hours: int = Query(default=24, ge=1, le=168),
    db=Depends(get_db),
):
    """List incidents from the database (injected faults + detected anomalies)."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM incident WHERE start_ts >= $1 ORDER BY start_ts DESC",
            since,
        )
    return {"anomalies": [dict(r) for r in rows]}


@router.post("/train", response_model=TrainAnomalyResponse)
async def train_anomaly_model(
    req: TrainAnomalyRequest = TrainAnomalyRequest(),
    db=Depends(get_db),
):
    """
    Train (or retrain) the Isolation Forest on historical multi-metric data.
    The model is held in memory for subsequent /anomalies/detect requests.
    """
    global _detector, _anomaly_train_meta

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="scikit-learn is not installed. Install with: pip install scikit-learn",
        )

    df = await _fetch_pivot_metrics(db, req.hours, req.bucket_minutes)
    if df.empty or len(df) < 30:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data: need at least 30 rows, got {len(df)}. "
                   f"Let the collector run longer.",
        )

    # Use only the metric columns that exist in the data
    feature_cols = [c for c in _ANOMALY_METRICS if c in df.columns]
    if len(feature_cols) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 metrics present, found: {feature_cols}",
        )

    try:
        detector = IsolationForestDetector(
            contamination=req.contamination,
            n_estimators=150,
            random_state=42,
        )
        detector.fit(df, feature_cols=feature_cols)

        _detector = detector
        _anomaly_train_meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "samples_used": len(df),
            "contamination": req.contamination,
            "feature_cols": feature_cols,
            "hours_used": req.hours,
        }

        try:
            os.makedirs(_ANOMALY_MODEL_CACHE_DIR, exist_ok=True)
            _detector.save(os.path.join(_ANOMALY_MODEL_CACHE_DIR, "detector.joblib"))
            with open(os.path.join(_ANOMALY_MODEL_CACHE_DIR, "train_meta.json"), "w") as f:
                json.dump(_anomaly_train_meta, f)
        except Exception as save_exc:
            logger.warning("Failed to save anomaly detector to disk: %s", save_exc)

        logger.info(
            "Isolation Forest trained: %d samples, contamination=%.2f, features=%s",
            len(df), req.contamination, feature_cols,
        )

        return TrainAnomalyResponse(
            status="trained",
            samples_used=len(df),
            contamination=req.contamination,
            message=f"Isolation Forest trained on {len(df)} samples with {len(feature_cols)} features.",
        )
    except Exception as exc:
        logger.exception("Anomaly model training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@router.get("/detect")
async def detect_anomalies(
    hours: int = Query(default=6, ge=1, le=168, description="Look-back window for detection"),
    bucket_minutes: int = Query(default=5, ge=1, le=60),
    contamination: float = Query(default=0.05, ge=0.01, le=0.5),
    record: bool = Query(default=False, description="Persist detected anomalies into the incident table"),
    db=Depends(get_db),
):
    """
    Run Isolation Forest anomaly detection on recent metric data.

    If a pre-trained model exists, uses it. Otherwise, trains an ad-hoc model
    on the requested window (less ideal but functional).
    """
    df = await _fetch_pivot_metrics(db, hours, bucket_minutes)

    if df.empty or len(df) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 10 data points, got {len(df)}.",
            "anomalies": [],
            "total_points": len(df),
        }

    timestamps = df["bucket"]

    if _detector is not None:
        feature_cols = _detector.feature_cols
        model_source = "pre_trained"
    else:
        feature_cols = [c for c in _ANOMALY_METRICS if c in df.columns]
        model_source = "ad_hoc"

    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < 2:
        return {
            "status": "insufficient_metrics",
            "message": f"Need at least 2 metrics, found: {available_cols}",
            "anomalies": [],
        }

    metric_df = df[available_cols].fillna(df[available_cols].median()).fillna(0.0)

    try:
        if _detector is not None and model_source == "pre_trained":
            anomaly_results = _detector.detect(df, timestamps)
        else:
            anomaly_results = detect_isolation_forest(
                metric_df, timestamps, contamination=contamination,
            )

        anomalies_out = []
        for a in anomaly_results:
            anomalies_out.append({
                "ts": a.ts.isoformat() if hasattr(a.ts, "isoformat") else str(a.ts),
                "score": round(a.score, 4),
                "is_anomaly": a.is_anomaly,
                "method": a.method,
                "metrics": a.context.get("metrics", {}),
            })

        # Persist detected anomalies to the incident table if requested
        recorded_count = 0
        if record and anomalies_out:
            async with db.acquire() as conn:
                for a in anomalies_out:
                    ts_val = pd.to_datetime(a["ts"])
                    existing = await conn.fetchval(
                        "SELECT id FROM incident WHERE source = 'detected' AND start_ts = $1",
                        ts_val,
                    )
                    if not existing:
                        sev = "high" if a["score"] > 0.8 else ("medium" if a["score"] > 0.4 else "low")
                        await conn.execute(
                            """
                            INSERT INTO incident (start_ts, kind, source, severity, description, suspected_causes)
                            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                            """,
                            ts_val,
                            "workload_anomaly",
                            "detected",
                            sev,
                            f"Isolation Forest detected multivariate anomaly (score: {a['score']})",
                            json.dumps(a["metrics"]),
                        )
                        recorded_count += 1

        return {
            "status": "detected",
            "model_source": model_source,
            "total_points": len(df),
            "anomaly_count": len(anomalies_out),
            "recorded_to_incidents": recorded_count if record else 0,
            "anomalies": anomalies_out,
            "feature_columns": available_cols,
            "model_meta": _anomaly_train_meta if model_source == "pre_trained" else None,
        }
    except Exception as exc:
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")


@router.get("/model-status")
async def anomaly_model_status():
    """Return the current state of the in-memory anomaly model."""
    if _detector is None:
        return {
            "trained": False,
            "message": "No model trained. POST /anomalies/train to train.",
        }
    return {"trained": True, **_anomaly_train_meta}

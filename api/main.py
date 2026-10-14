"""
FastAPI application: exposes metrics, forecasts, anomalies, and recommendations.

Includes ML model lifecycle: optional auto-training on startup when sufficient
historical data is available in the metrics DB.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.routers import health, metrics, forecast, anomalies, incidents, recommendations, whatif

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # ── Optional: auto-train ML models if enough data exists ────────────
    auto_train = os.environ.get("AUTO_TRAIN_ML", "false").lower() == "true"
    if auto_train:
        logger.info("AUTO_TRAIN_ML enabled — attempting to train forecast and anomaly models...")
        try:
            from api.database import get_db
            db = await get_db()

            # Count available data points
            async with db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) AS cnt FROM metric_snapshot WHERE metric = 'db.qps'"
                )
                data_points = row["cnt"] if row else 0

            if data_points >= 60:
                logger.info("Found %d data points — training models...", data_points)
                # Train forecast model
                try:
                    from api.routers.forecast import train_forecast_model, TrainRequest
                    await train_forecast_model(TrainRequest(metric="db.qps", hours=48), db=db)
                    logger.info("Forecast model auto-trained successfully")
                except Exception as exc:
                    logger.warning("Forecast auto-train failed (non-fatal): %s", exc)

                # Train anomaly model
                try:
                    from api.routers.anomalies import train_anomaly_model, TrainAnomalyRequest
                    await train_anomaly_model(TrainAnomalyRequest(hours=48), db=db)
                    logger.info("Anomaly model auto-trained successfully")
                except Exception as exc:
                    logger.warning("Anomaly auto-train failed (non-fatal): %s", exc)
            else:
                logger.info(
                    "Only %d data points available — skipping auto-train (need ≥60)",
                    data_points,
                )
        except Exception as exc:
            logger.warning("Auto-train check failed (non-fatal): %s", exc)

    yield


app = FastAPI(
    title="DB Workload Intelligence API",
    description="PostgreSQL workload forecasting, anomaly detection, and index recommendations",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ────────────────────────────────────────────────────────────
app.include_router(health.router,           tags=["Health"])
app.include_router(metrics.router,          prefix="/metrics",          tags=["Metrics"])
app.include_router(forecast.router,         prefix="/forecast",         tags=["Forecast"])
app.include_router(anomalies.router,        prefix="/anomalies",        tags=["Anomalies"])
app.include_router(incidents.router,        prefix="/incidents",        tags=["Incidents"])
app.include_router(recommendations.router,  prefix="/recommendations",  tags=["Recommendations"])
app.include_router(whatif.router,           prefix="/whatif",           tags=["What-If"])


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus scrape endpoint exposing service health."""
    return "# HELP api_up API service health status\n# TYPE api_up gauge\napi_up 1\n"


@app.get("/")
def root():
    return {
        "service": "DB Workload Intelligence",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "ml_endpoints": {
            "forecast_train": "POST /forecast/train",
            "forecast_predict": "GET /forecast/",
            "forecast_status": "GET /forecast/model-status",
            "anomaly_train": "POST /anomalies/train",
            "anomaly_detect": "GET /anomalies/detect",
            "anomaly_status": "GET /anomalies/model-status",
        },
    }

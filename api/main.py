"""
FastAPI application: exposes metrics, forecasts, anomalies, and recommendations.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.routers import health, metrics, forecast, anomalies, incidents, recommendations, whatif


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="DB Workload Intelligence API",
    description="PostgreSQL workload forecasting, anomaly detection, and index recommendations",
    version="0.1.0",
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
        "docs": "/docs",
        "health": "/health",
    }

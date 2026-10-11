"""Metrics summary router."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.database import get_db

router = APIRouter()


@router.get("/summary")
async def metrics_summary(
    hours: int = Query(default=1, ge=1, le=168, description="Look-back window in hours"),
    db=Depends(get_db),
):
    """Return recent summary metrics: QPS, cache hit ratio, connections, etc."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT metric, AVG(value) AS avg_val, MAX(value) AS max_val, MIN(value) AS min_val
            FROM metric_snapshot
            WHERE ts >= $1
              AND metric IN (
                'db.qps', 'db.cache_hit_ratio', 'db.connections',
                'db.deadlocks', 'db.rollback_ratio', 'db.lock_waiters'
              )
            GROUP BY metric
            ORDER BY metric
            """,
            since,
        )
    return {
        "window_hours": hours,
        "since": since.isoformat(),
        "metrics": [dict(r) for r in rows],
    }


@router.get("/timeseries")
async def metrics_timeseries(
    metric: str = Query(..., description="Metric name e.g. db.qps"),
    hours: int = Query(default=24, ge=1, le=168),
    bucket_minutes: int = Query(default=5, ge=1, le=60),
    db=Depends(get_db),
):
    """Return time-bucketed metric values for charting."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                date_bin($1::interval, ts, TIMESTAMP '2000-01-01') AS bucket,
                AVG(value) AS avg_val,
                MIN(value) AS min_val,
                MAX(value) AS max_val
            FROM metric_snapshot
            WHERE ts >= $2 AND metric = $3
            GROUP BY bucket
            ORDER BY bucket
            """,
            timedelta(minutes=bucket_minutes), since, metric,
        )
    return {
        "metric": metric,
        "points": [dict(r) for r in rows],
    }


@router.get("/top-queries")
async def top_queries(
    hours: int = Query(default=1, ge=1, le=24),
    limit: int = Query(default=20, ge=1, le=100),
    db=Depends(get_db),
):
    """Return top N queries by total execution time."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                queryid,
                MAX(query_text) AS query_text,
                SUM(calls) AS total_calls,
                SUM(total_exec_ms) AS total_exec_ms,
                AVG(mean_ms) AS avg_mean_ms,
                SUM(blks_read) AS total_blks_read,
                SUM(temp_blks_read) AS total_temp_blks
            FROM query_stat_delta
            WHERE ts >= $1
            GROUP BY queryid
            ORDER BY total_exec_ms DESC
            LIMIT $2
            """,
            since, limit,
        )
    return {"queries": [dict(r) for r in rows]}

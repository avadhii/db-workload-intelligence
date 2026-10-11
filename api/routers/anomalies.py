"""Anomalies router."""
from fastapi import APIRouter, Depends, Query
from datetime import datetime, timezone, timedelta
from api.database import get_db

router = APIRouter()

@router.get("/")
async def list_anomalies(
    hours: int = Query(default=24, ge=1, le=168),
    db=Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM incident WHERE start_ts >= $1 ORDER BY start_ts DESC",
            since,
        )
    return {"anomalies": [dict(r) for r in rows]}

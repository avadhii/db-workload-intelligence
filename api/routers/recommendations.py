"""
Recommendations router — Phase 5/6: Listing, creation, approval, and runbook generation.
"""
from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.database import get_db
from whatif.runbook import generate_index_runbook

router = APIRouter()
logger = logging.getLogger(__name__)


class CreateRecommendationRequest(BaseModel):
    kind: str = "create_index"
    target_table: str
    target_columns: list[str]
    ddl: str
    rollback_ddl: str
    est_cost_before: float
    est_cost_after: float
    source_query: str = ""
    rationale: str = ""
    safety_score: float = 75.0


@router.get("/")
async def list_recommendations(
    status: str = Query(default="all"),
    db=Depends(get_db),
):
    """List recommendations filtered by status or all."""
    async with db.acquire() as conn:
        if status.lower() == "all":
            rows = await conn.fetch(
                "SELECT * FROM recommendation ORDER BY created_at DESC LIMIT 100;"
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM recommendation WHERE status = $1 ORDER BY created_at DESC LIMIT 100;",
                status.lower(),
            )
    return {"recommendations": [dict(r) for r in rows]}


@router.post("/")
async def create_recommendation(
    req: CreateRecommendationRequest,
    db=Depends(get_db),
):
    """
    Persist an index recommendation generated from the What-If Simulator.
    """
    pct = 0.0
    if req.est_cost_before > 0:
        pct = ((req.est_cost_before - req.est_cost_after) / req.est_cost_before) * 100.0

    idx_name = f"idx_{req.target_table}_{'_'.join(req.target_columns)}"[:63]
    runbook_text = generate_index_runbook(
        table_name=req.target_table,
        columns=req.target_columns,
        index_name=idx_name,
        ddl=req.ddl,
        rollback_ddl=req.rollback_ddl,
        source_query=req.source_query,
        rationale=req.rationale or f"Estimated {pct:.1f}% planner cost reduction",
        est_cost_before=req.est_cost_before,
        est_cost_after=req.est_cost_after,
        cost_reduction_pct=pct,
    )

    async with db.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO recommendation (
                kind, target_table, target_columns, ddl, rollback_ddl,
                est_cost_before, est_cost_after, safety_score,
                source_query, rationale, runbook, status, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending', now())
            RETURNING *;
            """,
            req.kind,
            req.target_table,
            req.target_columns,
            req.ddl,
            req.rollback_ddl,
            req.est_cost_before,
            req.est_cost_after,
            req.safety_score,
            req.source_query,
            req.rationale,
            runbook_text,
        )
    return {"message": "Recommendation saved successfully", "recommendation": dict(row)}


@router.get("/{rec_id}/runbook")
async def get_runbook(rec_id: int, db=Depends(get_db)):
    """Retrieve the generated DBA runbook for an index recommendation."""
    async with db.acquire() as conn:
        row = await conn.fetchrow("SELECT runbook, ddl, rollback_ddl, status FROM recommendation WHERE id = $1;", rec_id)
    if not row:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {
        "id": rec_id,
        "runbook": row["runbook"],
        "ddl": row["ddl"],
        "rollback_ddl": row["rollback_ddl"],
        "status": row["status"],
    }


@router.post("/{rec_id}/approve")
async def approve_recommendation(rec_id: int, db=Depends(get_db)):
    """
    Approve an index recommendation.
    NOTE: NEVER auto-applies DDL; marks status as approved and generates runbook.
    """
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE recommendation SET status='approved', approved_by='admin', applied_at=NULL WHERE id=$1 RETURNING *;",
            rec_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {
        "message": "Approved (human approval registered — runbook ready; DDL is NOT auto-applied)",
        "recommendation": dict(row),
    }


@router.post("/{rec_id}/reject")
async def reject_recommendation(rec_id: int, db=Depends(get_db)):
    """Mark an index recommendation as rejected."""
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE recommendation SET status='rejected' WHERE id=$1 RETURNING *;",
            rec_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"message": "Recommendation rejected", "recommendation": dict(row)}

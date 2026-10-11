"""
What-If router — Phase 5: Index simulation, candidate generation, and HypoPG plan comparison.
"""
from __future__ import annotations

import logging
import os
import psycopg2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from whatif.candidates import extract_candidates, IndexCandidate
from whatif.hypopg_eval import evaluate_candidate_hypopg

router = APIRouter()
logger = logging.getLogger(__name__)

TARGET_DB_DSN = os.environ.get(
    "TARGET_DB_DSN",
    "postgresql://targetuser:targetpass@target-db:5432/targetdb",
)


def _get_target_conn():
    return psycopg2.connect(TARGET_DB_DSN)


def _fetch_existing_indexes_and_schema(conn) -> tuple[list[dict], dict[str, list[str]]]:
    """Fetch existing indexes and valid table/columns from target database catalog."""
    cur = conn.cursor()
    # 1. Existing indexes
    cur.execute("""
        SELECT
            t.relname AS table_name,
            i.relname AS index_name,
            array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) AS columns
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
        GROUP BY t.relname, i.relname;
    """)
    indexes = [
        {"table_name": r[0], "index_name": r[1], "columns": list(r[2]) if r[2] else []}
        for r in cur.fetchall()
    ]

    # 2. Valid tables & columns
    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public';
    """)
    schema: dict[str, list[str]] = {}
    for tbl, col in cur.fetchall():
        schema.setdefault(tbl.lower(), []).append(col.lower())

    cur.close()
    return indexes, schema


# ─── Pydantic Models ───────────────────────────────────────────────────

class CandidateRequest(BaseModel):
    query_text: str = Field(..., description="SQL SELECT query to generate index candidates for")


class CandidateResponse(BaseModel):
    table_name: str
    columns: list[str]
    index_type: str
    create_sql: str
    drop_sql: str
    reason: str
    index_name: str


class EvaluateCandidateRequest(BaseModel):
    query_text: str
    table_name: str
    columns: list[str]
    reason: str = ""


class EvaluationResponse(BaseModel):
    table_name: str
    columns: list[str]
    index_name: str
    create_sql: str
    rollback_sql: str
    cost_before: float
    cost_after: float
    cost_reduction_pct: float
    index_used: bool
    hypo_index_name: str
    baseline_plan_summary: str
    candidate_plan_summary: str
    baseline_plan_text: str
    candidate_plan_text: str
    is_conclusive: bool
    warning: str


# ─── Endpoints ─────────────────────────────────────────────────────────

@router.get("/queries")
def get_evaluable_queries(limit: int = 30):
    """
    Fetch active, non-internal SELECT queries from pg_stat_statements in target-db.
    """
    try:
        conn = _get_target_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                queryid,
                query,
                calls,
                total_exec_time,
                mean_exec_time,
                rows,
                shared_blks_hit,
                shared_blks_read
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
              AND query ILIKE 'SELECT%%'
              AND query NOT ILIKE '%%pg_stat%%'
              AND query NOT ILIKE '%%hypopg%%'
              AND query NOT ILIKE '%%collector_state%%'
              AND query NOT ILIKE '%%information_schema%%'
            ORDER BY total_exec_time DESC
            LIMIT %s;
        """, (int(limit),))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        queries = [
            {
                "queryid": r[0],
                "query": r[1],
                "calls": r[2],
                "total_exec_time_ms": round(r[3], 2),
                "mean_exec_time_ms": round(r[4], 2),
                "rows": r[5],
                "shared_blks_hit": r[6],
                "shared_blks_read": r[7],
            }
            for r in rows
        ]
        return {"queries": queries}
    except Exception as exc:
        logger.error("Failed to fetch queries from pg_stat_statements: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database query fetch failed: {exc}")


@router.post("/candidates", response_model=list[CandidateResponse])
def generate_candidates(req: CandidateRequest):
    """
    Parse SQL query, cross-reference against actual database catalog,
    and generate valid non-duplicate index candidates.
    """
    if not req.query_text.strip().upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are supported for index candidate generation.")

    try:
        conn = _get_target_conn()
        existing_idx, schema = _fetch_existing_indexes_and_schema(conn)
        conn.close()

        cands = extract_candidates(
            query=req.query_text,
            existing_indexes=existing_idx,
            valid_schema=schema,
        )

        return [
            CandidateResponse(
                table_name=c.table_name,
                columns=c.columns,
                index_type=c.index_type,
                create_sql=c.create_sql,
                drop_sql=c.drop_sql,
                reason=c.reason,
                index_name=c.index_name,
            )
            for c in cands
        ]
    except Exception as exc:
        logger.error("Candidate generation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Candidate generation error: {exc}")


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_candidate(req: EvaluateCandidateRequest):
    """
    Run isolated HypoPG simulation comparing execution plans before & after.
    """
    if not req.query_text.strip().upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are supported for plan evaluation.")

    try:
        conn = _get_target_conn()
        cand = IndexCandidate(
            table_name=req.table_name,
            columns=req.columns,
            reason=req.reason,
        )
        res = evaluate_candidate_hypopg(conn, cand, req.query_text)
        conn.close()

        return EvaluationResponse(
            table_name=res.candidate.table_name,
            columns=res.candidate.columns,
            index_name=res.candidate.index_name,
            create_sql=res.candidate.create_sql,
            rollback_sql=res.candidate.drop_sql,
            cost_before=round(res.cost_before, 2),
            cost_after=round(res.cost_after, 2),
            cost_reduction_pct=round(res.cost_reduction_pct, 1),
            index_used=res.index_used,
            hypo_index_name=res.hypo_index_name,
            baseline_plan_summary=res.baseline_plan_summary,
            candidate_plan_summary=res.candidate_plan_summary,
            baseline_plan_text=res.baseline_plan_text,
            candidate_plan_text=res.candidate_plan_text,
            is_conclusive=res.is_conclusive,
            warning=res.warning,
        )
    except Exception as exc:
        logger.error("HypoPG evaluation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"HypoPG plan evaluation error: {exc}")

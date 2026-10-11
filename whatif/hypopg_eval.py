"""
hypopg_eval.py — HypoPG-based fast cost estimation (Phase 5c).

Creates a hypothetical index with HypoPG, runs read-only EXPLAIN on queries,
compares costs before and after, captures raw and formatted plans,
and guarantees isolation via hypopg_reset().
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from whatif.candidates import IndexCandidate

logger = logging.getLogger(__name__)


@dataclass
class HypoPGEvaluation:
    candidate: IndexCandidate
    cost_before: float
    cost_after: float
    cost_reduction_pct: float
    index_used: bool
    hypo_index_name: str = ""
    baseline_plan_summary: str = ""
    candidate_plan_summary: str = ""
    baseline_plan_text: str = ""
    candidate_plan_text: str = ""
    raw_baseline_plan: dict = field(default_factory=dict)
    raw_candidate_plan: dict = field(default_factory=dict)
    is_conclusive: bool = True
    warning: str = ""


def _parse_total_cost(plan_json: Any) -> float:
    """Extract total planner cost from EXPLAIN (FORMAT JSON) output."""
    try:
        if isinstance(plan_json, list) and len(plan_json) > 0:
            return float(plan_json[0]["Plan"]["Total Cost"])
        elif isinstance(plan_json, dict) and "Plan" in plan_json:
            return float(plan_json["Plan"]["Total Cost"])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return float("inf")


def _summarize_plan_node(plan_dict: dict) -> str:
    """Recursively summarize primary access paths in the execution plan."""
    if not isinstance(plan_dict, dict):
        return ""
    node_type = plan_dict.get("Node Type", "Unknown")
    rel_name = plan_dict.get("Relation Name") or plan_dict.get("Alias") or ""
    idx_name = plan_dict.get("Index Name", "")

    detail = node_type
    if rel_name:
        detail += f" on {rel_name}"
    if idx_name:
        detail += f" using {idx_name}"

    child_summaries = []
    for sub in plan_dict.get("Plans", []):
        sub_s = _summarize_plan_node(sub)
        if sub_s:
            child_summaries.append(sub_s)

    if child_summaries:
        return f"{detail} -> [{' | '.join(child_summaries)}]"
    return detail


def _prepare_explain_query(sql: str) -> tuple[str, bool]:
    """
    Prepare query for read-only EXPLAIN execution.
    Prefers EXPLAIN (FORMAT JSON, GENERIC_PLAN) if parameters $1, $2 are used.
    If generic plan fails due to syntax or no placeholders, falls back to standard EXPLAIN.
    """
    clean_sql = sql.strip().rstrip(";")

    # Security check: Only allow SELECT queries in EXPLAIN
    if not re.match(r"^\s*SELECT\b", clean_sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries can be evaluated by the What-If index simulator.")

    has_params = bool(re.search(r"\$\d+", clean_sql))
    return clean_sql, has_params


def run_explain(cur, sql: str, fmt: str = "JSON") -> tuple[Any, str]:
    """
    Execute read-only EXPLAIN with GENERIC_PLAN if parameterized, fallback to literal replacement.
    Returns (parsed_data, text_output).
    """
    clean_sql, has_params = _prepare_explain_query(sql)

    # Try GENERIC_PLAN first if query has $1 placeholders
    if has_params:
        try:
            cur.execute(f"EXPLAIN (FORMAT {fmt}, GENERIC_PLAN) {clean_sql}")
            res = cur.fetchall()
            if fmt == "JSON":
                return res[0][0], ""
            return None, "\n".join(r[0] for r in res)
        except Exception:
            # If GENERIC_PLAN failed (e.g. interval arithmetic syntax or nested function), replace placeholders
            cur.connection.rollback()

        # Fallback: substitute placeholders with realistic dummy values for EXPLAIN
        sub_sql = re.sub(r"INTERVAL\s+\$\d+", "INTERVAL '30 days'", clean_sql, flags=re.IGNORECASE)
        sub_sql = re.sub(r"\$\d+", "'1'", sub_sql)
        cur.execute(f"EXPLAIN (FORMAT {fmt}) {sub_sql}")
        res = cur.fetchall()
        if fmt == "JSON":
            return res[0][0], ""
        return None, "\n".join(r[0] for r in res)
    else:
        cur.execute(f"EXPLAIN (FORMAT {fmt}) {clean_sql}")
        res = cur.fetchall()
        if fmt == "JSON":
            return res[0][0], ""
        return None, "\n".join(r[0] for r in res)


def evaluate_candidate_hypopg(
    conn: psycopg2.extensions.connection,
    candidate: IndexCandidate,
    query_text: str,
) -> HypoPGEvaluation:
    """
    Evaluate a candidate index on target connection using HypoPG in complete isolation:
    1. Verify HypoPG is loaded and reset any leftover hypothetical indexes.
    2. Run baseline read-only EXPLAIN (JSON & TEXT).
    3. Create hypothetical index via hypopg_create_index().
    4. Run candidate read-only EXPLAIN (JSON & TEXT).
    5. Calculate cost changes, scan if hypothetical index was used.
    6. Always reset HypoPG in finally block to ensure no pollution.
    """
    cur = conn.cursor()
    hypo_idx_name = ""

    try:
        # Step 1: Clean slate
        cur.execute("SELECT hypopg_reset();")
        conn.commit()

        # Step 2: Baseline plan
        base_json, _ = run_explain(cur, query_text, fmt="JSON")
        _, base_text = run_explain(cur, query_text, fmt="TEXT")
        cost_before = _parse_total_cost(base_json)
        base_root = base_json[0].get("Plan", {}) if isinstance(base_json, list) and base_json else {}
        base_summary = _summarize_plan_node(base_root)

        # Step 3: Create hypothetical index
        # HypoPG doesn't use CONCURRENTLY in its memory structure
        create_stmt = candidate.create_sql.replace("CONCURRENTLY ", "")
        cur.execute("SELECT * FROM hypopg_create_index(%s);", (create_stmt,))
        hypo_row = cur.fetchone()
        if hypo_row:
            hypo_idx_name = str(hypo_row[1])

        # Step 4: Plan with hypothetical index
        cand_json, _ = run_explain(cur, query_text, fmt="JSON")
        _, cand_text = run_explain(cur, query_text, fmt="TEXT")
        cost_after = _parse_total_cost(cand_json)
        cand_root = cand_json[0].get("Plan", {}) if isinstance(cand_json, list) and cand_json else {}
        cand_summary = _summarize_plan_node(cand_root)

        # Step 5: Check usage and compute cost delta
        cand_text_lower = cand_text.lower()
        hypo_clean = hypo_idx_name.lower()
        cand_idx_lower = candidate.index_name.lower()

        index_used = (
            (hypo_clean in cand_text_lower if hypo_clean else False)
            or (cand_idx_lower in cand_text_lower)
            or ("index scan" in cand_summary.lower() and candidate.table_name.lower() in cand_summary.lower())
        )

        if cost_before > 0 and cost_before != float("inf"):
            cost_reduction_pct = ((cost_before - cost_after) / cost_before) * 100.0
        elif cost_before == 0 and cost_after == 0:
            cost_reduction_pct = 0.0
        else:
            cost_reduction_pct = 0.0

        is_conclusive = True
        warning = ""
        if cost_before == 0:
            is_conclusive = False
            warning = "Baseline cost is 0.0; planner estimates are inconclusive."
        elif not index_used and cost_reduction_pct <= 0:
            warning = "The PostgreSQL planner did not select this index for the query."

        return HypoPGEvaluation(
            candidate=candidate,
            cost_before=cost_before,
            cost_after=cost_after,
            cost_reduction_pct=cost_reduction_pct,
            index_used=index_used,
            hypo_index_name=hypo_idx_name,
            baseline_plan_summary=base_summary,
            candidate_plan_summary=cand_summary,
            baseline_plan_text=base_text,
            candidate_plan_text=cand_text,
            raw_baseline_plan=base_root,
            raw_candidate_plan=cand_root,
            is_conclusive=is_conclusive,
            warning=warning,
        )

    finally:
        # Guarantee cleanup
        try:
            cur.execute("SELECT hypopg_reset();")
            conn.commit()
        except Exception:
            pass
        cur.close()

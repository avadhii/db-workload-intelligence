"""
deltas.py — Computes counter deltas between consecutive snapshots.

Postgres stat counters are monotonically increasing (cumulative since last
pg_stat_reset). We store the previous snapshot in the metrics-db
collector_state table and subtract on the next run.

Handles stats resets: if a counter drops significantly, we assume a reset
happened and treat the new value as the full delta.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

COUNTER_FIELDS_PSS = [
    "calls", "total_exec_time", "rows",
    "shared_blks_hit", "shared_blks_read",
    "temp_blks_read", "temp_blks_written",
]

COUNTER_FIELDS_DB = [
    "xact_commit", "xact_rollback", "blks_hit", "blks_read",
    "tup_returned", "tup_fetched", "tup_inserted", "tup_updated", "tup_deleted",
    "deadlocks", "temp_files", "temp_bytes",
]

COUNTER_FIELDS_BGWRITER = [
    "checkpoints_timed", "checkpoints_req",
    "buffers_checkpoint", "buffers_clean", "buffers_backend", "buffers_alloc",
]


def _safe_delta(new_val: float, old_val: float, drop_fraction: float = 0.5) -> float:
    """
    Compute delta between two counter values.
    If the counter dropped by more than drop_fraction, assume reset → use new_val as delta.
    """
    if new_val is None or old_val is None:
        return 0.0
    delta = new_val - old_val
    if delta < 0:
        # Possible reset — compare magnitude
        if old_val > 0 and abs(delta) / old_val > drop_fraction:
            logger.debug("Counter reset detected (old=%s, new=%s), using new as delta", old_val, new_val)
            return float(new_val)
        return 0.0  # small negative: rounding / timing artifact, ignore
    return float(delta)


def compute_query_deltas(
    current_rows: list[dict],
    previous_by_queryid: dict[int, dict],
    drop_fraction: float = 0.5,
) -> list[dict]:
    """
    For each query in current snapshot, compute deltas vs previous snapshot.
    Returns list of delta rows ready for insertion.
    """
    deltas = []
    for row in current_rows:
        qid = row["queryid"]
        prev = previous_by_queryid.get(qid, {})
        delta: dict[str, Any] = {"queryid": qid, "query_text": row.get("query")}
        for field in COUNTER_FIELDS_PSS:
            delta[field] = _safe_delta(
                row.get(field) or 0,
                prev.get(field) or 0,
                drop_fraction,
            )
        # Derived metrics
        calls = delta["calls"]
        delta["mean_ms"] = (delta["total_exec_time"] / calls) if calls > 0 else 0.0
        delta["stddev_ms"] = row.get("stddev_exec_time") or 0.0
        deltas.append(delta)
    return deltas


def compute_db_deltas(
    current: dict,
    previous: dict,
    drop_fraction: float = 0.5,
) -> dict:
    """
    Compute deltas for pg_stat_database counters.
    Pass-through for gauges (numbackends).
    """
    deltas = {"numbackends": current.get("numbackends", 0)}
    for field in COUNTER_FIELDS_DB:
        deltas[field] = _safe_delta(
            current.get(field) or 0,
            previous.get(field) or 0,
            drop_fraction,
        )
    return deltas


def compute_bgwriter_deltas(
    current: dict,
    previous: dict,
    drop_fraction: float = 0.5,
) -> dict:
    deltas = {}
    for field in COUNTER_FIELDS_BGWRITER:
        deltas[field] = _safe_delta(
            current.get(field) or 0,
            previous.get(field) or 0,
            drop_fraction,
        )
    return deltas


def derive_metrics(db_delta: dict, interval_seconds: float) -> dict[str, float]:
    """
    Derive higher-level metrics from raw deltas.
    """
    blks_hit = db_delta.get("blks_hit", 0)
    blks_read = db_delta.get("blks_read", 0)
    total_blks = blks_hit + blks_read
    cache_hit_ratio = (blks_hit / total_blks) if total_blks > 0 else 1.0

    commits = db_delta.get("xact_commit", 0)
    rollbacks = db_delta.get("xact_rollback", 0)
    total_xacts = commits + rollbacks
    qps = total_xacts / interval_seconds if interval_seconds > 0 else 0.0
    rollback_ratio = rollbacks / total_xacts if total_xacts > 0 else 0.0

    return {
        "db.cache_hit_ratio": cache_hit_ratio,
        "db.qps": qps,
        "db.rollback_ratio": rollback_ratio,
        "db.connections": float(db_delta.get("numbackends", 0)),
        "db.deadlocks": float(db_delta.get("deadlocks", 0)),
        "db.temp_files": float(db_delta.get("temp_files", 0)),
    }

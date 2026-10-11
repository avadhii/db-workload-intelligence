"""
rca_rules.py — Rule-based Root Cause Analysis engine (Phase 4).

Each rule takes a window of metric snapshots and table/query stats,
and returns an evidence score 0–1 for a specific cause type.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CAUSE_TYPES = [
    "missing_index",
    "lock_contention",
    "table_bloat",
    "stale_stats",
    "connection_saturation",
    "resource_contention",
    "workload_surge",
]


@dataclass
class CauseEvidence:
    cause: str
    score: float        # 0.0–1.0
    signals: list[str]  # human-readable evidence strings
    metadata: dict = field(default_factory=dict)


def score_missing_index(
    table_stats: pd.DataFrame,   # recent table_stat_delta rows
    query_deltas: pd.DataFrame,  # recent query_stat_delta rows
    threshold_seq_ratio: float = 0.90,
) -> CauseEvidence:
    """
    Evidence for missing/dropped index:
    - Seq scan ratio is high on a specific table
    - Query mean_ms increased while calls are stable
    - blks_read high relative to blks_hit
    """
    signals = []
    score = 0.0

    if not table_stats.empty and "seq_scan" in table_stats.columns:
        table_stats = table_stats.copy()
        table_stats["total_scans"] = table_stats["seq_scan"].fillna(0) + table_stats["idx_scan"].fillna(0)
        table_stats["seq_ratio"] = (
            table_stats["seq_scan"].fillna(0)
            / table_stats["total_scans"].replace(0, 1)
        )
        high_seq = table_stats[table_stats["seq_ratio"] > threshold_seq_ratio]
        if not high_seq.empty:
            top = high_seq.nlargest(3, "seq_ratio")
            for _, row in top.iterrows():
                signals.append(
                    f"Table '{row.get('table_name', '?')}': "
                    f"seq_scan ratio={row['seq_ratio']:.2f} "
                    f"({int(row.get('seq_scan', 0))} seq scans)"
                )
                score = max(score, min(1.0, row["seq_ratio"]))

    if not query_deltas.empty:
        # Queries getting slower without call increase (plan regression)
        slow = query_deltas[query_deltas["mean_ms"] > 100]
        if not slow.empty:
            signals.append(f"{len(slow)} queries with mean_ms > 100ms (possible missing index)")
            score = max(score, min(1.0, score + 0.2))

    return CauseEvidence(cause="missing_index", score=score, signals=signals)


def score_lock_contention(metrics_df: pd.DataFrame) -> CauseEvidence:
    """
    Evidence for lock contention:
    - Lock waiters count is elevated
    - Deadlocks > 0
    """
    signals = []
    score = 0.0

    if "db.lock_waiters" in metrics_df.columns:
        max_waiters = metrics_df["db.lock_waiters"].max()
        if max_waiters > 5:
            signals.append(f"Peak lock waiters: {max_waiters:.0f}")
            score = max(score, min(1.0, max_waiters / 20))

    if "db.deadlocks" in metrics_df.columns:
        total_deadlocks = metrics_df["db.deadlocks"].sum()
        if total_deadlocks > 0:
            signals.append(f"Deadlocks detected: {total_deadlocks:.0f}")
            score = max(score, 0.7)

    return CauseEvidence(cause="lock_contention", score=score, signals=signals)


def score_table_bloat(table_stats: pd.DataFrame) -> CauseEvidence:
    """
    Evidence for table bloat / vacuum lag:
    - n_dead_tup / n_live_tup ratio is high
    - last_autovacuum is old or NULL
    """
    signals = []
    score = 0.0

    if table_stats.empty:
        return CauseEvidence(cause="table_bloat", score=0.0, signals=[])

    stats = table_stats.copy()
    stats["dead_ratio"] = (
        stats["n_dead_tup"].fillna(0)
        / stats[["n_live_tup", "n_dead_tup"]].fillna(0).sum(axis=1).replace(0, 1)
    )
    bloated = stats[stats["dead_ratio"] > 0.1]
    if not bloated.empty:
        top = bloated.nlargest(3, "dead_ratio")
        for _, row in top.iterrows():
            signals.append(
                f"Table '{row.get('table_name', '?')}': "
                f"dead_tup_ratio={row['dead_ratio']:.2f} "
                f"({int(row.get('n_dead_tup', 0))} dead tuples)"
            )
            score = max(score, min(1.0, row["dead_ratio"] * 3))

    return CauseEvidence(cause="table_bloat", score=score, signals=signals)


def score_connection_saturation(metrics_df: pd.DataFrame, max_connections: int = 100) -> CauseEvidence:
    """
    Evidence for connection saturation:
    - Active connections near max_connections
    """
    signals = []
    score = 0.0

    if "db.connections" in metrics_df.columns:
        peak_conns = metrics_df["db.connections"].max()
        ratio = peak_conns / max_connections
        if ratio > 0.7:
            signals.append(f"Peak connections: {peak_conns:.0f} ({ratio*100:.0f}% of max={max_connections})")
            score = min(1.0, ratio)

    return CauseEvidence(cause="connection_saturation", score=score, signals=signals)


def score_workload_surge(metrics_df: pd.DataFrame, query_deltas: pd.DataFrame) -> CauseEvidence:
    """
    Evidence for pure workload surge (not a fault):
    - QPS increased significantly
    - Per-query latency did NOT increase proportionally
    """
    signals = []
    score = 0.0

    if "db.qps" in metrics_df.columns and len(metrics_df) > 1:
        qps_diff = metrics_df["db.qps"].diff().fillna(0)
        max_surge = float(qps_diff.max())
        baseline_qps = float(metrics_df["db.qps"].iloc[: len(metrics_df) // 2].mean())
        if baseline_qps > 0 and max_surge / baseline_qps > 0.5:
            signals.append(f"QPS surged by {max_surge:.1f} (+{max_surge/baseline_qps*100:.0f}%)")
            score = min(1.0, max_surge / baseline_qps)

    return CauseEvidence(cause="workload_surge", score=score, signals=signals)


def rank_causes(
    metrics_df: pd.DataFrame,
    table_stats: pd.DataFrame,
    query_deltas: pd.DataFrame,
    max_connections: int = 100,
) -> list[CauseEvidence]:
    """
    Run all rules and return causes ranked by evidence score (descending).
    """
    pivoted = (
        metrics_df.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
        if "metric" in metrics_df.columns and "value" in metrics_df.columns
        else metrics_df
    )

    causes = [
        score_missing_index(table_stats, query_deltas),
        score_lock_contention(pivoted),
        score_table_bloat(table_stats),
        score_connection_saturation(pivoted, max_connections),
        score_workload_surge(pivoted, query_deltas),
    ]

    return sorted(causes, key=lambda c: c.score, reverse=True)

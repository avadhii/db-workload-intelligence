"""
writer.py — Inserts computed metrics and deltas into the metrics store.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def write_metric_snapshots(cur, ts: datetime, metrics: dict[str, float], dim: str | None = None):
    """Insert scalar metrics into metric_snapshot."""
    rows = [(ts, name, dim, value) for name, value in metrics.items()]
    cur.executemany(
        """
        INSERT INTO metric_snapshot (ts, metric, dim, value)
        VALUES (%s, %s, %s, %s)
        """,
        rows,
    )


def write_query_deltas(cur, ts: datetime, deltas: list[dict]):
    """Insert query-level delta rows into query_stat_delta."""
    rows = [
        (
            ts,
            d["queryid"],
            d.get("query_text", ""),
            d.get("calls", 0),
            d.get("total_exec_time", 0),
            d.get("mean_ms", 0),
            d.get("stddev_ms", 0),
            d.get("rows", 0),
            d.get("shared_blks_hit", 0),
            d.get("shared_blks_read", 0),
            d.get("temp_blks_read", 0),
            d.get("temp_blks_written", 0),
        )
        for d in deltas
    ]
    cur.executemany(
        """
        INSERT INTO query_stat_delta (
            ts, queryid, query_text, calls, total_exec_ms,
            mean_ms, stddev_ms, rows, blks_hit, blks_read,
            temp_blks_read, temp_blks_written
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def write_table_stats(cur, ts: datetime, table_rows: list[dict]):
    """Insert table-level stat snapshots."""
    rows = [
        (
            ts,
            r["schema_name"],
            r["table_name"],
            r.get("seq_scan"),
            r.get("idx_scan"),
            r.get("n_live_tup"),
            r.get("n_dead_tup"),
            r.get("last_autovacuum"),
            r.get("last_autoanalyze"),
        )
        for r in table_rows
    ]
    cur.executemany(
        """
        INSERT INTO table_stat_delta (
            ts, schema_name, table_name, seq_scan, idx_scan,
            n_live_tup, n_dead_tup, last_autovacuum, last_autoanalyze
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def write_index_stats(cur, ts: datetime, index_rows: list[dict]):
    """Insert index-level stat snapshots."""
    rows = [
        (
            ts,
            r["schema_name"],
            r["table_name"],
            r["index_name"],
            r.get("idx_scan"),
            r.get("idx_size_bytes"),
        )
        for r in index_rows
    ]
    cur.executemany(
        """
        INSERT INTO index_stat_snapshot (
            ts, schema_name, table_name, index_name, idx_scan, idx_size_bytes
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def save_collector_state(cur, key: str, state: dict):
    """Upsert collector state (e.g. previous snapshot values)."""
    import json
    cur.execute(
        """
        INSERT INTO collector_state (key, value, updated_at)
        VALUES (%s, %s, now())
        ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                updated_at = EXCLUDED.updated_at
        """,
        (key, json.dumps(state)),
    )


def load_collector_state(cur, key: str) -> dict | None:
    """Load previously saved collector state."""
    import json
    cur.execute("SELECT value FROM collector_state WHERE key = %s", (key,))
    row = cur.fetchone()
    if not row:
        return None
    val = row["value"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]
    if isinstance(val, (dict, list)):
        return val
    return json.loads(val) if val else None

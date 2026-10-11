"""
snapshots.py — Queries pg_stat_* views and returns raw rows.
Each function returns a list of dicts.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg2.extras

logger = logging.getLogger(__name__)


def fetch_pg_stat_statements(cur, max_queries: int = 500) -> list[dict]:
    """
    Fetch top N queries by total_exec_time from pg_stat_statements.
    Returns normalized rows regardless of PG version (14/15/16 compat).
    """
    cur.execute("""
        SELECT
            queryid,
            query,
            calls,
            total_exec_time,
            mean_exec_time,
            stddev_exec_time,
            rows,
            shared_blks_hit,
            shared_blks_read,
            temp_blks_read,
            temp_blks_written
        FROM pg_stat_statements
        WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
        ORDER BY total_exec_time DESC
        LIMIT %s
    """, (max_queries,))
    return [dict(r) for r in cur.fetchall()]


def fetch_pg_stat_database(cur) -> dict:
    """
    Fetch aggregate stats for the current database.
    """
    cur.execute("""
        SELECT
            numbackends,
            xact_commit,
            xact_rollback,
            blks_hit,
            blks_read,
            tup_returned,
            tup_fetched,
            tup_inserted,
            tup_updated,
            tup_deleted,
            deadlocks,
            temp_files,
            temp_bytes
        FROM pg_stat_database
        WHERE datname = current_database()
    """)
    row = cur.fetchone()
    return dict(row) if row else {}


def fetch_pg_stat_user_tables(cur) -> list[dict]:
    """
    Fetch per-table stats for all user tables.
    """
    cur.execute("""
        SELECT
            schemaname      AS schema_name,
            relname         AS table_name,
            seq_scan,
            idx_scan,
            n_live_tup,
            n_dead_tup,
            last_autovacuum,
            last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY seq_scan DESC NULLS LAST
    """)
    return [dict(r) for r in cur.fetchall()]


def fetch_pg_stat_user_indexes(cur) -> list[dict]:
    """
    Fetch per-index stats including size.
    """
    cur.execute("""
        SELECT
            schemaname      AS schema_name,
            relname         AS table_name,
            indexrelname    AS index_name,
            idx_scan,
            pg_relation_size(indexrelid) AS idx_size_bytes
        FROM pg_stat_user_indexes
        ORDER BY idx_scan ASC NULLS FIRST
    """)
    return [dict(r) for r in cur.fetchall()]


def fetch_pg_stat_activity(cur) -> list[dict]:
    """
    Fetch active connections, grouping by state and wait_event_type.
    """
    cur.execute("""
        SELECT
            state,
            wait_event_type,
            wait_event,
            COUNT(*)            AS count,
            MAX(EXTRACT(EPOCH FROM (now() - query_start))) AS max_duration_secs
        FROM pg_stat_activity
        WHERE pid <> pg_backend_pid()
        GROUP BY state, wait_event_type, wait_event
    """)
    return [dict(r) for r in cur.fetchall()]


def fetch_pg_stat_bgwriter(cur) -> dict:
    """
    Fetch background writer / checkpoint stats.
    """
    cur.execute("""
        SELECT
            checkpoints_timed,
            checkpoints_req,
            buffers_checkpoint,
            buffers_clean,
            buffers_backend,
            buffers_alloc
        FROM pg_stat_bgwriter
    """)
    row = cur.fetchone()
    return dict(row) if row else {}

"""
faults.py — Fault injection library for labeled anomaly generation.
Each fault is a context manager (or start/stop pair) that logs an
incident row with source='injected'.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import psycopg2

logger = logging.getLogger(__name__)


def _log_incident(
    metrics_conn: psycopg2.extensions.connection,
    kind: str,
    description: str,
    start_ts: datetime,
    end_ts: datetime | None = None,
):
    """Insert an incident row into the metrics store."""
    cur = metrics_conn.cursor()
    cur.execute(
        """
        INSERT INTO incident (start_ts, end_ts, kind, source, description)
        VALUES (%s, %s, %s, 'injected', %s)
        RETURNING id
        """,
        (start_ts, end_ts, kind, description),
    )
    incident_id = cur.fetchone()[0]
    metrics_conn.commit()
    cur.close()
    logger.info("🔴 Fault injected: %s (incident_id=%d)", kind, incident_id)
    return incident_id


@contextmanager
def drop_index(
    target_conn: psycopg2.extensions.connection,
    metrics_conn: psycopg2.extensions.connection,
    index_name: str,
    table_name: str,
    create_sql: str,
    duration_seconds: float = 120.0,
) -> Generator[None, None, None]:
    """
    Drop a critical index for `duration_seconds`, then recreate it.
    Simulates a missing-index anomaly.
    """
    start = datetime.now(timezone.utc)
    cur = target_conn.cursor()
    incident_id = None
    try:
        logger.warning("Dropping index %s on %s...", index_name, table_name)
        cur.execute(f"DROP INDEX IF EXISTS {index_name}")
        target_conn.commit()
        incident_id = _log_incident(
            metrics_conn, "dropped_index",
            f"Dropped index '{index_name}' on '{table_name}' for {duration_seconds}s",
            start,
        )
        yield
    finally:
        time.sleep(duration_seconds)
        end = datetime.now(timezone.utc)
        logger.info("Recreating index %s...", index_name)
        cur.execute(create_sql)
        target_conn.commit()
        cur.close()
        # Update incident end_ts
        if incident_id:
            m_cur = metrics_conn.cursor()
            m_cur.execute(
                "UPDATE incident SET end_ts = %s WHERE id = %s",
                (end, incident_id),
            )
            metrics_conn.commit()
            m_cur.close()
        logger.info("Index %s recreated", index_name)


@contextmanager
def lock_storm(
    target_conn: psycopg2.extensions.connection,
    metrics_conn: psycopg2.extensions.connection,
    table_name: str,
    duration_seconds: float = 60.0,
) -> Generator[None, None, None]:
    """
    Hold an exclusive lock on a table for `duration_seconds`.
    Simulates lock contention / long-running transaction.
    """
    start = datetime.now(timezone.utc)
    cur = target_conn.cursor()
    incident_id = None
    try:
        logger.warning("Acquiring exclusive lock on %s...", table_name)
        target_conn.autocommit = False
        cur.execute(f"LOCK TABLE {table_name} IN EXCLUSIVE MODE")
        incident_id = _log_incident(
            metrics_conn, "lock_storm",
            f"Exclusive lock on '{table_name}' for {duration_seconds}s",
            start,
        )
        yield
        time.sleep(duration_seconds)
    finally:
        end = datetime.now(timezone.utc)
        target_conn.rollback()
        target_conn.autocommit = True
        cur.close()
        if incident_id:
            m_cur = metrics_conn.cursor()
            m_cur.execute(
                "UPDATE incident SET end_ts = %s WHERE id = %s",
                (end, incident_id),
            )
            metrics_conn.commit()
            m_cur.close()
        logger.info("Lock released on %s", table_name)


def inject_bloat(
    target_conn: psycopg2.extensions.connection,
    metrics_conn: psycopg2.extensions.connection,
    table_name: str,
    batch_size: int = 100_000,
):
    """
    Mass UPDATE without VACUUM to simulate bloat.
    """
    start = datetime.now(timezone.utc)
    cur = target_conn.cursor()
    logger.warning("Injecting bloat on %s (batch_size=%d)...", table_name, batch_size)

    try:
        # Touch rows to create dead tuples
        cur.execute(f"""
            UPDATE {table_name}
            SET updated_at = now()
            WHERE order_id IN (
                SELECT order_id FROM {table_name} ORDER BY random() LIMIT %s
            )
        """, (batch_size,))
        target_conn.commit()
        end = datetime.now(timezone.utc)
        _log_incident(
            metrics_conn, "table_bloat",
            f"Mass UPDATE on '{table_name}' ({batch_size} rows, no VACUUM)",
            start, end,
        )
    except Exception as exc:
        logger.error("Bloat injection failed: %s", exc)
        target_conn.rollback()
    finally:
        cur.close()


def inject_connection_spike(
    target_dsn: str,
    metrics_conn: psycopg2.extensions.connection,
    num_connections: int = 80,
    duration_seconds: float = 90.0,
):
    """
    Open many idle connections simultaneously to saturate connection pool.
    """
    start = datetime.now(timezone.utc)
    logger.warning("Opening %d idle connections for %ds...", num_connections, duration_seconds)

    conns = []
    try:
        for _ in range(num_connections):
            try:
                c = psycopg2.connect(target_dsn)
                c.autocommit = True
                conns.append(c)
            except Exception as e:
                logger.debug("Could not open extra connection: %s", e)
                break

        incident_id = _log_incident(
            metrics_conn, "connection_spike",
            f"Opened {len(conns)} extra idle connections for {duration_seconds}s",
            start,
        )
        time.sleep(duration_seconds)
        end = datetime.now(timezone.utc)
        m_cur = metrics_conn.cursor()
        m_cur.execute(
            "UPDATE incident SET end_ts = %s WHERE id = %s",
            (end, incident_id),
        )
        metrics_conn.commit()
        m_cur.close()
    finally:
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        logger.info("Connection spike ended (%d connections closed)", len(conns))


def inject_stale_stats(
    target_conn: psycopg2.extensions.connection,
    metrics_conn: psycopg2.extensions.connection,
    table_name: str,
):
    start = datetime.now(timezone.utc)
    cur = target_conn.cursor()
    target_table = "event_log_default" if table_name == "event_log" else table_name
    logger.warning("Disabling autovacuum analyze on %s...", target_table)
    try:
        cur.execute(f"ALTER TABLE {target_table} SET (autovacuum_analyze_scale_factor = 0.99)")
        target_conn.commit()
        _log_incident(
            metrics_conn, "stale_stats",
            f"Disabled autovacuum analyze on '{target_table}'; stats will drift",
            start,
        )
    except Exception as exc:
        logger.error("stale_stats injection failed: %s", exc)
        target_conn.rollback()
    finally:
        cur.close()

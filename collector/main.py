"""
main.py — Collector service entry point.
Runs a periodic loop: snapshot → delta → write to metrics store.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from collector.config import CollectorConfig
from collector.snapshots import (
    fetch_pg_stat_statements,
    fetch_pg_stat_database,
    fetch_pg_stat_user_tables,
    fetch_pg_stat_user_indexes,
    fetch_pg_stat_activity,
    fetch_pg_stat_bgwriter,
)
from collector.deltas import (
    compute_query_deltas,
    compute_db_deltas,
    compute_bgwriter_deltas,
    derive_metrics,
)
from collector.writer import (
    write_metric_snapshots,
    write_query_deltas,
    write_table_stats,
    write_index_stats,
    save_collector_state,
    load_collector_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("collector")


def collect_once(
    target_conn: psycopg2.extensions.connection,
    metrics_conn: psycopg2.extensions.connection,
    cfg: CollectorConfig,
    interval: float,
):
    """Perform one collection cycle."""
    ts = datetime.now(timezone.utc)

    # ── 1. Open cursors ──────────────────────────────────────────────
    t_cur = target_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    m_cur = metrics_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── 2. Load previous state from metrics store ────────────────────
    prev_pss   = load_collector_state(m_cur, "pss_prev") or {}
    prev_db    = load_collector_state(m_cur, "db_prev") or {}
    prev_bgw   = load_collector_state(m_cur, "bgw_prev") or {}

    # ── 3. Snapshot target DB ────────────────────────────────────────
    try:
        pss_rows   = fetch_pg_stat_statements(t_cur, cfg.max_queries_tracked)
        db_stats   = fetch_pg_stat_database(t_cur)
        tbl_stats  = fetch_pg_stat_user_tables(t_cur)
        idx_stats  = fetch_pg_stat_user_indexes(t_cur)
        act_stats  = fetch_pg_stat_activity(t_cur)
        bgw_stats  = fetch_pg_stat_bgwriter(t_cur)
    except Exception as exc:
        logger.error("Error fetching from target DB: %s", exc)
        t_cur.close()
        m_cur.close()
        return

    # ── 4. Compute deltas ────────────────────────────────────────────
    is_initial_snapshot = not prev_db
    prev_pss_by_qid = {int(r["queryid"]): r for r in prev_pss} if isinstance(prev_pss, list) else {}
    q_deltas   = compute_query_deltas(pss_rows, prev_pss_by_qid, cfg.reset_detection_drop_fraction)
    db_delta   = compute_db_deltas(db_stats, prev_db, cfg.reset_detection_drop_fraction)
    bgw_delta  = compute_bgwriter_deltas(bgw_stats, prev_bgw, cfg.reset_detection_drop_fraction)
    metrics    = derive_metrics(db_delta, interval)

    # On the very first snapshot after startup, raw counters are cumulative totals, not 30s deltas.
    # Set rate metrics to 0 for the first baseline snapshot to avoid false multi-thousand QPS spikes.
    if is_initial_snapshot:
        metrics["db.qps"] = 0.0
        q_deltas = []

    # Add connection breakdown metrics
    total_waiting = sum(
        int(r.get("count", 0)) for r in act_stats
        if r.get("wait_event_type") == "Lock"
    )
    metrics["db.lock_waiters"] = float(total_waiting)

    # Checkpoint pressure metrics
    metrics["bg.checkpoints_req"] = float(bgw_delta.get("checkpoints_req", 0))
    metrics["bg.buffers_backend"] = float(bgw_delta.get("buffers_backend", 0))

    # ── 5. Write to metrics store ────────────────────────────────────
    write_metric_snapshots(m_cur, ts, metrics)
    write_query_deltas(m_cur, ts, q_deltas)
    write_table_stats(m_cur, ts, tbl_stats)
    write_index_stats(m_cur, ts, idx_stats)

    # ── 6. Save new state ────────────────────────────────────────────
    save_collector_state(m_cur, "pss_prev", pss_rows)
    save_collector_state(m_cur, "db_prev", db_stats)
    save_collector_state(m_cur, "bgw_prev", bgw_stats)

    metrics_conn.commit()

    t_cur.close()
    m_cur.close()

    logger.info(
        "Collected at %s | QPS=%.1f | cache_hit=%.3f | queries=%d",
        ts.isoformat(),
        metrics.get("db.qps", 0),
        metrics.get("db.cache_hit_ratio", 0),
        len(q_deltas),
    )


def run():
    cfg = CollectorConfig.from_env()
    logger.info("Starting collector — interval=%ds", cfg.interval_seconds)

    # Connect with keepalive / auto-reconnect
    target_conn = psycopg2.connect(cfg.target_dsn)
    metrics_conn = psycopg2.connect(cfg.metrics_dsn)
    target_conn.autocommit = True
    metrics_conn.autocommit = False

    interval = float(cfg.interval_seconds)

    while True:
        start = time.monotonic()
        try:
            collect_once(target_conn, metrics_conn, cfg, interval)
        except psycopg2.OperationalError as exc:
            logger.warning("DB connection error: %s — reconnecting...", exc)
            try:
                target_conn = psycopg2.connect(cfg.target_dsn)
                metrics_conn = psycopg2.connect(cfg.metrics_dsn)
                target_conn.autocommit = True
                metrics_conn.autocommit = False
            except Exception as e:
                logger.error("Reconnect failed: %s", e)
        except Exception as exc:
            logger.exception("Unexpected error in collector loop: %s", exc)

        elapsed = time.monotonic() - start
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    run()

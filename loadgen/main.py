"""
main.py — Load generator entry point.

Runs a time-varying workload with:
- Daily sine-wave pattern (peak at 14:00, trough at 03:00)
- Weekly pattern (Mon-Fri 2× weekend)
- Long-term growth trend
- Random bursts
- Scheduled fault injections
"""
from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from loadgen.workload import ALL_QUERIES, WEIGHTS
from loadgen.faults import (
    drop_index,
    inject_bloat,
    inject_connection_spike,
    inject_stale_stats,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("loadgen")

TARGET_DSN  = os.environ["TARGET_DB_DSN"]
METRICS_DSN = os.environ["METRICS_DB_DSN"]

# Base throughput: queries per second at peak
BASE_QPS = float(os.getenv("BASE_QPS", "50"))

# Time compression factor (1.0 = realtime, 24.0 = 1 simulated day per hour)
TIME_COMPRESSION = float(os.getenv("TIME_COMPRESSION", "1.0"))


def throughput_multiplier(sim_time: datetime) -> float:
    """
    Compute a throughput multiplier [0.1, 2.0] based on time-of-day
    and day-of-week patterns.
    """
    hour = sim_time.hour + sim_time.minute / 60.0
    day  = sim_time.weekday()  # 0=Mon, 6=Sun

    # Daily sine wave: peak at 14:00, trough at 3:00
    # Shifted cosine: cos(2π*(hour-14)/24) mapped to [0.1, 1.0]
    daily = 0.55 + 0.45 * math.cos(2 * math.pi * (hour - 14) / 24)

    # Weekly: weekday 1.0, weekend 0.5
    weekly = 1.0 if day < 5 else 0.5

    # Random micro-burst (5% chance): 2× spike
    burst = 2.0 if random.random() < 0.05 else 1.0

    return max(0.1, daily * weekly * burst)


def get_sample_params(query_name: str, cur) -> tuple:
    """Fetch realistic parameter values for each query type from existing records."""
    try:
        if query_name == "customer_by_email":
            cur.execute("SELECT email FROM customer ORDER BY random() LIMIT 1")
            r = cur.fetchone()
            return (r[0],) if r else (None,)

        elif query_name in ("orders_by_customer", "customer_events_seqscan"):
            cur.execute("SELECT customer_id FROM customer ORDER BY random() LIMIT 1")
            r = cur.fetchone()
            return (r[0],) if r else (None,)

        elif query_name == "order_detail":
            cur.execute("SELECT order_id FROM orders ORDER BY random() LIMIT 1")
            r = cur.fetchone()
            return (r[0],) if r else (None,)

        elif query_name == "products_by_category":
            cur.execute("SELECT category_id FROM category ORDER BY random() LIMIT 1")
            r = cur.fetchone()
            return (r[0],) if r else (None,)

        elif query_name == "recent_orders_by_status":
            return (random.choice(["pending", "confirmed", "shipped", "delivered"]),)

        elif query_name == "orders_date_range_seqscan":
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            days_back = random.randint(1, 30)
            return (now - timedelta(days=days_back), now - timedelta(days=days_back - 1))

        elif query_name == "insert_event":
            cur.execute("SELECT customer_id FROM customer ORDER BY random() LIMIT 1")
            r_cust = cur.fetchone()
            cur.execute("SELECT product_id FROM product ORDER BY random() LIMIT 1")
            r_prod = cur.fetchone()
            if not r_cust or not r_prod:
                return ()
            return (r_cust[0], random.choice(["view", "add_to_cart", "search"]), "product", r_prod[0])

        elif query_name == "insert_order":
            cur.execute("SELECT customer_id FROM customer ORDER BY random() LIMIT 1")
            r = cur.fetchone()
            if not r:
                return ()
            return (r[0], round(random.uniform(10, 500), 2))

        elif query_name == "update_order_status":
            cur.execute("SELECT order_id FROM orders WHERE status='pending' ORDER BY random() LIMIT 1")
            r = cur.fetchone()
            if not r:
                return ()
            return (random.choice(["confirmed", "shipped"]), r[0])

        else:
            return ()
    except Exception:
        return ()


def run_query(conn, query, params):
    try:
        cur = conn.cursor()
        cur.execute(query.sql, params)
        if not query.is_write:
            cur.fetchall()
        else:
            conn.commit()
        cur.close()
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.debug("Query %s failed: %s", query.name, exc)


def schedule_faults(target_dsn: str, metrics_dsn: str):
    """Run a fault injection schedule in a background thread."""
    target_conn = psycopg2.connect(target_dsn)
    metrics_conn = psycopg2.connect(metrics_dsn)
    target_conn.autocommit = True
    metrics_conn.autocommit = False

    while True:
        # Wait 5–15 minutes between faults
        delay = random.uniform(300, 900)
        logger.info("Next fault in %.0fs...", delay)
        time.sleep(delay)

        fault_choice = random.choice(["bloat", "connection_spike", "stale_stats"])

        try:
            if fault_choice == "bloat":
                inject_bloat(target_conn, metrics_conn, "orders", batch_size=50_000)

            elif fault_choice == "connection_spike":
                inject_connection_spike(target_dsn, metrics_conn, num_connections=40, duration_seconds=60)

            elif fault_choice == "stale_stats":
                inject_stale_stats(target_conn, metrics_conn, "event_log")

        except Exception as exc:
            logger.error("Fault injection error: %s", exc)


def run():
    logger.info("Starting load generator — BASE_QPS=%.0f, TIME_COMPRESSION=%.1f",
                BASE_QPS, TIME_COMPRESSION)

    conn = psycopg2.connect(TARGET_DSN)
    conn.autocommit = True
    param_cur = conn.cursor()

    # Pre-flight check: ensure target database has required seed data
    logger.info("Checking target database for required seed data...")
    while True:
        try:
            param_cur.execute("SELECT count(*) FROM customer;")
            cnt = param_cur.fetchone()[0]
            if cnt > 0:
                logger.info("Found %d seeded customers. Ready to generate load.", cnt)
                break
        except Exception as e:
            logger.debug("Database not yet seeded: %s", e)
        logger.info("Waiting for database seed data (customer table empty)...")
        time.sleep(3)

    # Start fault scheduler in background
    fault_thread = threading.Thread(
        target=schedule_faults,
        args=(TARGET_DSN, METRICS_DSN),
        daemon=True,
    )
    fault_thread.start()

    cycle = 0
    while True:
        sim_time = datetime.now(timezone.utc)
        multiplier = throughput_multiplier(sim_time)
        target_qps = BASE_QPS * multiplier

        # Choose a query weighted by frequency
        query = random.choices(ALL_QUERIES, weights=WEIGHTS, k=1)[0]
        params = get_sample_params(query.name, param_cur)

        # Reconnect if needed
        try:
            run_query(conn, query, params)
        except psycopg2.OperationalError:
            logger.warning("Reconnecting to target DB...")
            conn = psycopg2.connect(TARGET_DSN)
            conn.autocommit = True
            param_cur = conn.cursor()

        # Sleep to achieve target QPS
        interval = 1.0 / max(target_qps, 0.1)
        time.sleep(interval / TIME_COMPRESSION)

        cycle += 1
        if cycle % 1000 == 0:
            logger.info(
                "cycle=%d | sim_time=%s | multiplier=%.2f | target_qps=%.1f",
                cycle, sim_time.strftime("%H:%M"), multiplier, target_qps,
            )


if __name__ == "__main__":
    run()

"""
workload.py — Defines realistic query patterns that the load generator runs.

Query mix:
- 60% read-heavy (lookups, listings, searches)
- 25% write (new orders, event inserts)
- 10% analytical (aggregations, reports)
-  5% heavy scans (intentionally unindexed, triggers advisor)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import psycopg2


@dataclass
class Query:
    name: str
    sql: str
    weight: float   # relative frequency
    is_write: bool = False


# ── Read queries ───────────────────────────────────────────────────────

CUSTOMER_BY_EMAIL = Query(
    name="customer_by_email",
    sql="SELECT * FROM customer WHERE email = %s",
    weight=10.0,
)

ORDERS_BY_CUSTOMER = Query(
    name="orders_by_customer",
    sql="""
        SELECT o.order_id, o.status, o.total_amount, o.created_at
        FROM orders o
        WHERE o.customer_id = %s
        ORDER BY o.created_at DESC
        LIMIT 20
    """,
    weight=15.0,
)

ORDER_DETAIL = Query(
    name="order_detail",
    sql="""
        SELECT oi.*, p.name, p.price
        FROM order_item oi
        JOIN product p ON oi.product_id = p.product_id
        WHERE oi.order_id = %s
    """,
    weight=8.0,
)

PRODUCTS_BY_CATEGORY = Query(
    name="products_by_category",
    sql="""
        SELECT p.product_id, p.name, p.price, p.stock_quantity
        FROM product p
        WHERE p.category_id = %s AND p.is_active = true
        ORDER BY p.price
        LIMIT 50
    """,
    weight=12.0,
)

RECENT_ORDERS_STATUS = Query(
    name="recent_orders_by_status",
    sql="""
        SELECT order_id, customer_id, total_amount, created_at
        FROM orders
        WHERE status = %s
        ORDER BY created_at DESC
        LIMIT 100
    """,
    weight=8.0,
)

# Intentionally missing index → seq scan on orders.created_at
ORDERS_DATE_RANGE = Query(
    name="orders_date_range_seqscan",
    sql="""
        SELECT COUNT(*), SUM(total_amount)
        FROM orders
        WHERE created_at >= %s AND created_at < %s
    """,
    weight=5.0,
)

# Intentionally missing index → seq scan on event_log
CUSTOMER_EVENTS = Query(
    name="customer_events_seqscan",
    sql="""
        SELECT event_type, COUNT(*), MIN(occurred_at), MAX(occurred_at)
        FROM event_log
        WHERE customer_id = %s
        GROUP BY event_type
    """,
    weight=4.0,
)

# ── Write queries ──────────────────────────────────────────────────────

INSERT_EVENT = Query(
    name="insert_event",
    sql="""
        INSERT INTO event_log (customer_id, event_type, entity_type, entity_id, occurred_at)
        VALUES (%s, %s, %s, %s, now())
    """,
    weight=20.0,
    is_write=True,
)

INSERT_ORDER = Query(
    name="insert_order",
    sql="""
        INSERT INTO orders (customer_id, status, total_amount, created_at, updated_at)
        VALUES (%s, 'pending', %s, now(), now())
        RETURNING order_id
    """,
    weight=5.0,
    is_write=True,
)

UPDATE_ORDER_STATUS = Query(
    name="update_order_status",
    sql="""
        UPDATE orders
        SET status = %s, updated_at = now()
        WHERE order_id = %s
    """,
    weight=5.0,
    is_write=True,
)

# ── Analytical queries ─────────────────────────────────────────────────

REVENUE_BY_CATEGORY = Query(
    name="revenue_by_category",
    sql="""
        SELECT c.name AS category, SUM(oi.quantity * oi.unit_price) AS revenue
        FROM order_item oi
        JOIN product p ON oi.product_id = p.product_id
        JOIN category c ON p.category_id = c.category_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.created_at >= now() - INTERVAL '30 days'
        GROUP BY c.name
        ORDER BY revenue DESC
    """,
    weight=2.0,
)

TOP_PRODUCTS = Query(
    name="top_products_last_week",
    sql="""
        SELECT p.name, SUM(oi.quantity) AS units_sold
        FROM order_item oi
        JOIN product p ON oi.product_id = p.product_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.created_at >= now() - INTERVAL '7 days'
        GROUP BY p.name
        ORDER BY units_sold DESC
        LIMIT 10
    """,
    weight=1.5,
)

ALL_QUERIES = [
    CUSTOMER_BY_EMAIL,
    ORDERS_BY_CUSTOMER,
    ORDER_DETAIL,
    PRODUCTS_BY_CATEGORY,
    RECENT_ORDERS_STATUS,
    ORDERS_DATE_RANGE,
    CUSTOMER_EVENTS,
    INSERT_EVENT,
    INSERT_ORDER,
    UPDATE_ORDER_STATUS,
    REVENUE_BY_CATEGORY,
    TOP_PRODUCTS,
]

WEIGHTS = [q.weight for q in ALL_QUERIES]

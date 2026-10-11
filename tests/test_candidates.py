"""
Unit tests for index candidate generation (Phase 5).
Validates SQL parsing, WHERE/JOIN/ORDER BY column extraction, and candidate creation.
"""
import pytest
from whatif.candidates import extract_candidates, IndexCandidate


class TestExtractCandidates:
    def test_single_where_column(self):
        sql = "SELECT * FROM customer WHERE email = $1"
        candidates = extract_candidates(sql)
        assert len(candidates) >= 1
        assert any(c.table_name == "customer" and c.columns == ["email"] for c in candidates)

    def test_composite_where_and_order_by(self):
        sql = """
            SELECT o.order_id, o.status, o.total_amount
            FROM orders o
            WHERE o.customer_id = $1
            ORDER BY o.created_at DESC
            LIMIT 20
        """
        candidates = extract_candidates(sql)
        # Should generate single-column candidate for customer_id
        col_sets = [c.columns for c in candidates]
        assert ["customer_id"] in col_sets
        # Should also suggest composite [customer_id, created_at]
        assert ["customer_id", "created_at"] in col_sets

    def test_join_condition_extraction(self):
        sql = """
            SELECT oi.*, p.name
            FROM order_item oi
            JOIN product p ON oi.product_id = p.product_id
            WHERE oi.order_id = $1
        """
        candidates = extract_candidates(sql)
        col_sets = [(c.table_name, c.columns) for c in candidates]
        # Should extract order_id on order_item
        assert any("order_id" in c.columns for c in candidates)
        # Should extract product_id
        assert any("product_id" in c.columns for c in candidates)

    def test_create_and_drop_sql_properties(self):
        cand = IndexCandidate(table_name="orders", columns=["customer_id", "created_at"])
        create_sql = cand.create_sql
        drop_sql = cand.drop_sql
        assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_customer_id_created_at" in create_sql
        assert "ON orders (customer_id, created_at)" in create_sql
        assert "DROP INDEX CONCURRENTLY IF EXISTS idx_orders_customer_id_created_at" in drop_sql

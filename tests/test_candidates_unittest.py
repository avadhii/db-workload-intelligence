"""
Tests for index candidate extraction using standard library unittest.
"""
import unittest
from whatif.candidates import extract_candidates, IndexCandidate


class TestCandidateExtraction(unittest.TestCase):
    def test_single_where_column(self):
        sql = "SELECT * FROM customer WHERE email = $1"
        candidates = extract_candidates(sql)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(any(c.table_name == "customer" and c.columns == ["email"] for c in candidates))

    def test_create_and_drop_sql_properties(self):
        cand = IndexCandidate(table_name="orders", columns=["customer_id", "created_at"])
        create_sql = cand.create_sql
        drop_sql = cand.drop_sql
        self.assertIn("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_customer_id_created_at", create_sql)
        self.assertIn("ON orders (customer_id, created_at)", create_sql)
        self.assertIn("DROP INDEX CONCURRENTLY IF EXISTS idx_orders_customer_id_created_at", drop_sql)


if __name__ == "__main__":
    unittest.main()

"""
Unit and integration tests for Phase 5:
- What-If candidate extraction with schema validation and existing index filtering
- HypoPG plan parsing, cost deltas, and isolation
- Recommendation runbook generation and safety rules
"""
import unittest
from whatif.candidates import extract_candidates, IndexCandidate
from whatif.hypopg_eval import _parse_total_cost, _summarize_plan_node
from whatif.runbook import generate_index_runbook


class TestWhatIfPhase5(unittest.TestCase):
    def setUp(self):
        self.sample_schema = {
            "orders": ["order_id", "customer_id", "status", "total_amount", "created_at"],
            "customer": ["customer_id", "email", "first_name", "last_name"],
            "order_item": ["order_item_id", "order_id", "product_id", "quantity", "unit_price"],
            "product": ["product_id", "category_id", "name", "price"],
        }
        self.existing_indexes = [
            {"table_name": "customer", "index_name": "customer_pkey", "columns": ["customer_id"]},
            {"table_name": "customer", "index_name": "customer_email_key", "columns": ["email"]},
            {"table_name": "orders", "index_name": "orders_pkey", "columns": ["order_id"]},
        ]

    def test_candidate_generation_with_filtering(self):
        # Query on customer by email -> email is already indexed, should NOT recommend it
        sql = "SELECT * FROM customer WHERE email = $1"
        cands = extract_candidates(
            sql,
            existing_indexes=self.existing_indexes,
            valid_schema=self.sample_schema,
        )
        self.assertEqual(len(cands), 0)

    def test_candidate_generation_unindexed_column(self):
        # Query on orders by customer_id -> customer_id is unindexed, SHOULD recommend
        sql = "SELECT * FROM orders WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 20"
        cands = extract_candidates(
            sql,
            existing_indexes=self.existing_indexes,
            valid_schema=self.sample_schema,
        )
        self.assertGreaterEqual(len(cands), 1)
        # Should have single-column customer_id
        col_sets = [tuple(c.columns) for c in cands]
        self.assertIn(("customer_id",), col_sets)
        # Should have composite (customer_id, created_at)
        self.assertIn(("customer_id", "created_at"), col_sets)

    def test_invalid_columns_excluded(self):
        # Query with hallucinated or non-existent column
        sql = "SELECT * FROM orders WHERE non_existent_col = $1"
        cands = extract_candidates(
            sql,
            existing_indexes=self.existing_indexes,
            valid_schema=self.sample_schema,
        )
        self.assertEqual(len(cands), 0)

    def test_join_condition_extraction(self):
        sql = "SELECT oi.*, p.name FROM order_item oi JOIN product p ON oi.product_id = p.product_id WHERE oi.order_id = $1"
        cands = extract_candidates(
            sql,
            existing_indexes=self.existing_indexes,
            valid_schema=self.sample_schema,
        )
        table_cols = [(c.table_name, tuple(c.columns)) for c in cands]
        # order_item (order_id) must be recommended
        self.assertIn(("order_item", ("order_id",)), table_cols)

    def test_cost_parsing(self):
        plan_json = [{"Plan": {"Node Type": "Seq Scan", "Total Cost": 356.51}}]
        cost = _parse_total_cost(plan_json)
        self.assertAlmostEqual(cost, 356.51)

        empty_json = []
        cost_empty = _parse_total_cost(empty_json)
        self.assertEqual(cost_empty, float("inf"))

    def test_plan_summary_builder(self):
        plan_dict = {
            "Node Type": "Limit",
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Relation Name": "orders",
                    "Index Name": "idx_orders_customer_id",
                }
            ],
        }
        summary = _summarize_plan_node(plan_dict)
        self.assertIn("Index Scan on orders using idx_orders_customer_id", summary)

    def test_runbook_generation(self):
        runbook = generate_index_runbook(
            table_name="orders",
            columns=["customer_id", "created_at"],
            index_name="idx_orders_customer_id_created_at",
            ddl="CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_customer_id_created_at ON orders (customer_id, created_at);",
            rollback_ddl="DROP INDEX CONCURRENTLY IF EXISTS idx_orders_customer_id_created_at;",
            source_query="SELECT * FROM orders WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 20;",
            rationale="Reduces plan cost by 96.8%",
            est_cost_before=364.78,
            est_cost_after=11.51,
            cost_reduction_pct=96.8,
        )
        self.assertIn("CREATE INDEX CONCURRENTLY", runbook)
        self.assertIn("DROP INDEX CONCURRENTLY", runbook)
        self.assertIn("Safety Checklist", runbook)
        self.assertIn("96.8%", runbook)


if __name__ == "__main__":
    unittest.main()

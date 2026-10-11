"""
Standard library unittest suite for core business logic:
- Collector deltas computation & reset detection
- Safety score calculator & threshold rules
"""
import unittest
from collector.deltas import _safe_delta, compute_query_deltas, derive_metrics
from whatif.candidates import IndexCandidate
from whatif.benchmark import BenchmarkMetrics, BenchmarkReport
from whatif.safety import compute_safety_score


class TestCollectorDeltas(unittest.TestCase):
    def test_normal_increment(self):
        self.assertEqual(_safe_delta(100.0, 80.0), 20.0)

    def test_zero_delta(self):
        self.assertEqual(_safe_delta(50.0, 50.0), 0.0)

    def test_small_negative_clamped(self):
        self.assertEqual(_safe_delta(99.9, 100.0), 0.0)

    def test_counter_reset_recovery(self):
        # Drops from 1000 to 5 -> treat 5 as new delta
        self.assertEqual(_safe_delta(5.0, 1000.0, drop_fraction=0.5), 5.0)

    def test_derive_metrics_cache_hit(self):
        db_delta = {
            "blks_hit": 900,
            "blks_read": 100,
            "xact_commit": 60,
            "xact_rollback": 0,
            "numbackends": 5,
            "deadlocks": 0,
            "temp_files": 0,
        }
        metrics = derive_metrics(db_delta, interval_seconds=30.0)
        self.assertAlmostEqual(metrics["db.cache_hit_ratio"], 0.9)
        self.assertAlmostEqual(metrics["db.qps"], 2.0)


class TestSafetyScoring(unittest.TestCase):
    def setUp(self):
        self.candidate = IndexCandidate(table_name="orders", columns=["customer_id"])

    def test_high_safety_index(self):
        before = BenchmarkMetrics(p50_ms=50, p95_ms=120, p99_ms=150, mean_ms=60, std_ms=10, qps=16)
        after = BenchmarkMetrics(p50_ms=10, p95_ms=25, p99_ms=30, mean_ms=12, std_ms=2, qps=83)
        report = BenchmarkReport(
            candidate=self.candidate,
            before=before,
            after=after,
            p95_improvement_pct=79.1,
            write_overhead_pct=2.1,
            build_time_secs=1.5,
            index_size_mb=4.0,
            is_statistically_significant=True,
            summary="High gain index",
        )
        eval_result = compute_safety_score(report, table_size_mb=100.0)
        self.assertTrue(eval_result.is_safe)
        self.assertGreater(eval_result.safety_score, 70.0)
        self.assertEqual(len(eval_result.rejection_reasons), 0)

    def test_rejected_excessive_write_overhead(self):
        before = BenchmarkMetrics(p50_ms=50, p95_ms=100, p99_ms=120, mean_ms=60, std_ms=5, qps=16)
        after = BenchmarkMetrics(p50_ms=30, p95_ms=70, p99_ms=90, mean_ms=40, std_ms=5, qps=25)
        report = BenchmarkReport(
            candidate=self.candidate,
            before=before,
            after=after,
            p95_improvement_pct=30.0,
            write_overhead_pct=45.0,  # Exceeds 25% threshold
            build_time_secs=2.0,
            index_size_mb=10.0,
            is_statistically_significant=True,
            summary="Slow writes",
        )
        eval_result = compute_safety_score(report, table_size_mb=100.0)
        self.assertFalse(eval_result.is_safe)
        self.assertTrue(any("Write overhead" in r for r in eval_result.rejection_reasons))


if __name__ == "__main__":
    unittest.main()

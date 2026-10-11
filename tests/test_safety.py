"""
Unit tests for safety scoring engine and benchmark reporting (Phase 6).
"""
import pytest
from whatif.candidates import IndexCandidate
from whatif.benchmark import BenchmarkMetrics, BenchmarkReport
from whatif.safety import compute_safety_score


@pytest.fixture
def sample_candidate():
    return IndexCandidate(table_name="orders", columns=["customer_id"])


def test_high_safety_index(sample_candidate):
    before = BenchmarkMetrics(p50_ms=50, p95_ms=120, p99_ms=150, mean_ms=60, std_ms=10, qps=16)
    after = BenchmarkMetrics(p50_ms=10, p95_ms=25, p99_ms=30, mean_ms=12, std_ms=2, qps=83)

    report = BenchmarkReport(
        candidate=sample_candidate,
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
    assert eval_result.is_safe is True
    assert eval_result.safety_score > 70.0
    assert len(eval_result.rejection_reasons) == 0


def test_rejected_high_write_overhead(sample_candidate):
    before = BenchmarkMetrics(p50_ms=50, p95_ms=100, p99_ms=120, mean_ms=60, std_ms=5, qps=16)
    after = BenchmarkMetrics(p50_ms=30, p95_ms=70, p99_ms=90, mean_ms=40, std_ms=5, qps=25)

    report = BenchmarkReport(
        candidate=sample_candidate,
        before=before,
        after=after,
        p95_improvement_pct=30.0,
        write_overhead_pct=45.0,  # Far above 25% threshold
        build_time_secs=2.0,
        index_size_mb=10.0,
        is_statistically_significant=True,
        summary="Slow writes",
    )

    eval_result = compute_safety_score(report, table_size_mb=100.0)
    assert eval_result.is_safe is False
    assert any("Write overhead" in r for r in eval_result.rejection_reasons)

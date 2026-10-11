"""
Unit tests for collector delta computation.
Tests the core delta logic including stats reset detection.
"""
import pytest
from collector.deltas import (
    _safe_delta,
    compute_query_deltas,
    compute_db_deltas,
    derive_metrics,
)


class TestSafeDelta:
    def test_normal_increment(self):
        assert _safe_delta(100.0, 80.0) == 20.0

    def test_zero_delta(self):
        assert _safe_delta(50.0, 50.0) == 0.0

    def test_small_negative_returns_zero(self):
        # Timing artifact — small negative should be clamped to 0
        assert _safe_delta(99.9, 100.0) == 0.0

    def test_counter_reset_large_drop(self):
        # Counter dropped from 1000 to 5 → assume reset, delta = new_val
        result = _safe_delta(5.0, 1000.0, drop_fraction=0.5)
        assert result == 5.0

    def test_none_values(self):
        assert _safe_delta(None, 100.0) == 0.0
        assert _safe_delta(100.0, None) == 0.0


class TestComputeQueryDeltas:
    def test_basic_delta(self):
        current = [
            {
                "queryid": 1,
                "query": "SELECT 1",
                "calls": 110,
                "total_exec_time": 550.0,
                "rows": 110,
                "shared_blks_hit": 500,
                "shared_blks_read": 10,
                "temp_blks_read": 0,
                "temp_blks_written": 0,
                "stddev_exec_time": 0.5,
            }
        ]
        previous = {1: {"calls": 100, "total_exec_time": 500.0, "rows": 100,
                         "shared_blks_hit": 450, "shared_blks_read": 5,
                         "temp_blks_read": 0, "temp_blks_written": 0}}

        deltas = compute_query_deltas(current, previous)
        assert len(deltas) == 1
        d = deltas[0]
        assert d["calls"] == 10
        assert d["total_exec_time"] == 50.0
        assert d["mean_ms"] == pytest.approx(5.0, rel=1e-3)

    def test_new_query_no_previous(self):
        current = [{"queryid": 99, "query": "SELECT 2", "calls": 50,
                     "total_exec_time": 100.0, "rows": 50,
                     "shared_blks_hit": 100, "shared_blks_read": 5,
                     "temp_blks_read": 0, "temp_blks_written": 0,
                     "stddev_exec_time": 0}]
        deltas = compute_query_deltas(current, {})
        assert deltas[0]["calls"] == 50.0


class TestDerivMetrics:
    def test_cache_hit_ratio(self):
        db_delta = {"blks_hit": 900, "blks_read": 100, "xact_commit": 60,
                     "xact_rollback": 0, "numbackends": 5, "deadlocks": 0,
                     "temp_files": 0}
        m = derive_metrics(db_delta, interval_seconds=30.0)
        assert m["db.cache_hit_ratio"] == pytest.approx(0.9)
        assert m["db.qps"] == pytest.approx(2.0)

    def test_zero_blocks(self):
        db_delta = {"blks_hit": 0, "blks_read": 0, "xact_commit": 0,
                     "xact_rollback": 0, "numbackends": 1, "deadlocks": 0,
                     "temp_files": 0}
        m = derive_metrics(db_delta, interval_seconds=30.0)
        assert m["db.cache_hit_ratio"] == 1.0   # default when no I/O
        assert m["db.qps"] == 0.0

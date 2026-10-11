"""
benchmark.py — Before/after benchmark protocol for index validation (Phase 5c).

Protocol:
1. Warm cache on target workload queries.
2. Run N=5 iterations before index creation (measure p50, p95, p99, buffers).
3. CREATE INDEX CONCURRENTLY (record build time + index size).
4. ANALYZE table.
5. Re-run identical benchmark iterations.
6. Measure write overhead (INSERT/UPDATE benchmarks).
7. Flag inconclusive tests if variance overlaps.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from whatif.candidates import IndexCandidate

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    qps: float
    raw_latencies: list[float] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    candidate: IndexCandidate
    before: BenchmarkMetrics
    after: BenchmarkMetrics
    p95_improvement_pct: float
    write_overhead_pct: float
    build_time_secs: float
    index_size_mb: float
    is_statistically_significant: bool
    summary: str


def _measure_queries(
    conn: psycopg2.extensions.connection,
    queries: list[str],
    iterations: int = 5,
) -> BenchmarkMetrics:
    """Runs a series of queries for multiple iterations and computes latency percentiles."""
    latencies: list[float] = []
    cur = conn.cursor()

    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            try:
                cur.execute(q)
                cur.fetchall()
            except Exception as e:
                logger.debug("Benchmark query failed: %s", e)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(dt)

    cur.close()

    if latencies:
        if HAS_NUMPY:
            arr = np.array(latencies)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            mean = float(np.mean(arr))
            std = float(np.std(arr))
        else:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[int(0.50 * n)]
            p95 = sorted_lat[min(n - 1, int(0.95 * n))]
            p99 = sorted_lat[min(n - 1, int(0.99 * n))]
            mean = sum(latencies) / n
            var = sum((x - mean) ** 2 for x in latencies) / n
            std = var ** 0.5
    else:
        p50 = p95 = p99 = mean = std = 0.0

    qps = 1000.0 / mean if mean > 0 else 0.0

    return BenchmarkMetrics(
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std,
        qps=qps,
        raw_latencies=latencies,
    )


def _measure_write_workload(
    conn: psycopg2.extensions.connection,
    table_name: str,
    n_writes: int = 100,
) -> float:
    """Measures avg latency in ms of lightweight write operations."""
    cur = conn.cursor()
    latencies = []
    # Generic benchmark write test
    for _ in range(n_writes):
        t0 = time.perf_counter()
        try:
            cur.execute(f"UPDATE {table_name} SET updated_at = now() WHERE ctid IN (SELECT ctid FROM {table_name} LIMIT 1)")
            conn.commit()
        except Exception:
            conn.rollback()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
    cur.close()
    return float(np.mean(latencies)) if latencies else 0.0


def run_before_after_benchmark(
    sandbox_dsn: str,
    candidate: IndexCandidate,
    test_queries: list[str],
    n_runs: int = 5,
) -> BenchmarkReport:
    """Full execution of the before/after benchmark protocol."""
    conn = psycopg2.connect(sandbox_dsn)
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Warm cache
    logger.info("Warming cache in sandbox...")
    for q in test_queries[:3]:
        try:
            cur.execute(q)
            cur.fetchall()
        except Exception:
            pass

    # 2. Before benchmark
    logger.info("Running baseline benchmark (N=%d)...", n_runs)
    before_metrics = _measure_queries(conn, test_queries, iterations=n_runs)
    write_before = _measure_write_workload(conn, candidate.table_name, n_writes=50)

    # 3. Create index concurrently and record metrics
    logger.info("Applying candidate index: %s", candidate.create_sql)
    t_start = time.perf_counter()
    cur.execute(candidate.create_sql)
    build_time = time.perf_counter() - t_start

    # Get index size
    idx_name = f"idx_{candidate.table_name}_{'_'.join(candidate.columns)}"[:63]
    cur.execute("SELECT pg_relation_size(%s)", (idx_name,))
    size_bytes = cur.fetchone()[0] or 0
    size_mb = size_bytes / (1024.0 * 1024.0)

    # 4. ANALYZE & re-warm
    cur.execute(f"ANALYZE {candidate.table_name}")
    for q in test_queries[:3]:
        try:
            cur.execute(q)
            cur.fetchall()
        except Exception:
            pass

    # 5. After benchmark
    logger.info("Running post-index benchmark (N=%d)...", n_runs)
    after_metrics = _measure_queries(conn, test_queries, iterations=n_runs)
    write_after = _measure_write_workload(conn, candidate.table_name, n_writes=50)

    # 6. Compute deltas & variance overlap
    p95_imp = (
        (before_metrics.p95_ms - after_metrics.p95_ms) / max(before_metrics.p95_ms, 1e-6) * 100.0
    )
    write_overhead = (
        (write_after - write_before) / max(write_before, 1e-6) * 100.0
        if write_before > 0 else 0.0
    )

    # Statistically significant if after p95 + std is still strictly less than before p95 - std
    is_significant = (after_metrics.mean_ms + after_metrics.std_ms) < (before_metrics.mean_ms - before_metrics.std_ms)

    summary = (
        f"Index {idx_name}: p95 changed from {before_metrics.p95_ms:.2f}ms to {after_metrics.p95_ms:.2f}ms "
        f"({p95_imp:+.1f}%). Write overhead: {write_overhead:+.1f}%. Size: {size_mb:.2f}MB."
    )

    cur.close()
    conn.close()

    return BenchmarkReport(
        candidate=candidate,
        before=before_metrics,
        after=after_metrics,
        p95_improvement_pct=p95_imp,
        write_overhead_pct=write_overhead,
        build_time_secs=build_time,
        index_size_mb=size_mb,
        is_statistically_significant=is_significant,
        summary=summary,
    )

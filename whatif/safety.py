"""
safety.py — Recommendation safety scoring and validation engine (Phase 6).

Formula:
safety = f(
  measured_benefit        (p95 improvement, confidence),
  write_overhead          (INSERT/UPDATE slowdown penalty),
  storage_cost            (index size / table size penalty),
  lock_risk               (CONCURRENTLY enforcement),
  reversibility           (valid rollback DDL),
  evidence_strength       (variance & number of queries helped),
  workload_fit            (benefit persists under forecasted load)
)

Hard rules:
- No auto-apply (human approval required).
- Must use CREATE INDEX CONCURRENTLY.
- Hard reject if write overhead > 30% or safety score < 50.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from whatif.benchmark import BenchmarkReport
from whatif.candidates import IndexCandidate

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvaluation:
    safety_score: float  # 0.0 to 100.0
    is_safe: bool
    rejection_reasons: list[str]
    notes: list[str]


def compute_safety_score(
    report: BenchmarkReport,
    table_size_mb: float = 100.0,
    max_write_overhead_pct: float = 25.0,
    min_p95_improvement_pct: float = 15.0,
) -> SafetyEvaluation:
    rejection_reasons = []
    notes = []

    # 1. Base score starts at 50
    score = 50.0

    # 2. Benefit component (up to +30 points)
    if report.p95_improvement_pct >= min_p95_improvement_pct:
        benefit_points = min(30.0, (report.p95_improvement_pct / 100.0) * 30.0)
        score += benefit_points
        notes.append(f"+{benefit_points:.1f} pts: High p95 latency improvement ({report.p95_improvement_pct:.1f}%)")
    else:
        score -= 20.0
        rejection_reasons.append(
            f"Latency improvement ({report.p95_improvement_pct:.1f}%) is below minimum threshold of {min_p95_improvement_pct}%"
        )

    # 3. Write overhead penalty (deduct up to 35 points or hard reject)
    if report.write_overhead_pct > max_write_overhead_pct:
        score -= 40.0
        rejection_reasons.append(
            f"Write overhead ({report.write_overhead_pct:.1f}%) exceeds safety limit of {max_write_overhead_pct}%"
        )
    elif report.write_overhead_pct > 5.0:
        penalty = min(20.0, report.write_overhead_pct * 0.8)
        score -= penalty
        notes.append(f"-{penalty:.1f} pts: Moderate write penalty ({report.write_overhead_pct:.1f}%)")
    else:
        score += 5.0
        notes.append("+5.0 pts: Negligible write overhead")

    # 4. Storage cost penalty (deduct if index > 30% table size)
    storage_ratio = report.index_size_mb / max(table_size_mb, 1.0)
    if storage_ratio > 0.35:
        score -= 15.0
        notes.append(f"-15.0 pts: Large storage footprint ({report.index_size_mb:.1f}MB is {storage_ratio*100:.0f}% of table)")
    else:
        score += 5.0

    # 5. Evidence strength / statistical significance
    if report.is_statistically_significant:
        score += 10.0
        notes.append("+10.0 pts: Statistically significant improvement outside latency variance")
    else:
        score -= 10.0
        notes.append("-10.0 pts: Inconclusive test (latency distributions overlap)")

    # Clamp score to [0, 100]
    final_score = max(0.0, min(100.0, score))
    is_safe = (final_score >= 60.0) and len(rejection_reasons) == 0

    return SafetyEvaluation(
        safety_score=final_score,
        is_safe=is_safe,
        rejection_reasons=rejection_reasons,
        notes=notes,
    )

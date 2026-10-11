"""Incidents router — includes a lightweight metric-correlation RCA baseline."""
from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException

from api.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

# Metrics correlated against incident timestamps
_METRICS = [
    "db.qps",
    "db.connections",
    "db.deadlocks",
    "db.cache_hit_ratio",
    "db.rollback_ratio",
    "db.temp_files",
]

# Human-readable explanations for each metric when it spikes / drops
_METRIC_LABELS = {
    "db.qps":             ("QPS", "query throughput"),
    "db.connections":     ("Connections", "active client connections"),
    "db.deadlocks":       ("Deadlocks", "deadlock events"),
    "db.cache_hit_ratio": ("Cache Hit Ratio", "buffer pool efficiency"),
    "db.rollback_ratio":  ("Rollback Ratio", "transaction rollback rate"),
    "db.temp_files":      ("Temp Files", "temp-file I/O spills"),
}

# Thresholds that signal a meaningful change (relative change fraction)
_THRESHOLD = {
    "db.qps":             0.20,   # 20% drop or spike
    "db.connections":     0.30,   # 30% spike
    "db.deadlocks":       0.0,    # any non-zero value after zero baseline is notable
    "db.cache_hit_ratio": 0.05,   # 5-point drop
    "db.rollback_ratio":  0.10,
    "db.temp_files":      0.0,
}

# RCA narrative templates per metric + direction
_TEMPLATES = {
    "db.qps": {
        "up":   "QPS rose sharply ({before:.1f} → {after:.1f}), indicating a workload surge that may have overwhelmed resources.",
        "down": "QPS dropped significantly ({before:.1f} → {after:.1f}), suggesting query failures, connection refusals, or workload drain.",
    },
    "db.connections": {
        "up":   "Active connections spiked ({before:.1f} → {after:.1f}), consistent with a connection pool exhaustion or storm.",
        "down": "Connection count dropped ({before:.1f} → {after:.1f}), possibly due to forced disconnections or server restart.",
    },
    "db.deadlocks": {
        "up":   "Deadlocks increased ({before:.1f} → {after:.1f}), indicating concurrent write contention on shared rows.",
        "down": "",
    },
    "db.cache_hit_ratio": {
        "up":   "",
        "down": "Cache hit ratio degraded ({before:.3f} → {after:.3f}), meaning the working set no longer fits in shared_buffers — expect higher I/O latency.",
    },
    "db.rollback_ratio": {
        "up":   "Rollback ratio increased ({before:.3f} → {after:.3f}), indicating application errors, constraint violations, or lock timeouts.",
        "down": "",
    },
    "db.temp_files": {
        "up":   "Temp-file spills increased ({before:.1f} → {after:.1f}), suggesting sort/hash operations exceeding work_mem.",
        "down": "",
    },
}


@router.get("/{incident_id}")
async def get_incident(incident_id: int, db=Depends(get_db)):
    async with db.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM incident WHERE id = $1", incident_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return dict(row)


@router.get("/{incident_id}/causes")
async def get_incident_causes(incident_id: int, db=Depends(get_db)):
    """
    Lightweight RCA baseline:
    1. Load the incident record (timestamp, kind, source, description).
    2. Query metric_snapshot for a ±30-min window around incident start.
    3. Compare pre-incident average vs at/after-incident average for each metric.
    4. Surface metrics that changed beyond their threshold as evidence.
    5. Return structured findings — no fabrication, honest 'insufficient evidence' if none found.
    """
    async with db.acquire() as conn:
        incident = await conn.fetchrow("SELECT * FROM incident WHERE id = $1", incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")

        incident = dict(incident)
        start_ts = incident["start_ts"]

        window_before_start = start_ts - timedelta(minutes=30)
        window_after_end    = start_ts + timedelta(minutes=30)

        findings = []
        metric_snapshots: dict[str, dict] = {}

        for metric in _METRICS:
            # Before window: 30 min prior
            before_rows = await conn.fetch(
                """
                SELECT AVG(value) AS avg_val
                FROM metric_snapshot
                WHERE metric = $1
                  AND ts >= $2 AND ts < $3
                """,
                metric, window_before_start, start_ts,
            )
            # After window: 30 min from incident start
            after_rows = await conn.fetch(
                """
                SELECT AVG(value) AS avg_val
                FROM metric_snapshot
                WHERE metric = $1
                  AND ts >= $2 AND ts <= $3
                """,
                metric, start_ts, window_after_end,
            )

            before_val = before_rows[0]["avg_val"] if before_rows and before_rows[0]["avg_val"] is not None else None
            after_val  = after_rows[0]["avg_val"]  if after_rows  and after_rows[0]["avg_val"]  is not None else None

            metric_snapshots[metric] = {
                "before": round(before_val, 4) if before_val is not None else None,
                "after":  round(after_val,  4) if after_val  is not None else None,
            }

            if before_val is None or after_val is None:
                continue

            threshold = _THRESHOLD.get(metric, 0.10)
            label, desc = _METRIC_LABELS.get(metric, (metric, metric))

            if metric == "db.deadlocks":
                # Any non-zero after-value when before was ~0 is significant
                if before_val < 0.5 and after_val >= 1.0:
                    tmpl = _TEMPLATES[metric]["up"]
                    findings.append({
                        "metric": metric,
                        "label": label,
                        "direction": "up",
                        "before": round(before_val, 4),
                        "after":  round(after_val,  4),
                        "explanation": tmpl.format(before=before_val, after=after_val),
                    })
                continue

            if metric == "db.temp_files":
                if before_val < 0.5 and after_val >= 1.0:
                    tmpl = _TEMPLATES[metric]["up"]
                    findings.append({
                        "metric": metric,
                        "label": label,
                        "direction": "up",
                        "before": round(before_val, 4),
                        "after":  round(after_val,  4),
                        "explanation": tmpl.format(before=before_val, after=after_val),
                    })
                continue

            # Relative change — guard against near-zero base producing misleading percentages.
            # For ratio metrics (values <= 1.0) with a near-zero before, use absolute-delta check.
            abs_delta = abs(after_val - before_val)
            if abs(before_val) < 0.001:
                # Near-zero baseline: only flag if the absolute change is meaningful
                if abs_delta < threshold:
                    continue
                rel_change = 1.0 if after_val > before_val else -1.0  # represent as ±100%
            else:
                rel_change = (after_val - before_val) / abs(before_val)
                if abs(rel_change) < threshold:
                    continue

            direction = "up" if rel_change > 0 else "down"
            tmpl = _TEMPLATES.get(metric, {}).get(direction, "")
            if not tmpl:
                continue  # No template for this direction → skip (e.g. QPS going up is not RCA)

            findings.append({
                "metric": metric,
                "label": label,
                "direction": direction,
                "before": round(before_val, 4),
                "after":  round(after_val,  4),
                "change_pct": round(rel_change * 100, 1),
                "explanation": tmpl.format(before=before_val, after=after_val),
            })

    # Sort by descending absolute change significance
    findings.sort(key=lambda f: abs(f.get("change_pct", 100)), reverse=True)

    source_note = (
        "This incident was **injected** by the load generator as a controlled fault."
        if incident.get("source") == "injected"
        else "This incident was **detected automatically** by the collector."
    )

    return {
        "incident_id": incident_id,
        "kind": incident.get("kind"),
        "source": incident.get("source"),
        "source_note": source_note,
        "description": incident.get("description"),
        "start_ts": str(incident.get("start_ts")),
        "end_ts": str(incident.get("end_ts")) if incident.get("end_ts") else None,
        "metric_snapshots": metric_snapshots,
        "findings": findings,
        "sufficient_evidence": len(findings) > 0,
        "summary": (
            f"Found {len(findings)} metric signal(s) correlated with this incident."
            if findings
            else "No significant metric changes detected in the ±30-minute window. "
                 "This may be an injected fault with no measurable metric footprint, "
                 "or metric data is sparse around this timestamp."
        ),
    }

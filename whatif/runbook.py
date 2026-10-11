"""
runbook.py — Generates DDL and operational rollback runbooks for index recommendations.

Enforces Safety Model:
- Recommendations are never auto-applied.
- Approval generates an operational runbook with DDL + rollback SQL.
- Notes rollback impact and lock implications (CONCURRENTLY).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def generate_index_runbook(
    table_name: str,
    columns: list[str],
    index_name: str,
    ddl: str,
    rollback_ddl: str,
    source_query: str,
    rationale: str,
    est_cost_before: float,
    est_cost_after: float,
    cost_reduction_pct: float,
) -> str:
    """Generate a markdown operational runbook for safe DBA review and deployment."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cols_formatted = ", ".join(columns)

    return f"""# Database Change Runbook: Index Recommendation

**Target Table:** `{table_name}`  
**Target Columns:** `({cols_formatted})`  
**Generated At:** `{ts}`  
**Planner Estimated Cost Change:** `{est_cost_before:.2f}` → `{est_cost_after:.2f}` (`{cost_reduction_pct:+.1f}%`)

---

## 1. Executive Summary & Rationale
{rationale}

### Source Query
```sql
{source_query}
```

---

## 2. Pre-Deployment Safety Checklist
- [ ] **Locking Strategy:** The proposed DDL uses `CREATE INDEX CONCURRENTLY` to avoid table write locks.
- [ ] **Maintenance Window:** Recommended deployment during low-traffic hours to prevent IO contention.
- [ ] **Disk Space Verification:** Ensure sufficient disk space is available for the new index.
- [ ] **Workload Fit:** While planner cost shows an estimated reduction of {cost_reduction_pct:.1f}%, measure actual query latency before/after in staging.

---

## 3. Deployment DDL
Execute in a `psql` session with autocommit or outside a transaction block:

```sql
{ddl}
```

---

## 4. Post-Deployment Verification
```sql
-- Check index existence and valid status
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS index_size, indisvalid
FROM pg_index ix
JOIN pg_class c ON ix.indexrelid = c.oid
JOIN pg_class t ON ix.indrelid = t.oid
WHERE t.relname = '{table_name}' AND c.relname = '{index_name}';

-- Update table statistics
ANALYZE {table_name};
```

---

## 5. Rollback Plan
If performance degrades or write overhead exceeds tolerance:

> **Warning:** Dropping this index removes optimization for the source query. Ensure rollback is reviewed prior to execution.

```sql
{rollback_ddl}
```
"""

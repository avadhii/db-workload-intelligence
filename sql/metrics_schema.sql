-- ════════════════════════════════════════════════════════════════
-- Metrics Store Schema
-- All telemetry collected from the target DB is stored here.
-- ════════════════════════════════════════════════════════════════

-- Note: pg_stat_statements is installed on target-db only (via infra/postgres/init.sql).
-- The metrics store does not need it.


-- ─── Generic metric time series ──────────────────────────────────────
-- Stores scalar metrics (cache hit ratio, QPS, connection count, etc.)
CREATE TABLE IF NOT EXISTS metric_snapshot (
    ts      timestamptz NOT NULL,
    metric  text        NOT NULL,   -- e.g. 'db.cache_hit_ratio', 'db.qps'
    dim     text,                   -- optional dimension: queryid, table name, etc.
    value   double precision NOT NULL
) PARTITION BY RANGE (ts);

-- Create partitions for current + next month (extend via maintenance job)
CREATE TABLE IF NOT EXISTS metric_snapshot_default
    PARTITION OF metric_snapshot DEFAULT;

-- ─── Query-level deltas from pg_stat_statements ───────────────────────
CREATE TABLE IF NOT EXISTS query_stat_delta (
    ts              timestamptz      NOT NULL,
    queryid         bigint           NOT NULL,
    query_text      text,                       -- normalized query (first capture)
    calls           bigint,
    total_exec_ms   double precision,
    mean_ms         double precision,
    stddev_ms       double precision,
    rows            bigint,
    blks_hit        bigint,
    blks_read       bigint,
    temp_blks_read  bigint,
    temp_blks_written bigint
) PARTITION BY RANGE (ts);

CREATE TABLE IF NOT EXISTS query_stat_delta_default
    PARTITION OF query_stat_delta DEFAULT;

-- ─── Table-level stats deltas ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS table_stat_delta (
    ts              timestamptz NOT NULL,
    schema_name     text        NOT NULL,
    table_name      text        NOT NULL,
    seq_scan        bigint,
    idx_scan        bigint,
    n_live_tup      bigint,
    n_dead_tup      bigint,
    last_autovacuum timestamptz,
    last_autoanalyze timestamptz
) PARTITION BY RANGE (ts);

CREATE TABLE IF NOT EXISTS table_stat_delta_default
    PARTITION OF table_stat_delta DEFAULT;

-- ─── Index-level stats ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS index_stat_snapshot (
    ts          timestamptz NOT NULL,
    schema_name text        NOT NULL,
    table_name  text        NOT NULL,
    index_name  text        NOT NULL,
    idx_scan    bigint,
    idx_size_bytes bigint
) PARTITION BY RANGE (ts);

CREATE TABLE IF NOT EXISTS index_stat_snapshot_default
    PARTITION OF index_stat_snapshot DEFAULT;

-- ─── Incidents (injected faults + detected anomalies) ─────────────────
CREATE TABLE IF NOT EXISTS incident (
    id              bigserial   PRIMARY KEY,
    start_ts        timestamptz NOT NULL,
    end_ts          timestamptz,
    kind            text        NOT NULL,   -- e.g. 'dropped_index', 'lock_storm', 'bloat'
    source          text        NOT NULL,   -- 'injected' | 'detected'
    severity        text,                   -- 'low' | 'medium' | 'high'
    description     text,
    suspected_causes jsonb,
    resolved        boolean     DEFAULT false,
    created_at      timestamptz DEFAULT now()
);

-- ─── Recommendations ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recommendation (
    id                      bigserial   PRIMARY KEY,
    created_at              timestamptz DEFAULT now(),
    kind                    text        NOT NULL,   -- 'create_index', 'drop_index', 'vacuum', etc.
    target_table            text,
    target_columns          text[],
    ddl                     text,                   -- SQL to apply
    rollback_ddl            text,                   -- SQL to undo
    -- Cost-based estimates (HypoPG)
    est_cost_before         double precision,
    est_cost_after          double precision,
    -- Sandbox measurements
    measured_p50_before_ms  double precision,
    measured_p95_before_ms  double precision,
    measured_p99_before_ms  double precision,
    measured_p50_after_ms   double precision,
    measured_p95_after_ms   double precision,
    measured_p99_after_ms   double precision,
    -- Overheads
    write_overhead_pct      double precision,
    index_size_mb           double precision,
    -- Safety
    safety_score            double precision,       -- 0–100
    safety_notes            jsonb,
    source_query            text,
    rationale               text,
    runbook                 text,
    -- Status
    status                  text DEFAULT 'pending', -- pending | approved | rejected | applied
    approved_by             text,
    applied_at              timestamptz,
    incident_id             bigint REFERENCES incident(id)
);

-- ─── Collector State (tracks last-seen counter values for deltas) ─────
CREATE TABLE IF NOT EXISTS collector_state (
    key     text PRIMARY KEY,
    value   jsonb NOT NULL,
    updated_at timestamptz DEFAULT now()
);

-- ─── Indexes ──────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_metric_snapshot_ts      ON metric_snapshot (ts);
CREATE INDEX IF NOT EXISTS idx_metric_snapshot_metric  ON metric_snapshot (metric, ts);
CREATE INDEX IF NOT EXISTS idx_query_stat_delta_ts     ON query_stat_delta (ts);
CREATE INDEX IF NOT EXISTS idx_query_stat_delta_qid    ON query_stat_delta (queryid, ts);
CREATE INDEX IF NOT EXISTS idx_incident_start_ts       ON incident (start_ts);
CREATE INDEX IF NOT EXISTS idx_recommendation_created  ON recommendation (created_at);

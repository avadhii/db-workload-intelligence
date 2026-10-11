"""
Streamlit Dashboard — DB Workload Intelligence
Pages: Overview · Forecast · Anomalies & Causes · What-If Lab · Recommendations
"""
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DB Workload Intelligence",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar navigation ─────────────────────────────────────────────────
st.sidebar.title("🗄️ DB Workload Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📈 Forecast", "🚨 Anomalies & Causes", "🔬 What-If Lab", "✅ Recommendations"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API Backend:** `{API_BASE}`")


def api_get(path: str, params: dict = None):
    """Helper to call the FastAPI backend GET endpoints."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE}. Is the backend service running?")
        return None
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None


def api_post(path: str, payload: dict = None):
    """Helper to call the FastAPI backend POST endpoints."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE}. Is the backend service running?")
        return None
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None


# ════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview — Live Workload Metrics")

    col1, col2 = st.columns([3, 1])
    with col2:
        hours = st.selectbox("Time Horizon", [1, 6, 24, 48, 168], index=2, format_func=lambda h: f"Last {h} hours")

    # Summary metrics
    data = api_get("/metrics/summary", {"hours": hours})
    if data:
        metrics = {m["metric"]: m for m in data.get("metrics", [])}

        c1, c2, c3, c4, c5 = st.columns(5)
        qps_val = metrics.get('db.qps', {}).get('avg_val')
        c1.metric(
            "Average QPS",
            f"{qps_val:.1f}" if qps_val is not None else "N/A",
            help="Queries per second computed across 30-second snapshot intervals",
        )

        chr_val = metrics.get('db.cache_hit_ratio', {}).get('avg_val')
        c2.metric(
            "Cache Hit Ratio",
            f"{chr_val:.3f}" if chr_val is not None else "Unavailable",
            help="Buffer cache hit ratio (blks_hit / total_blks). 1.0 indicates entire working set in RAM",
        )

        conn_val = metrics.get('db.connections', {}).get('max_val')
        c3.metric(
            "Peak Connections",
            f"{conn_val:.0f}" if conn_val is not None else "0",
            help="Maximum simultaneous backend client connections",
        )

        deadlocks_val = metrics.get('db.deadlocks', {}).get('avg_val')
        c4.metric(
            "Deadlocks / Interval",
            f"{deadlocks_val:.1f}" if deadlocks_val is not None else "0.0",
            help="Detected deadlocks",
        )

        waiters_val = metrics.get('db.lock_waiters', {}).get('max_val')
        c5.metric(
            "Peak Lock Waiters",
            f"{waiters_val:.0f}" if waiters_val is not None else "0",
            help="Peak concurrent queries waiting on table or row locks",
        )

    st.markdown("---")

    # QPS time-series chart
    st.subheader("Throughput: Queries Per Second (QPS)")
    st.caption("Interval rate (queries/sec) bucketed into 5-minute intervals. Missing intervals are preserved as gaps.")
    ts_data = api_get("/metrics/timeseries", {"metric": "db.qps", "hours": hours, "bucket_minutes": 5})
    if ts_data and ts_data.get("points"):
        import pandas as pd
        df = pd.DataFrame(ts_data["points"])
        df["bucket"] = pd.to_datetime(df["bucket"])
        df = df.set_index("bucket")
        st.line_chart(df[["avg_val"]], use_container_width=True)
    else:
        st.info("No QPS observations in this time window.")

    # Cache hit ratio chart
    st.subheader("Efficiency: Buffer Cache Hit Ratio")
    st.caption("Ratio of buffer pool memory hits vs disk reads over time.")
    cr_data = api_get("/metrics/timeseries", {"metric": "db.cache_hit_ratio", "hours": hours, "bucket_minutes": 5})
    if cr_data and cr_data.get("points"):
        import pandas as pd
        df = pd.DataFrame(cr_data["points"])
        df["bucket"] = pd.to_datetime(df["bucket"])
        df = df.set_index("bucket")
        st.line_chart(df[["avg_val"]], use_container_width=True)

    st.markdown("---")

    # Top queries
    st.subheader("🔝 Top Workload Queries (from pg_stat_statements)")
    q_data = api_get("/metrics/top-queries", {"hours": hours, "limit": 10})
    if q_data and q_data.get("queries"):
        import pandas as pd
        df = pd.DataFrame(q_data["queries"])
        df = df[["queryid", "total_calls", "total_exec_ms", "avg_mean_ms", "query_text"]]
        df.columns = ["Query ID", "Calls", "Total ms", "Avg ms", "Query Text"]
        df["Query Text"] = df["Query Text"].str[:90] + "..."
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No query executions recorded yet.")


# ════════════════════════════════════════════════════════════
# PAGE: Forecast  (rolling-average baseline — not an ML model)
# ════════════════════════════════════════════════════════════
elif page == "📈 Forecast":
    import pandas as pd

    st.title("📈 Workload Forecast")
    st.caption(
        "**Baseline forecast** — 4-period (20-min) rolling average of historical QPS. "
        "This is a naive baseline, **not** a trained ML model."
    )

    hours = st.selectbox(
        "Historical window",
        [6, 12, 24, 48],
        index=2,
        format_func=lambda h: f"Last {h} hours",
    )

    ts_data = api_get("/metrics/timeseries", {"metric": "db.qps", "hours": hours, "bucket_minutes": 5})

    if ts_data and ts_data.get("points"):
        df = pd.DataFrame(ts_data["points"])
        df["bucket"] = pd.to_datetime(df["bucket"])
        df = df.set_index("bucket").sort_index()

        if len(df) < 2:
            st.warning("Fewer than 2 data points — cannot compute a rolling forecast. Wait for more metric data.")
        else:
            # 4-period (20 min) rolling-average baseline shifted by 1 to act as a forecast
            df["Actual QPS"]           = df["avg_val"]
            df["Forecast (rolling avg)"] = df["avg_val"].rolling(window=4, min_periods=1).mean().shift(1)

            st.subheader("QPS: Actual vs Baseline Forecast")
            st.line_chart(df[["Actual QPS", "Forecast (rolling avg)"]], use_container_width=True)

            last_actual   = float(df["Actual QPS"].dropna().iloc[-1])
            last_forecast = float(df["Forecast (rolling avg)"].dropna().iloc[-1]) if df["Forecast (rolling avg)"].dropna().shape[0] > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Last Actual QPS",      f"{last_actual:.2f}")
            c2.metric("Baseline Forecast QPS", f"{last_forecast:.2f}")
            c3.metric("Historical Data Points", len(df))

            with st.expander("ℹ️ How this forecast works"):
                st.markdown(
                    "Each forecast point is the **mean of the previous 4 observed QPS buckets** (each bucket = 5 min). "
                    "The forecast is shifted by one period so it represents a look-ahead prediction. "
                    "No LightGBM, ARIMA, or other statistical model is used — this is a transparent naive baseline."
                )
    else:
        st.info(
            "No QPS metric data found in the last "
            f"{hours} hours. Ensure the collector and load generator are running."
        )


# ════════════════════════════════════════════════════════════
# PAGE: Anomalies & Causes  (with RCA baseline)
# ════════════════════════════════════════════════════════════
elif page == "🚨 Anomalies & Causes":
    st.title("🚨 Anomalies & Root Causes")

    hours = st.slider("Look-back window (hours)", 1, 168, 24)
    data = api_get("/anomalies/", {"hours": hours})

    if data and data.get("anomalies"):
        import pandas as pd

        incidents = data["anomalies"]
        df = pd.DataFrame(incidents)
        st.success(f"Found **{len(df)}** recorded incidents in the last {hours} hours.")
        st.dataframe(
            df[["id", "start_ts", "end_ts", "kind", "source", "description"]],
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🔍 Incident Root Cause Analysis")

        # Pre-populate selector with actual incident IDs from the table
        inc_ids = [str(r["id"]) for r in incidents]
        selected_id_str = st.selectbox(
            "Select incident to analyse:",
            inc_ids,
            format_func=lambda i: f"#{i} — {next((r['kind'] for r in incidents if str(r['id']) == i), '')} ({next((r['source'] for r in incidents if str(r['id']) == i), '')})",
        )
        selected_id = int(selected_id_str)

        if st.button("🔬 Load Root Cause Analysis", type="primary"):
            rca = api_get(f"/incidents/{selected_id}/causes")
            if rca:
                st.session_state["rca_result"] = rca

        rca = st.session_state.get("rca_result")
        if rca and rca.get("incident_id") == selected_id:
            st.markdown("#### Incident Summary")

            source_icon = "💉" if rca.get("source") == "injected" else "🤖"
            st.info(
                f"{source_icon} **{rca.get('kind', '').replace('_', ' ').title()}** — "
                f"{rca.get('source_note', '')}  \n"
                f"**Description:** {rca.get('description', 'N/A')}  \n"
                f"**Start:** {rca.get('start_ts', 'N/A')}  \n"
                f"**End:** {rca.get('end_ts') or 'ongoing'}"
            )

            st.markdown("#### Metric Signals (±30 min around incident)")

            snap = rca.get("metric_snapshots", {})
            if snap:
                rows = []
                for m, vals in snap.items():
                    before = vals.get("before")
                    after  = vals.get("after")
                    if before is None and after is None:
                        continue
                    delta = ""
                    if before is not None and after is not None and before != 0:
                        pct = (after - before) / abs(before) * 100
                        delta = f"{pct:+.1f}%"
                    rows.append({
                        "Metric": m,
                        "Before (avg)": f"{before:.3f}" if before is not None else "—",
                        "After (avg)":  f"{after:.3f}"  if after  is not None else "—",
                        "Change":       delta or "—",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("#### Findings")
            findings = rca.get("findings", [])
            if findings:
                for f in findings:
                    direction_icon = "⬆️" if f.get("direction") == "up" else "⬇️"
                    st.warning(
                        f"{direction_icon} **{f.get('label', f.get('metric'))}**: "
                        f"{f.get('explanation', '')}"
                    )
            else:
                st.info(
                    "⚠️ " + rca.get(
                        "summary",
                        "No significant metric changes detected in the ±30-minute window."
                    )
                )

            st.caption(f"RCA summary: {rca.get('summary', '')}")
    else:
        st.info(f"No incidents detected in the last {hours}h.")


# ════════════════════════════════════════════════════════════
# PAGE: What-If Lab
# ════════════════════════════════════════════════════════════
elif page == "🔬 What-If Lab":
    st.title("🔬 What-If Index Simulator")
    st.markdown(
        "Simulate index optimizations against the PostgreSQL query planner in real time using **HypoPG**. "
        "No physical indexes are created on disk; estimates are computed through hypothetical index cost modeling."
    )

    # ── Step A: Select a query ──────────────────────────────────────────
    st.subheader("Step 1: Select a Target Query")
    query_resp = api_get("/whatif/queries")

    if not query_resp or not query_resp.get("queries"):
        st.warning("No candidate SELECT queries available from `pg_stat_statements`. Ensure the load generator is active.")
        selected_sql = st.text_area(
            "Or enter a custom SELECT query:",
            value="SELECT * FROM orders WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 20;",
            height=100,
        )
    else:
        q_list = query_resp["queries"]
        options = [
            f"Query #{q['queryid']} | Calls: {q['calls']:,} | Avg: {q['mean_exec_time_ms']}ms | {q['query'][:80]}..."
            for q in q_list
        ]
        chosen_idx = st.selectbox(
            "Choose a frequently executed or expensive query:",
            range(len(options)),
            format_func=lambda i: options[i],
        )
        chosen_item = q_list[chosen_idx]

        col_q1, col_q2, col_q3 = st.columns(3)
        col_q1.metric("Execution Count",      f"{chosen_item['calls']:,}")
        col_q2.metric("Total Execution Time", f"{chosen_item['total_exec_time_ms']:.1f} ms")
        col_q3.metric("Avg Latency",          f"{chosen_item['mean_exec_time_ms']:.2f} ms")

        selected_sql = chosen_item["query"]
        with st.expander("View Full Query SQL", expanded=True):
            st.code(selected_sql, language="sql")

    st.markdown("---")

    # ── Step B: Generate Candidates ─────────────────────────────────────
    st.subheader("Step 2: Generate & Inspect Candidate Indexes")

    if st.button("🔍 Analyze Query & Generate Candidate Indexes", type="primary"):
        with st.spinner("Parsing query AST, cross-referencing schema & filtering existing indexes..."):
            cands = api_post("/whatif/candidates", {"query_text": selected_sql})
            if cands is not None:
                st.session_state["whatif_candidates"] = cands
                st.session_state["whatif_query_sql"]  = selected_sql
                # Clear previous evaluation when the query changes
                st.session_state.pop("whatif_eval_result", None)
                if isinstance(cands, list) and len(cands) == 0:
                    st.info(
                        "✅ Query parsed successfully, but no new index candidates were generated. "
                        "All predicate columns may already be indexed, or the query has no WHERE/JOIN predicates."
                    )

    cands       = st.session_state.get("whatif_candidates", [])
    current_sql = st.session_state.get("whatif_query_sql", selected_sql)

    if not cands:
        if "whatif_candidates" not in st.session_state:
            st.info("Click **'Analyze Query & Generate Candidate Indexes'** above to begin.")
    else:
        st.success(f"Generated **{len(cands)}** valid non-duplicate candidate index(es) for this query.")
        cand_labels = [
            f"#{i+1}: {c['table_name']} ({', '.join(c['columns'])}) — {c['reason']}"
            for i, c in enumerate(cands)
        ]
        selected_cand_idx = st.radio(
            "Select a candidate to simulate:",
            range(len(cand_labels)),
            format_func=lambda i: cand_labels[i],
        )
        selected_cand = cands[selected_cand_idx]

        st.code(selected_cand["create_sql"], language="sql")

        st.markdown("---")

        # ── Step C: HypoPG plan comparison ──────────────────────────────
        st.subheader("Step 3: Simulate Plan with HypoPG")
        st.caption(
            "HypoPG tells the PostgreSQL planner to consider this index virtually without building it. "
            "Costs are planner-estimated units — not wall-clock time."
        )

        if st.button("⚡ Run What-If Plan Comparison", type="primary"):
            with st.spinner("Executing isolated HypoPG simulation & plan analysis..."):
                eval_res = api_post(
                    "/whatif/evaluate",
                    {
                        "query_text": current_sql,
                        "table_name": selected_cand["table_name"],
                        "columns":    selected_cand["columns"],
                        "reason":     selected_cand["reason"],
                    },
                )
                if eval_res:
                    st.session_state["whatif_eval_result"] = eval_res

        # ── Step D: Display results & save ──────────────────────────────
        res = st.session_state.get("whatif_eval_result")
        if res:
            st.markdown("### Plan Comparison Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline Plan Cost",     f"{res['cost_before']:.2f}")
            m2.metric("Hypothetical Plan Cost",  f"{res['cost_after']:.2f}")

            imp_pct = res["cost_reduction_pct"]
            m3.metric(
                "Estimated Cost Delta",
                f"{imp_pct:+.1f}%",
                delta=f"{imp_pct:.1f}% reduction" if imp_pct > 0 else f"{imp_pct:.1f}% increase",
            )
            m4.metric("Index Selected by Planner?", "✅ Yes" if res["index_used"] else "❌ No")

            if res.get("warning"):
                st.warning(f"⚠️ {res['warning']}")

            tab1, tab2 = st.tabs(["Execution Plan Summaries", "Raw Plan Text (EXPLAIN)"])
            with tab1:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**Baseline Plan Path:**")
                    st.info(res["baseline_plan_summary"] or "Seq Scan")
                with col_p2:
                    st.markdown("**Candidate Plan Path:**")
                    st.success(res["candidate_plan_summary"] or "Index Scan")

            with tab2:
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Baseline `EXPLAIN` Plan:**")
                    st.code(res["baseline_plan_text"], language="text")
                with col_t2:
                    st.markdown("**Candidate `EXPLAIN` Plan:**")
                    st.code(res["candidate_plan_text"], language="text")

            st.markdown("---")

            # ── Save as recommendation ───────────────────────────────────
            st.subheader("Step 4: Persist as Recommendation")
            st.caption(
                "Saving stores the candidate in the Index Recommendations registry for human review. "
                "No index is ever physically created or auto-applied."
            )

            rec_rationale = st.text_input(
                "Recommendation Rationale:",
                value=f"HypoPG simulation shows {imp_pct:.1f}% estimated planner cost reduction for {selected_cand['reason']}.",
            )

            if st.button("💾 Save to Recommendations Registry", type="primary"):
                save_payload = {
                    "kind":            "create_index",
                    "target_table":    res["table_name"],
                    "target_columns":  res["columns"],
                    "ddl":             res["create_sql"],
                    "rollback_ddl":    res["rollback_sql"],
                    "est_cost_before": res["cost_before"],
                    "est_cost_after":  res["cost_after"],
                    "source_query":    current_sql,
                    "rationale":       rec_rationale,
                    "safety_score":    85.0 if res["index_used"] and imp_pct > 20 else 60.0,
                }
                save_res = api_post("/recommendations/", save_payload)
                if save_res:
                    st.success("✅ Recommendation saved! Navigate to the **✅ Recommendations** tab to review.")


# ════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ════════════════════════════════════════════════════════════
elif page == "✅ Recommendations":
    st.title("✅ Index Recommendations & Safety Registry")
    st.markdown(
        "Review candidate index recommendations generated by the What-If Simulator. "
        "Approval generates an **operational runbook with DDL and rollback plans**. **Changes are NEVER auto-applied**."
    )

    status_filter = st.selectbox("Filter by Status", ["pending", "approved", "rejected", "all"], index=0)
    data = api_get("/recommendations/", {"status": status_filter})

    if data and data.get("recommendations"):
        recs = data["recommendations"]
        st.write(f"Showing **{len(recs)}** recommendation(s):")

        for rec in recs:
            rec_id = rec["id"]
            status_badge = "🟡 PENDING"
            if rec["status"] == "approved":
                status_badge = "🟢 APPROVED"
            elif rec["status"] == "rejected":
                status_badge = "🔴 REJECTED"

            table_name  = rec.get("target_table", "unknown")
            cols        = rec.get("target_columns") or []
            cost_before = rec.get("est_cost_before") or 0.0
            cost_after  = rec.get("est_cost_after")  or 0.0
            imp = ((cost_before - cost_after) / cost_before * 100.0) if cost_before > 0 else 0.0

            with st.expander(
                f"#{rec_id} — {status_badge} | Table: `{table_name}` ({', '.join(cols)}) | Cost Delta: {imp:+.1f}%",
                expanded=(rec["status"] == "pending"),
            ):
                st.markdown(f"**Rationale:** {rec.get('rationale', 'N/A')}")
                if rec.get("source_query"):
                    st.markdown("**Originating Query:**")
                    st.code(rec["source_query"], language="sql")

                c1, c2, c3 = st.columns(3)
                c1.metric("Est. Cost Before", f"{cost_before:.2f}")
                c2.metric("Est. Cost After",  f"{cost_after:.2f}")
                c3.metric("Planner Gain",      f"{imp:.1f}%")

                st.markdown("**Proposed DDL (`CONCURRENTLY` enforced):**")
                st.code(rec.get("ddl", ""), language="sql")

                st.markdown("**Rollback DDL:**")
                st.code(rec.get("rollback_ddl", ""), language="sql")

                # Action buttons
                col_b1, col_b2, col_b3 = st.columns(3)

                if rec["status"] == "pending":
                    with col_b1:
                        if st.button("✅ Approve Recommendation", key=f"app_{rec_id}", type="primary"):
                            r = api_post(f"/recommendations/{rec_id}/approve")
                            if r:
                                st.success("Recommendation approved. Runbook is ready.")
                                st.rerun()

                    with col_b2:
                        if st.button("❌ Reject", key=f"rej_{rec_id}"):
                            r = api_post(f"/recommendations/{rec_id}/reject")
                            if r:
                                st.warning("Recommendation marked as rejected.")
                                st.rerun()

                # View Runbook button — available for all statuses
                with col_b3:
                    if st.button("📋 View DBA Runbook", key=f"rb_{rec_id}"):
                        rb_data = api_get(f"/recommendations/{rec_id}/runbook")
                        if rb_data and rb_data.get("runbook"):
                            st.session_state[f"runbook_{rec_id}"] = rb_data["runbook"]

                if f"runbook_{rec_id}" in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📋 Generated DBA Operational Runbook")
                    st.markdown(st.session_state[f"runbook_{rec_id}"])
    else:
        st.info(f"No recommendations found with status '{status_filter}'. Run the **What-If Lab** to generate candidates.")

    st.markdown("---")
    st.warning(
        "⚠️ **Safety rule:** Recommendations are NEVER auto-applied. "
        "Approval generates a runbook with DDL + rollback SQL for manual human execution."
    )

# DB Workload Intelligence

DB Workload Intelligence is a complete, end-to-end PostgreSQL performance monitoring, anomaly detection, and index optimization platform. It collects live workload telemetry, surfaces anomalies, provides baseline workload forecasting, and features a zero-impact "What-If" index simulator.

---

## 🏗️ Architecture & Component Design

The project is designed using a decoupled microservices architecture to ensure the monitoring system does not negatively impact the database being monitored (the "observer effect"). 

### 1. Target Database (`target-db`)
* **What:** A PostgreSQL 16 instance running the actual application workload.
* **Extensions:** `pg_stat_statements` (for query tracking) and `hypopg` (for virtual index simulation).
* **Why:** By keeping this database strictly for the workload, we ensure that our heavy analytical monitoring queries don't artificially inflate disk I/O, cache eviction, or CPU usage on the production-like database.

### 2. Metrics Database (`metrics-db`)
* **What:** A separate PostgreSQL instance storing historical time-series telemetry.
* **Schema:** Uses partition-ready tables (`metric_snapshot`, `query_stat_delta`) to store interval-based data, incidents, and index recommendations.
* **Why:** Decoupling storage ensures that if the telemetry data grows to hundreds of gigabytes, it won't consume the Target DB's disk space or memory footprint.

### 3. Telemetry Collector (`collector/`)
* **What:** A continuous Python background daemon that queries `pg_stat_*` views every 30 seconds.
* **How it works:** PostgreSQL internal stats are mostly cumulative (ever-increasing counters). The collector fetches the current state, subtracts the previous state (handling counter resets gracefully), and writes the *delta* to the Metrics DB.
* **Why:** Polling from an external agent is much safer and more reliable than trying to push data from inside PostgreSQL using triggers or background workers.

### 4. FastAPI Backend (`api/`)
* **What:** The RESTful interface serving data to the frontend.
* **Why:** Decoupling the frontend from the database allows for secure credential management, efficient connection pooling (`asyncpg`), and provides a programmatic API that other tools (like Grafana or CI/CD pipelines) could potentially consume.

### 5. Streamlit Dashboard (`ui/`)
* **What:** An interactive web application for DBAs and developers.
* **Why:** Streamlit allows for rapid iteration of data-heavy dashboards, seamlessly integrating with Python's data ecosystem (Pandas) to render time-series charts and interactive tables.

### 6. Load Generator & Fault Injector (`loadgen/`)
* **What:** A Python script simulating a realistic e-commerce workload (60% read, 25% write). It also contains a background thread that periodically injects faults (e.g., connection spikes, table bloat, stale stats).
* **Why:** A monitoring tool is useless without data. The loadgen ensures there is constant, realistic traffic and introduces controlled chaos so the Anomaly & Root Cause Analysis engines have real incidents to detect.

---

## 🧠 Core Design Decisions (For Technical Interviews)

This project was built with production-grade safety and reliability in mind. Here is the rationale behind the three most critical architectural choices:

### 1. Why use HypoPG instead of just creating indexes?
Creating a physical index on a large production table is a highly expensive operation. Even using `CREATE INDEX CONCURRENTLY`, it consumes significant disk I/O, CPU, and takes time to build. If you build it and the PostgreSQL query planner decides *not* to use it, you've wasted resources and now have to drop it. 
**HypoPG** creates a virtual index in memory in milliseconds (costing only a few kilobytes). It tricks the PostgreSQL optimizer into generating an execution plan *as if* the index existed, allowing us to mathematically prove if the planner will use the index and how much the cost drops, with **zero risk to production performance**.

### 2. Why is telemetry stored in a completely separate database?
The **Observer Effect**. Monitoring tools generate a massive amount of time-series data (query snapshots, hardware metrics, incident logs). If we stored this data inside the target application database, the monitoring system's own `INSERT` operations would cause table bloat, push application data out of the RAM buffer cache, and compete for disk I/O. Decoupling the telemetry into a `metrics-db` ensures that the monitoring infrastructure never degrades the performance of the system it is trying to protect.

### 3. Why generate a manual DBA Runbook instead of auto-applying the DDL?
While the What-If engine can prove that the planner *estimates* a lower read cost, AI and cost-optimizers do not have the full business context. 
Every new index adds **write-overhead** (it slows down `INSERT`, `UPDATE`, and `DELETE` operations). A 90% read-cost improvement on a query executed once a day is not worth slowing down a write operation executed 10,000 times a second. By generating an operational runbook (with safe `CONCURRENTLY` creation scripts, rollback scripts, and a safety checklist), the tool does the heavy analytical lifting, but intentionally keeps a **human-in-the-loop** to make the final risk-vs-reward business decision.

---

## ✨ Key Features & Deep Dives

### 🔬 The What-If Index Simulator (Zero-Impact Tuning)
Traditionally, testing a new database index requires running `CREATE INDEX CONCURRENTLY`, which takes time, consumes disk space, and adds write-overhead to the table. If the planner doesn't use it, you have to drop it.

**Our Approach:**
1. **AST Parsing:** Uses `sqlglot` to parse expensive queries from `pg_stat_statements` and automatically extract `WHERE`, `JOIN`, and `ORDER BY` columns to propose candidate indexes.
2. **HypoPG Simulation:** Instead of building a real index, the API uses the `hypopg` extension to inject a *virtual* index into PostgreSQL's memory. 
3. **Planner Costing:** We run a read-only `EXPLAIN` query. The PostgreSQL query planner considers the virtual index and returns the estimated execution cost.
4. **Safety & Runbooks:** If the simulated cost drops significantly, the system generates a DBA-friendly "Runbook" containing the exact DDL, rollback DDL, and safety checklists. **No physical DDL is ever executed automatically.**

### 🚨 Anomalies & Root Cause Analysis (RCA)
When the load generator injects a fault, the API performs an automated Root Cause Analysis.
* It scans a ±30-minute window around the incident.
* It compares baseline metrics (QPS, connection count, deadlocks, cache hit ratio, temp files) against the incident window.
* It surfaces structured findings (e.g., "Active connections spiked 55.2%, consistent with a connection pool exhaustion").

### 📈 Workload Forecasting
The dashboard includes a transparent, seasonal-naive baseline forecast. It uses a rolling-average on historical QPS data to project short-term future load. It is intentionally designed as an honest baseline rather than an opaque ML model, ensuring DBAs know exactly how the prediction is calculated.

---

## 🚀 Getting Started

### Prerequisites
* Docker and Docker Compose
* Python 3.12+ (for local development/testing)

### Running the Stack
1. Clone the repository:
   ```bash
   git clone https://github.com/avadhii/db-workload-intelligence.git
   cd db-workload-intelligence
   ```
2. Create your environment file:
   ```bash
   cp .env.example .env
   # Edit .env if you want to customize passwords/ports
   ```
3. Start the core infrastructure and UI:
   ```bash
   docker compose up -d target-db metrics-db api ui collector
   ```
4. Start the workload generator to populate data:
   ```bash
   docker compose --profile loadgen up -d loadgen
   ```
5. Access the dashboard:
   * **UI:** [http://localhost:8501](http://localhost:8501)
   * **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Running Tests
The project includes a comprehensive `pytest` suite covering AST parsing, delta calculations, and safety scoring:
```bash
python -m pytest tests/ -v
```

---

## 📁 Directory Structure

```text
db-workload-intelligence/
├── api/             # FastAPI backend services and routers
├── collector/       # Telemetry polling agent and delta computation
├── infra/           # Dockerfiles and PostgreSQL init scripts
├── loadgen/         # Workload simulation and fault injection
├── sql/             # Database schemas (metrics store and target Pagila app)
├── tests/           # Pytest suite
├── ui/              # Streamlit dashboard application
├── whatif/          # AST parsing, HypoPG eval, and Runbook generation
├── docker-compose.yml
└── requirements.txt
```

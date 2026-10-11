.PHONY: help up down logs build restart clean setup-local load-schema

# ─── Default ──────────────────────────────────────────────────────────
help:
	@echo "DB Workload Intelligence — Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make up            Start all core services (Docker)"
	@echo "  make up-loadgen    Start core services + load generator"
	@echo "  make down          Stop all services"
	@echo "  make logs          Tail logs from all services"
	@echo "  make build         Rebuild Docker images"
	@echo "  make restart       Restart all services"
	@echo "  make clean         Remove containers, volumes, images"
	@echo "  make setup-local   Install Python deps locally (venv)"
	@echo "  make load-schema   Load the sample schema into target-db"
	@echo "  make test          Run unit tests"
	@echo "  make lint          Run linters"

# ─── Docker Commands ──────────────────────────────────────────────────
up:
	docker compose up -d

up-loadgen:
	docker compose --profile loadgen up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

restart:
	docker compose restart

clean:
	docker compose down -v --rmi local

# ─── Local (no Docker) Setup ──────────────────────────────────────────
setup-local:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Activate with: source .venv/bin/activate"

# ─── Database ──────────────────────────────────────────────────────────
load-schema:
	@echo "Loading sample schema (Pagila)..."
	psql $${TARGET_DB_DSN} -f sql/pagila_schema.sql
	psql $${TARGET_DB_DSN} -f sql/pagila_data.sql
	psql $${TARGET_DB_DSN} -f sql/metrics_schema.sql

metrics-schema:
	@echo "Loading metrics schema into metrics-db..."
	psql $${METRICS_DB_DSN} -f sql/metrics_schema.sql

# ─── Tests ───────────────────────────────────────────────────────────
test:
	. .venv/bin/activate && pytest tests/ -v

lint:
	. .venv/bin/activate && ruff check . && black --check .

# ─── Collector (local run) ────────────────────────────────────────────
run-collector:
	. .venv/bin/activate && python -m collector.main

run-loadgen:
	. .venv/bin/activate && python -m loadgen.main

run-api:
	. .venv/bin/activate && uvicorn api.main:app --reload --port 8000

run-ui:
	. .venv/bin/activate && streamlit run ui/app.py --server.port 8501

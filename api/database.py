"""
database.py — Async database connection pool for FastAPI.
"""
from __future__ import annotations

import os

import asyncpg
from dotenv import load_dotenv

load_dotenv()

_pool: asyncpg.Pool | None = None


async def init_db():
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=os.environ["METRICS_DB_DSN"],
        min_size=2,
        max_size=10,
    )


async def get_db() -> asyncpg.Pool:
    if _pool is None:
        await init_db()
    return _pool

"""
sandbox_manager.py — Ephemeral Postgres sandbox lifecycle management (Phase 5b).

Spawns an isolated Postgres container (matching target DB settings),
replicates schema & sampled data, runs benchmarks safely without polluting production,
and ensures automatic destruction via timeout/TTL.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional
from contextlib import contextmanager

import psycopg2

logger = logging.getLogger(__name__)

# Docker SDK is optional to allow running tests or sandbox mocking without docker installed
try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False


@dataclass
class SandboxConfig:
    image: str = "postgres:16"
    port: int = 5435
    user: str = "sandboxuser"
    password: str = "sandboxpass"
    dbname: str = "sandboxdb"
    container_name_prefix: str = "pg-sandbox-"
    ttl_seconds: int = 600
    # Resource limits
    cpu_limit: float = 1.0       # 1 CPU
    mem_limit: str = "1g"        # 1 GB RAM

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@localhost:{self.port}/{self.dbname}"


class SandboxManager:
    """Manages ephemeral Postgres sandbox containers via Docker SDK."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._client = docker.from_env() if HAS_DOCKER else None

    def create_sandbox(self) -> str:
        """Starts an ephemeral PostgreSQL container with strict resource limits."""
        if not HAS_DOCKER or self._client is None:
            raise RuntimeError("Docker SDK is required to spin up containerized sandboxes.")

        container_name = f"{self.config.container_name_prefix}{int(time.time())}"
        logger.info("Spawning sandbox container %s on port %d...", container_name, self.config.port)

        env = {
            "POSTGRES_USER": self.config.user,
            "POSTGRES_PASSWORD": self.config.password,
            "POSTGRES_DB": self.config.dbname,
        }

        container = self._client.containers.run(
            self.config.image,
            name=container_name,
            detach=True,
            environment=env,
            ports={"5432/tcp": self.config.port},
            nano_cpus=int(self.config.cpu_limit * 1e9),
            mem_limit=self.config.mem_limit,
            tmpfs={"/var/lib/postgresql/data": "rw,noexec,nosuid,size=512m"},
            remove=False,
        )

        # Wait for database ready
        self._wait_for_ready(self.config.dsn, timeout=30)
        logger.info("Sandbox container %s is ready.", container_name)
        return container.id

    def destroy_sandbox(self, container_id_or_name: str) -> None:
        """Stops and forcefully removes the sandbox container."""
        if not HAS_DOCKER or self._client is None:
            return

        try:
            container = self._client.containers.get(container_id_or_name)
            logger.info("Stopping and removing sandbox container %s...", container.name)
            container.stop(timeout=2)
            container.remove(force=True)
        except Exception as exc:
            logger.warning("Failed to clean up sandbox container %s: %s", container_id_or_name, exc)

    def _wait_for_ready(self, dsn: str, timeout: int = 30) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                conn = psycopg2.connect(dsn, connect_timeout=2)
                conn.close()
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError(f"Sandbox DB failed to become ready within {timeout}s.")

    @contextmanager
    def run_ephemeral_sandbox(self) -> Generator[str, None, None]:
        """Context manager for automatic creation and teardown of a sandbox."""
        container_id = self.create_sandbox()
        try:
            yield self.config.dsn
        finally:
            self.destroy_sandbox(container_id)

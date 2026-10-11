"""
config.py — Collector configuration loaded from environment variables.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class CollectorConfig:
    target_dsn: str
    metrics_dsn: str
    interval_seconds: int
    max_queries_tracked: int
    # How many seconds before we consider a stats-reset has occurred
    # (counter drops by more than this fraction → assume reset)
    reset_detection_drop_fraction: float

    @classmethod
    def from_env(cls) -> "CollectorConfig":
        return cls(
            target_dsn=os.environ["TARGET_DB_DSN"],
            metrics_dsn=os.environ["METRICS_DB_DSN"],
            interval_seconds=int(os.getenv("COLLECT_INTERVAL_SECONDS", "30")),
            max_queries_tracked=int(os.getenv("MAX_QUERIES_TRACKED", "500")),
            reset_detection_drop_fraction=float(os.getenv("RESET_DROP_FRACTION", "0.5")),
        )

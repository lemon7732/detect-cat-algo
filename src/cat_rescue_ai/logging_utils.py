"""Logging setup helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def setup_logging(config: dict[str, Any]) -> logging.Logger:
    logging_config = config.get("logging", {})
    level_name = logging_config.get("level", "INFO")
    log_format = logging_config.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.basicConfig(level=getattr(logging, level_name.upper(), logging.INFO), format=log_format)
    logger = logging.getLogger(config.get("project_name", "cat-rescue-ai"))

    log_dir = config.get("paths", {}).get("log_dir")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    return logger

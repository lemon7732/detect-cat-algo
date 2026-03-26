"""Helpers for optional imports."""

from __future__ import annotations

import importlib
from typing import Any

from cat_rescue_ai.exceptions import DependencyNotAvailableError


def require(module_name: str, pip_name: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        package_name = pip_name or module_name
        raise DependencyNotAvailableError(
            f"Missing dependency '{module_name}'. Install it with: pip install {package_name}"
        ) from exc

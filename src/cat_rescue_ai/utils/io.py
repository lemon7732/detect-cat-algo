"""Output and serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def save_json(path: str | Path, payload: Any) -> Path:
    destination = ensure_parent(path)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return destination


def save_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    destination = ensure_parent(path)
    rows = list(rows)
    if not rows:
        with destination.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return destination
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def save_text(path: str | Path, content: str) -> Path:
    destination = ensure_parent(path)
    destination.write_text(content, encoding="utf-8")
    return destination


def export_model_summary(model: Any, output_path: str | Path) -> Path:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return save_text(output_path, "\n".join(lines))


def export_training_curves(history: dict[str, list[float]], output_path: str | Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return None

    destination = ensure_parent(output_path)
    plt.figure(figsize=(10, 4))
    for key, values in history.items():
        if values:
            plt.plot(values, label=key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    return destination

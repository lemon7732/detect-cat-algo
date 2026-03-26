"""Helpers for generating gallery metadata CSV files from directory datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cat_rescue_ai.utils.image import iter_image_files
from cat_rescue_ai.utils.io import save_csv


def generate_gallery_metadata(
    gallery_root: str | Path,
    output_csv: str | Path,
    image_extensions: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(gallery_root)
    extensions = image_extensions or [".jpg", ".jpeg", ".png"]
    rows: list[dict[str, Any]] = []
    for cat_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cat_id = cat_dir.name
        for image_path in iter_image_files(cat_dir, extensions):
            rows.append(
                {
                    "cat_id": cat_id,
                    "name": cat_id,
                    "sex": "",
                    "age": "",
                    "description": "",
                    "image_path": str(image_path.relative_to(root)),
                }
            )
    save_csv(output_csv, rows)
    return {"gallery_root": str(root), "output_csv": str(output_csv), "row_count": len(rows)}

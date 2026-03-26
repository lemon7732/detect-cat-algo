"""Gallery build, load, and match helpers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cat_rescue_ai.config import load_config
from cat_rescue_ai.exceptions import GalleryNotBuiltError
from cat_rescue_ai.features.similarity import apply_rejection_policy, rank_gallery
from cat_rescue_ai.utils.coords import mean_vector
from cat_rescue_ai.utils.io import save_json


def load_gallery_metadata(metadata_csv: str | Path) -> list[dict[str, Any]]:
    with Path(metadata_csv).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_gallery_index(pipeline, config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata_rows = load_gallery_metadata(config["metadata_csv"])
    gallery_root = Path(config.get("gallery_root", "."))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for row in metadata_rows:
        cat_id = row["cat_id"]
        image_path = row.get("image_path")
        if not image_path:
            failures.append({"cat_id": cat_id, "reason": "missing image_path"})
            continue
        resolved_image_path = Path(image_path)
        if not resolved_image_path.is_absolute():
            resolved_image_path = gallery_root / resolved_image_path
        try:
            result = pipeline.extract_features(str(resolved_image_path))
            grouped[cat_id].append(
                {
                    "cat_id": cat_id,
                    "name": row.get("name", cat_id),
                    "sex": row.get("sex"),
                    "age": row.get("age"),
                    "description": row.get("description"),
                    "image_path": str(resolved_image_path),
                    "feature_vector": result["feature_vector"],
                    "bbox": result["bbox"],
                }
            )
        except Exception as exc:
            failures.append({"cat_id": cat_id, "image_path": str(resolved_image_path), "reason": str(exc)})

    entries = []
    for cat_id, rows in grouped.items():
        prototype_vector = mean_vector([row["feature_vector"] for row in rows])
        first = rows[0]
        entries.append(
            {
                "cat_id": cat_id,
                "name": first.get("name", cat_id),
                "sex": first.get("sex"),
                "age": first.get("age"),
                "description": first.get("description"),
                "prototype_vector": prototype_vector,
                "image_vectors": [row["feature_vector"] for row in rows],
                "image_paths": [row["image_path"] for row in rows],
            }
        )

    output_cfg = config["output"]
    save_json(output_cfg["json_path"], {"entries": entries})
    save_json(output_cfg["failures_json"], {"failures": failures})
    try:
        import numpy as np  # type: ignore

        np.savez(
            output_cfg["npz_path"],
            cat_ids=np.array([entry["cat_id"] for entry in entries], dtype=object),
            names=np.array([entry["name"] for entry in entries], dtype=object),
            prototype_vectors=np.array([entry["prototype_vector"] for entry in entries], dtype=float),
        )
    except ModuleNotFoundError:
        pass
    return {"entries": entries, "failures": failures}


def load_gallery_index(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    json_path = Path(config["output"]["json_path"])
    if not json_path.exists():
        raise GalleryNotBuiltError(f"Gallery index not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not entries:
        raise GalleryNotBuiltError("Gallery index exists but contains no entries.")
    return {"config": config, "entries": entries}


def match_against_gallery(
    query_vector: list[float],
    gallery_payload: dict[str, Any],
    top_k: int | None = None,
    cosine_threshold: float | None = None,
    euclidean_threshold: float | None = None,
) -> dict[str, Any]:
    config = gallery_payload.get("config", {})
    match_cfg = config.get("matching", {})
    ranked = rank_gallery(
        query_vector,
        gallery_payload["entries"],
        top_k=top_k or int(match_cfg.get("top_k", 5)),
        mode=str(match_cfg.get("mode", "image")).strip().lower(),
    )
    best = apply_rejection_policy(
        ranked,
        cosine_threshold=float(cosine_threshold if cosine_threshold is not None else match_cfg.get("cosine_threshold", 0.85)),
        euclidean_threshold=float(
            euclidean_threshold if euclidean_threshold is not None else match_cfg.get("euclidean_threshold", 0.25)
        ),
    )
    return {"best": best, "top_k": ranked}

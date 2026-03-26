"""Feature vector similarity functions."""

from __future__ import annotations

import math
from typing import Any, Sequence

from cat_rescue_ai.exceptions import UnknownCatError


def euclidean_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(vector_a, vector_b)))


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")
    dot = sum(float(a) * float(b) for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(float(a) ** 2 for a in vector_a))
    norm_b = math.sqrt(sum(float(b) ** 2 for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_vector_pair(query_vector: Sequence[float], reference_vector: Sequence[float]) -> dict[str, float]:
    return {
        "cosine_score": cosine_similarity(query_vector, reference_vector),
        "euclidean_distance": euclidean_distance(query_vector, reference_vector),
    }


def _score_candidate(query_vector: Sequence[float], entry: dict[str, Any], mode: str) -> dict[str, Any]:
    prototype_scores = _score_vector_pair(query_vector, entry["prototype_vector"])
    prototype_candidate = {
        **prototype_scores,
        "match_source": "prototype",
        "matched_image_path": None,
    }

    best_image_candidate: dict[str, Any] | None = None
    image_vectors = entry.get("image_vectors", [])
    image_paths = entry.get("image_paths", [])
    for index, vector in enumerate(image_vectors):
        candidate = {
            **_score_vector_pair(query_vector, vector),
            "match_source": "image",
            "matched_image_path": image_paths[index] if index < len(image_paths) else None,
        }
        if best_image_candidate is None or (
            candidate["cosine_score"],
            -candidate["euclidean_distance"],
        ) > (
            best_image_candidate["cosine_score"],
            -best_image_candidate["euclidean_distance"],
        ):
            best_image_candidate = candidate

    if mode == "prototype" or best_image_candidate is None:
        selected = prototype_candidate
    elif mode == "image":
        selected = best_image_candidate
    else:
        selected = max(
            [prototype_candidate, best_image_candidate],
            key=lambda item: (item["cosine_score"], -item["euclidean_distance"]),
        )

    return {
        **entry,
        **selected,
        "prototype_cosine_score": prototype_candidate["cosine_score"],
        "prototype_euclidean_distance": prototype_candidate["euclidean_distance"],
        "best_image_cosine_score": None if best_image_candidate is None else best_image_candidate["cosine_score"],
        "best_image_euclidean_distance": None
        if best_image_candidate is None
        else best_image_candidate["euclidean_distance"],
        "best_image_path": None if best_image_candidate is None else best_image_candidate["matched_image_path"],
    }


def rank_gallery(
    query_vector: Sequence[float],
    gallery_entries: list[dict[str, Any]],
    top_k: int = 5,
    mode: str = "image",
) -> list[dict[str, Any]]:
    if mode not in {"prototype", "image", "hybrid"}:
        raise ValueError(f"Unsupported gallery ranking mode: {mode}")
    ranked: list[dict[str, Any]] = []
    for entry in gallery_entries:
        ranked.append(_score_candidate(query_vector, entry, mode=mode))
    ranked.sort(key=lambda item: (-item["cosine_score"], item["euclidean_distance"]))
    return ranked[:top_k]


def apply_rejection_policy(
    ranked_results: list[dict[str, Any]],
    cosine_threshold: float,
    euclidean_threshold: float,
) -> dict[str, Any]:
    if not ranked_results:
        raise UnknownCatError("Gallery is empty.")
    best = ranked_results[0]
    if best["cosine_score"] < cosine_threshold or best["euclidean_distance"] > euclidean_threshold:
        raise UnknownCatError("No gallery entry passed the similarity thresholds.")
    return best

"""Coordinate and landmark helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


LANDMARK_NAMES = [
    "left_eye",
    "right_eye",
    "mouth",
    "left_ear1",
    "left_ear2",
    "left_ear3",
    "right_ear1",
    "right_ear2",
    "right_ear3",
]

LANDMARK_COLUMNS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]


def flatten_points(points: Sequence[Sequence[float]]) -> list[float]:
    flattened: list[float] = []
    for x, y in points:
        flattened.extend([float(x), float(y)])
    return flattened


def chunk_points(vector: Sequence[float]) -> list[tuple[float, float]]:
    if len(vector) % 2 != 0:
        raise ValueError("Landmark vector must contain an even number of values.")
    return [(float(vector[i]), float(vector[i + 1])) for i in range(0, len(vector), 2)]


def normalize_landmarks(vector: Sequence[float], width: float, height: float) -> list[float]:
    normalized: list[float] = []
    for index, value in enumerate(vector):
        denominator = width if index % 2 == 0 else height
        normalized.append(float(value) / float(denominator))
    return normalized


def denormalize_landmarks(vector: Sequence[float], width: float, height: float) -> list[float]:
    denormalized: list[float] = []
    for index, value in enumerate(vector):
        scale = width if index % 2 == 0 else height
        denormalized.append(float(value) * float(scale))
    return denormalized


def normalize_by_bbox(vector: Sequence[float], bbox: Sequence[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    normalized: list[float] = []
    for index, value in enumerate(vector):
        if index % 2 == 0:
            normalized.append((float(value) - x) / max(w, 1e-6))
        else:
            normalized.append((float(value) - y) / max(h, 1e-6))
    return normalized


def mean_vector(vectors: Iterable[Sequence[float]]) -> list[float]:
    vectors = [list(map(float, vector)) for vector in vectors]
    if not vectors:
        raise ValueError("Cannot compute mean of empty vectors.")
    length = len(vectors[0])
    totals = [0.0] * length
    for vector in vectors:
        if len(vector) != length:
            raise ValueError("All vectors must have identical length.")
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]


def mean_absolute_error(predicted: Sequence[float], target: Sequence[float]) -> float:
    if len(predicted) != len(target):
        raise ValueError("Landmark vectors must share the same length.")
    return sum(abs(float(a) - float(b)) for a, b in zip(predicted, target)) / len(predicted)


def root_mean_squared_error(predicted: Sequence[float], target: Sequence[float]) -> float:
    if len(predicted) != len(target):
        raise ValueError("Landmark vectors must share the same length.")
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(predicted, target)) / len(predicted)
    return math.sqrt(mse)

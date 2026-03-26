"""Pydantic schemas defined lazily to avoid hard dependency on import."""

from __future__ import annotations


def create_schema_namespace():
    from pydantic import BaseModel, Field

    class HealthResponse(BaseModel):
        status: str = "ok"

    class SpeciesResponse(BaseModel):
        is_cat: bool
        cat_probability: float

    class DetectionResponse(BaseModel):
        faces: list[dict]
        primary_bbox: list[int] | None

    class LandmarkResponse(BaseModel):
        bbox: list[int]
        landmarks: list[float]
        normalized_landmarks: list[float]

    class IdentifyResponse(BaseModel):
        is_cat: bool
        cat_probability: float
        face_detected: bool
        landmarks: list[float]
        matched_cat_id: str | None
        matched_name: str | None
        cosine_score: float | None
        euclidean_distance: float | None
        match_source: str | None = None
        matched_image_path: str | None = None
        top_k: list[dict] = Field(default_factory=list)
        is_unknown: bool

    class GalleryRebuildResponse(BaseModel):
        entries: int
        failures: int

    return {
        "HealthResponse": HealthResponse,
        "SpeciesResponse": SpeciesResponse,
        "DetectionResponse": DetectionResponse,
        "LandmarkResponse": LandmarkResponse,
        "IdentifyResponse": IdentifyResponse,
        "GalleryRebuildResponse": GalleryRebuildResponse,
    }

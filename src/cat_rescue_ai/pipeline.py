"""End-to-end cat recognition pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cat_rescue_ai.config import load_config
from cat_rescue_ai.detection.cat_face import CatFaceDetector
from cat_rescue_ai.exceptions import (
    CatFaceNotFoundError,
    LandmarkPredictionError,
    NonCatImageError,
    UnknownCatError,
)
from cat_rescue_ai.gallery.index import load_gallery_index, match_against_gallery
from cat_rescue_ai.inference.species import _prepare_binary_image
from cat_rescue_ai.models.binary import build_binary_model
from cat_rescue_ai.models.landmarks import build_vgg_landmark_model
from cat_rescue_ai.utils.deps import require
from cat_rescue_ai.utils.coords import denormalize_landmarks, normalize_by_bbox
from cat_rescue_ai.utils.image import read_image


class RecognitionPipeline:
    def __init__(
        self,
        binary_config_path: str | Path,
        landmarks_config_path: str | Path,
        gallery_config_path: str | Path | None = None,
        cascade_path: str | Path | None = None,
        allow_full_image_fallback: bool = False,
    ) -> None:
        self.binary_config = load_config(binary_config_path)
        self.landmark_config = load_config(landmarks_config_path)
        self.binary_model = build_binary_model(self.binary_config)
        self.binary_model.load_weights(self.binary_config["artifacts"]["best_weights"])
        self.landmark_model = build_vgg_landmark_model(self.landmark_config)
        self.landmark_model.load_weights(self.landmark_config["artifacts"]["best_weights"])
        self.detector = CatFaceDetector(cascade_path=cascade_path)
        self.gallery_payload = load_gallery_index(gallery_config_path) if gallery_config_path else None
        self.allow_full_image_fallback = allow_full_image_fallback

    def classify_species(self, image_source: Any) -> dict[str, Any]:
        image = read_image(image_source)
        tensor = _prepare_binary_image(self.binary_config, image)
        probability = float(self.binary_model.predict(tensor, verbose=0)[0][0])
        return {"is_cat": probability >= 0.5, "cat_probability": probability}

    def detect_cat_face(self, image_source: Any) -> dict[str, Any]:
        return self.detector.detect(image_source)

    def predict_landmarks(self, image_source: Any) -> dict[str, Any]:
        image = read_image(image_source)
        try:
            face_crop, bbox = self.detector.crop_primary_face(image)
        except CatFaceNotFoundError:
            if not self.allow_full_image_fallback:
                raise
            face_crop = image
            bbox = [0, 0, image.size[0], image.size[1]]
        prepared = face_crop.resize(tuple(self.landmark_config["model"].get("input_size", [224, 224])))
        tf = require("tensorflow")
        tensor = tf.keras.utils.img_to_array(prepared) / 255.0
        tensor = tf.expand_dims(tensor, axis=0)
        predicted = self.landmark_model.predict(tensor, verbose=0)[0].tolist()
        crop_width, crop_height = face_crop.size
        if len(predicted) != 18:
            raise LandmarkPredictionError("Landmark model did not return 18 coordinates.")
        local_landmarks = denormalize_landmarks(predicted, crop_width, crop_height)
        x, y, w, h = bbox
        global_landmarks: list[float] = []
        for index, value in enumerate(local_landmarks):
            global_landmarks.append(float(value) + (x if index % 2 == 0 else y))
        return {"bbox": bbox, "landmarks": global_landmarks, "normalized_landmarks": predicted}

    def extract_features(self, image_source: Any) -> dict[str, Any]:
        species = self.classify_species(image_source)
        if not species["is_cat"]:
            raise NonCatImageError("The uploaded image was classified as not-cat.")
        landmark_result = self.predict_landmarks(image_source)
        feature_vector = normalize_by_bbox(landmark_result["landmarks"], landmark_result["bbox"])
        return {
            "is_cat": species["is_cat"],
            "cat_probability": species["cat_probability"],
            "bbox": landmark_result["bbox"],
            "landmarks": landmark_result["landmarks"],
            "feature_vector": feature_vector,
        }

    def identify(self, image_source: Any, top_k: int = 5) -> dict[str, Any]:
        if self.gallery_payload is None:
            raise ValueError("RecognitionPipeline was initialized without a gallery configuration.")
        extracted = self.extract_features(image_source)
        try:
            matched = match_against_gallery(extracted["feature_vector"], self.gallery_payload, top_k=top_k)
            best = matched["best"]
            return {
                **extracted,
                "face_detected": True,
                "matched_cat_id": best["cat_id"],
                "matched_name": best["name"],
                "cosine_score": best["cosine_score"],
                "euclidean_distance": best["euclidean_distance"],
                "match_source": best.get("match_source"),
                "matched_image_path": best.get("matched_image_path"),
                "top_k": matched["top_k"],
                "is_unknown": False,
            }
        except UnknownCatError:
            return {
                **extracted,
                "face_detected": True,
                "matched_cat_id": None,
                "matched_name": None,
                "cosine_score": None,
                "euclidean_distance": None,
                "top_k": [],
                "is_unknown": True,
            }

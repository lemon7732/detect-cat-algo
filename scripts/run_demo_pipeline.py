from __future__ import annotations

import json
from pathlib import Path

from bootstrap_demo_data import main as bootstrap_demo_main
from cat_rescue_ai.gallery.index import build_gallery_index
from cat_rescue_ai.pipeline import RecognitionPipeline
from cat_rescue_ai.training.binary_trainer import train_binary_model
from cat_rescue_ai.training.landmark_trainer import train_landmark_model


def main() -> None:
    bootstrap_demo_main()
    train_binary_model("configs/demo_binary.yaml")
    train_landmark_model("configs/demo_landmarks.yaml", cascade_path="assets/cascades/haarcascade_frontalcatface.xml")

    pipeline = RecognitionPipeline(
        binary_config_path="configs/demo_binary.yaml",
        landmarks_config_path="configs/demo_landmarks.yaml",
        cascade_path="assets/cascades/haarcascade_frontalcatface.xml",
        allow_full_image_fallback=True,
    )
    build_gallery_index(pipeline, "configs/demo_gallery.yaml")

    identify_pipeline = RecognitionPipeline(
        binary_config_path="configs/demo_binary.yaml",
        landmarks_config_path="configs/demo_landmarks.yaml",
        gallery_config_path="configs/demo_gallery.yaml",
        cascade_path="assets/cascades/haarcascade_frontalcatface.xml",
        allow_full_image_fallback=True,
    )
    results = {}
    for query_path in sorted(Path("data/demo/query").glob("*.png")):
        results[query_path.name] = identify_pipeline.identify(str(query_path), top_k=3)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cat_rescue_ai.config import load_config
from cat_rescue_ai.pipeline import RecognitionPipeline
from cat_rescue_ai.utils.image import iter_image_files
from cat_rescue_ai.utils.io import save_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch cat identification.")
    parser.add_argument("--config", default="configs/api.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", default="artifacts/predict_batch/results.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RecognitionPipeline(
        binary_config_path=config["pipeline"]["binary_config"],
        landmarks_config_path=config["pipeline"]["landmarks_config"],
        gallery_config_path=config["pipeline"]["gallery_config"],
        cascade_path=config["pipeline"].get("cascade_path"),
        allow_full_image_fallback=bool(config["pipeline"].get("allow_full_image_fallback", False)),
    )

    rows = []
    for image_path in iter_image_files(args.input_dir, [".jpg", ".jpeg", ".png"]):
        try:
            result = pipeline.identify(str(image_path), top_k=int(config["matching"].get("top_k", 5)))
            rows.append(
                {
                    "image_path": str(image_path),
                    "matched_cat_id": result["matched_cat_id"],
                    "matched_name": result["matched_name"],
                    "cat_probability": result["cat_probability"],
                    "cosine_score": result["cosine_score"],
                    "euclidean_distance": result["euclidean_distance"],
                    "is_unknown": result["is_unknown"],
                }
            )
        except Exception as exc:
            rows.append({"image_path": str(image_path), "error": str(exc)})
    save_csv(args.output_csv, rows)
    print(json.dumps({"count": len(rows), "output_csv": str(Path(args.output_csv))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

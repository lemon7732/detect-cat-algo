from __future__ import annotations

import argparse
import json

from cat_rescue_ai.config import load_config
from cat_rescue_ai.gallery.index import build_gallery_index
from cat_rescue_ai.pipeline import RecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the campus cat gallery index.")
    parser.add_argument("--config", default="configs/gallery.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = RecognitionPipeline(
        binary_config_path=config["pipeline"]["binary_config"],
        landmarks_config_path=config["pipeline"]["landmarks_config"],
        cascade_path=config["pipeline"].get("cascade_path"),
        allow_full_image_fallback=bool(config["pipeline"].get("allow_full_image_fallback", False)),
    )
    result = build_gallery_index(pipeline, args.config)
    print(json.dumps({"entries": len(result["entries"]), "failures": result["failures"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

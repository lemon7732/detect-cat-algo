from __future__ import annotations

import argparse
import json

from cat_rescue_ai.config import load_config
from cat_rescue_ai.pipeline import RecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify a cat from one image.")
    parser.add_argument("--config", default="configs/api.yaml")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = RecognitionPipeline(
        binary_config_path=config["pipeline"]["binary_config"],
        landmarks_config_path=config["pipeline"]["landmarks_config"],
        gallery_config_path=config["pipeline"]["gallery_config"],
        cascade_path=config["pipeline"].get("cascade_path"),
        allow_full_image_fallback=bool(config["pipeline"].get("allow_full_image_fallback", False)),
    )
    print(json.dumps(pipeline.identify(args.image, top_k=int(config["matching"].get("top_k", 5))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

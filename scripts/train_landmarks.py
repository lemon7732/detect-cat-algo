from __future__ import annotations

import argparse
import json

from cat_rescue_ai.training.landmark_trainer import train_landmark_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cat landmark regressor.")
    parser.add_argument("--config", default="configs/landmarks.yaml")
    parser.add_argument("--cascade-path", default=None)
    args = parser.parse_args()
    result = train_landmark_model(args.config, cascade_path=args.cascade_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from cat_rescue_ai.training.landmark_trainer import evaluate_landmark_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the cat landmark regressor.")
    parser.add_argument("--config", default="configs/landmarks.yaml")
    parser.add_argument("--cascade-path", default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_landmark_model(args.config, cascade_path=args.cascade_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

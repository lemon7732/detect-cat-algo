from __future__ import annotations

import argparse
import json

from cat_rescue_ai.training.binary_trainer import evaluate_binary_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the cat-vs-not-cat model.")
    parser.add_argument("--config", default="configs/binary.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate_binary_model(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

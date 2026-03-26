from __future__ import annotations

import argparse
import json

from cat_rescue_ai.training.binary_trainer import train_binary_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cat-vs-not-cat model.")
    parser.add_argument("--config", default="configs/binary.yaml")
    args = parser.parse_args()
    result = train_binary_model(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

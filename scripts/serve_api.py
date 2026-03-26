from __future__ import annotations

import argparse

from cat_rescue_ai.api.app import create_app
from cat_rescue_ai.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--config", default="configs/api.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    app = create_app(args.config)
    import uvicorn  # type: ignore

    uvicorn.run(
        app,
        host=config.get("host", "0.0.0.0"),
        port=int(config.get("port", 8000)),
        reload=bool(config.get("reload", False)),
    )


if __name__ == "__main__":
    main()

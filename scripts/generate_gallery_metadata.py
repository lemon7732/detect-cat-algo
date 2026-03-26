from __future__ import annotations

import argparse
import json

from cat_rescue_ai.gallery.metadata import generate_gallery_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gallery metadata.csv from per-cat image folders.")
    parser.add_argument("--gallery-root", required=True)
    parser.add_argument("--output", default="data/gallery/metadata.csv")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png"])
    args = parser.parse_args()
    result = generate_gallery_metadata(args.gallery_root, args.output, image_extensions=args.extensions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from cat_rescue_ai.utils.download_checks import (
    check_cat_dataset,
    check_cat_individual_images_dataset,
    check_catflw_dataset,
    check_tfds_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether required public datasets have finished downloading.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cats_vs_dogs", "oxford_iiit_pet", "cat_dataset", "catflw", "cat_individual_images"],
        choices=["cats_vs_dogs", "oxford_iiit_pet", "cat_dataset", "catflw", "cat_individual_images"],
    )
    parser.add_argument("--tfds-dir", default="data/tfds")
    parser.add_argument("--cat-dataset-dir", default="data/cat_dataset")
    parser.add_argument("--catflw-dir", default="data/catflw")
    parser.add_argument("--wildlife-dir", default="data/wildlife/CatIndividualImages")
    args = parser.parse_args()

    results = []
    for dataset_name in args.datasets:
        if dataset_name in {"cats_vs_dogs", "oxford_iiit_pet"}:
            results.append(check_tfds_dataset(dataset_name, args.tfds_dir))
        elif dataset_name == "cat_dataset":
            results.append(check_cat_dataset(args.cat_dataset_dir))
        elif dataset_name == "catflw":
            results.append(check_catflw_dataset(args.catflw_dir))
        elif dataset_name == "cat_individual_images":
            results.append(check_cat_individual_images_dataset(args.wildlife_dir))
    print(json.dumps({"datasets": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

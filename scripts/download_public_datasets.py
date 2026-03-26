from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _configure_ssl_env() -> None:
    try:
        import certifi

        cert_path = certifi.where()
    except Exception:
        return
    os.environ.setdefault("SSL_CERT_FILE", cert_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
    os.environ.setdefault("CURL_CA_BUNDLE", cert_path)
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", cert_path)


def _download_tfds_dataset(name: str, data_dir: str | None) -> dict[str, str]:
    _configure_ssl_env()
    import tensorflow_datasets as tfds

    builder = tfds.builder(name, data_dir=data_dir)
    builder.download_and_prepare()
    return {
        "name": name,
        "data_dir": str(builder.data_dir),
        "status": "downloaded",
    }


def _download_cat_individual_images(root: str | Path) -> dict[str, str]:
    from wildlife_datasets import datasets

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    datasets.CatIndividualImages.get_data(str(destination))
    return {
        "name": "CatIndividualImages",
        "data_dir": str(destination),
        "status": "downloaded",
    }


def _download_kaggle_dataset(ref: str, destination: str | Path, unzip: bool = True) -> dict[str, str]:
    _configure_ssl_env()
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", ref, "-p", str(target_dir)]
    if unzip:
        command.append("--unzip")
    subprocess.run(command, check=True)
    return {
        "name": ref,
        "data_dir": str(target_dir),
        "status": "downloaded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets used by the project.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cats_vs_dogs", "oxford_iiit_pet"],
        choices=[
            "cats_vs_dogs",
            "oxford_iiit_pet",
            "celeb_a",
            "caltech_birds2011",
            "cat_dataset",
            "cat_individual_images",
        ],
        help="Datasets to download.",
    )
    parser.add_argument("--tfds-dir", default="data/tfds")
    parser.add_argument("--cat-dataset-dir", default="data/cat_dataset")
    parser.add_argument("--wildlife-dir", default="data/wildlife")
    args = parser.parse_args()

    results = []
    for dataset_name in args.datasets:
        if dataset_name in {"cats_vs_dogs", "oxford_iiit_pet", "celeb_a", "caltech_birds2011"}:
            results.append(_download_tfds_dataset(dataset_name, args.tfds_dir))
        elif dataset_name == "cat_dataset":
            results.append(_download_kaggle_dataset("crawford/cat-dataset", args.cat_dataset_dir, unzip=True))
        elif dataset_name == "cat_individual_images":
            results.append(_download_cat_individual_images(Path(args.wildlife_dir) / "CatIndividualImages"))
    print(json.dumps({"downloads": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

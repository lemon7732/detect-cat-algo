"""Helpers for checking dataset download status."""

from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_SUFFIXES = {".json", ".csv", ".txt", ".xml"}


def _directory_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _latest_version_dir(dataset_root: Path) -> Path | None:
    if not dataset_root.exists():
        return None
    candidates = [path for path in dataset_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def check_tfds_dataset(name: str, tfds_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(tfds_root) / name
    version_dir = _latest_version_dir(dataset_root)
    if version_dir is None:
        return {"name": name, "status": "missing", "path": str(dataset_root)}

    dataset_info = version_dir / "dataset_info.json"
    shards = sorted(version_dir.glob("*.tfrecord*")) + sorted(version_dir.glob("*.array_record*"))
    status = "completed" if dataset_info.exists() and shards else "downloading"

    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "path": str(version_dir),
        "dataset_info_exists": dataset_info.exists(),
        "shard_count": len(shards),
        "size_bytes": _directory_size_bytes(version_dir),
    }

    try:
        import tensorflow_datasets as tfds

        builder = tfds.builder(name, data_dir=str(tfds_root))
        result["splits"] = {
            split_name: int(split_info.num_examples) for split_name, split_info in builder.info.splits.items()
        }
    except Exception as exc:
        result["splits_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _count_matching_files(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    count = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            count += 1
    return count


def check_catflw_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root)
    zip_path = dataset_root / "catflw.zip"
    image_count = _count_matching_files(dataset_root, IMAGE_SUFFIXES)
    annotation_count = _count_matching_files(dataset_root, ANNOTATION_SUFFIXES)
    non_zip_file_count = sum(1 for path in dataset_root.rglob("*") if path.is_file() and path != zip_path)

    if image_count > 0 and annotation_count > 0:
        status = "completed"
    elif zip_path.exists() or non_zip_file_count > 0:
        status = "downloading"
    else:
        status = "missing"

    return {
        "name": "catflw",
        "status": status,
        "path": str(dataset_root),
        "zip_exists": zip_path.exists(),
        "image_count": image_count,
        "annotation_count": annotation_count,
        "non_zip_file_count": non_zip_file_count,
        "size_bytes": _directory_size_bytes(dataset_root),
    }


def check_cat_individual_images_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root)
    zip_files = sorted(dataset_root.glob("*.zip")) if dataset_root.exists() else []
    image_count = _count_matching_files(dataset_root, IMAGE_SUFFIXES)
    annotation_count = _count_matching_files(dataset_root, ANNOTATION_SUFFIXES)
    non_zip_file_count = sum(1 for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() != ".zip")

    if image_count > 0:
        status = "completed"
    elif zip_files or non_zip_file_count > 0:
        status = "downloading"
    else:
        status = "missing"

    return {
        "name": "cat_individual_images",
        "status": status,
        "path": str(dataset_root),
        "zip_count": len(zip_files),
        "image_count": image_count,
        "annotation_count": annotation_count,
        "non_zip_file_count": non_zip_file_count,
        "size_bytes": _directory_size_bytes(dataset_root),
    }


def check_cat_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root)
    zip_files = sorted(dataset_root.glob("*.zip")) if dataset_root.exists() else []
    image_count = _count_matching_files(dataset_root, IMAGE_SUFFIXES)
    annotation_count = sum(1 for path in dataset_root.rglob("*.cat")) if dataset_root.exists() else 0
    non_zip_file_count = sum(1 for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() != ".zip")

    if image_count > 0 and annotation_count > 0:
        status = "completed"
    elif zip_files or non_zip_file_count > 0:
        status = "downloading"
    else:
        status = "missing"

    return {
        "name": "cat_dataset",
        "status": status,
        "path": str(dataset_root),
        "zip_count": len(zip_files),
        "image_count": image_count,
        "annotation_count": annotation_count,
        "non_zip_file_count": non_zip_file_count,
        "size_bytes": _directory_size_bytes(dataset_root),
    }

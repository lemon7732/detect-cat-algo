from __future__ import annotations

from pathlib import Path

from cat_rescue_ai.utils.download_checks import (
    check_cat_dataset,
    check_cat_individual_images_dataset,
    check_catflw_dataset,
    check_tfds_dataset,
)


def test_check_tfds_dataset_completed(tmp_path: Path):
    version_dir = tmp_path / "cats_vs_dogs" / "4.0.1"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    (version_dir / "cats_vs_dogs-train.tfrecord-00000-of-00001").write_text("x", encoding="utf-8")
    result = check_tfds_dataset("cats_vs_dogs", tmp_path)
    assert result["status"] == "completed"
    assert result["dataset_info_exists"] is True
    assert result["shard_count"] == 1


def test_check_catflw_dataset_states(tmp_path: Path):
    missing = check_catflw_dataset(tmp_path / "missing")
    assert missing["status"] == "missing"

    downloading_root = tmp_path / "catflw"
    downloading_root.mkdir(parents=True, exist_ok=True)
    (downloading_root / "catflw.zip").write_text("zip", encoding="utf-8")
    downloading = check_catflw_dataset(downloading_root)
    assert downloading["status"] == "downloading"

    completed_root = tmp_path / "catflw_done"
    (completed_root / "images").mkdir(parents=True, exist_ok=True)
    (completed_root / "labels").mkdir(parents=True, exist_ok=True)
    (completed_root / "images" / "a.jpg").write_text("img", encoding="utf-8")
    (completed_root / "labels" / "a.json").write_text("{}", encoding="utf-8")
    completed = check_catflw_dataset(completed_root)
    assert completed["status"] == "completed"


def test_check_cat_individual_images_dataset_states(tmp_path: Path):
    missing = check_cat_individual_images_dataset(tmp_path / "missing")
    assert missing["status"] == "missing"

    downloading_root = tmp_path / "wildlife"
    downloading_root.mkdir(parents=True, exist_ok=True)
    (downloading_root / "cat-individuals.zip").write_text("zip", encoding="utf-8")
    downloading = check_cat_individual_images_dataset(downloading_root)
    assert downloading["status"] == "downloading"

    completed_root = tmp_path / "wildlife_done"
    (completed_root / "cat_1").mkdir(parents=True, exist_ok=True)
    (completed_root / "cat_1" / "1.jpg").write_text("img", encoding="utf-8")
    completed = check_cat_individual_images_dataset(completed_root)
    assert completed["status"] == "completed"


def test_check_cat_dataset_states(tmp_path: Path):
    missing = check_cat_dataset(tmp_path / "missing")
    assert missing["status"] == "missing"

    downloading_root = tmp_path / "cat_dataset"
    downloading_root.mkdir(parents=True, exist_ok=True)
    (downloading_root / "cat-dataset.zip").write_text("zip", encoding="utf-8")
    downloading = check_cat_dataset(downloading_root)
    assert downloading["status"] == "downloading"

    completed_root = tmp_path / "cat_dataset_done" / "CAT_00"
    completed_root.mkdir(parents=True, exist_ok=True)
    (completed_root / "1.jpg").write_text("img", encoding="utf-8")
    (completed_root / "1.jpg.cat").write_text("9 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9", encoding="utf-8")
    completed = check_cat_dataset(tmp_path / "cat_dataset_done")
    assert completed["status"] == "completed"

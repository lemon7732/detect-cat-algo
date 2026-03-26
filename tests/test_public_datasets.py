from __future__ import annotations

import json
from pathlib import Path

from cat_rescue_ai.datasets import binary_dataset as binary_dataset_module
from cat_rescue_ai.datasets.binary_dataset import preprocess_binary_dataset
from cat_rescue_ai.datasets.landmark_dataset import (
    parse_cat_dataset_annotations,
    parse_catflw_annotations,
    preprocess_landmark_dataset,
)
from cat_rescue_ai.gallery.metadata import generate_gallery_metadata


class _FakeLabelFeature:
    def __init__(self, names: dict[int, str]):
        self.names = names

    def int2str(self, value: int) -> str:
        return self.names[int(value)]


class _FakeBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.info = type(
            "Info",
            (),
            {"features": {"label": _FakeLabelFeature({0: "cat", 1: "dog"})}},
        )()

    def download_and_prepare(self):
        return None

    def as_dataset(self, split, shuffle_files=False):
        return list(self.dataset[split])


class _FakeTFDS:
    def __init__(self, dataset):
        self._dataset = dataset

    def builder(self, dataset_name, data_dir=None):
        return _FakeBuilder(self._dataset)

    def as_numpy(self, dataset):
        return dataset


def test_preprocess_binary_dataset_supports_tfds_source(tmp_path: Path, monkeypatch):
    import numpy as np

    fake_dataset = {
        "train": [
            {"image": np.full((32, 32, 3), 20, dtype=np.uint8), "label": 0, "file_name": b"cat_0.jpg"},
            {"image": np.full((32, 32, 3), 220, dtype=np.uint8), "label": 1, "file_name": b"dog_0.jpg"},
        ]
    }
    monkeypatch.setattr(binary_dataset_module, "require", lambda module, pip_name=None: _FakeTFDS(fake_dataset))

    config = {
        "data": {
            "source": "tfds_cats_vs_dogs",
            "processed_dir": str(tmp_path / "processed"),
            "tfds_splits": ["train"],
            "max_edge": 32,
            "positive_label_names": ["cat"],
            "negative_label_names": ["dog"],
        }
    }
    samples = preprocess_binary_dataset(config)
    assert len(samples) == 2
    assert {sample.label for sample in samples} == {0, 1}
    assert all(Path(sample.path).exists() for sample in samples)


def test_preprocess_binary_dataset_supports_fixed_label_tfds_source(tmp_path: Path, monkeypatch):
    import numpy as np

    fake_dataset = {
        "train": [
            {"image": np.full((32, 32, 3), 20, dtype=np.uint8), "file_name": b"human_0.jpg"},
            {"image": np.full((32, 32, 3), 220, dtype=np.uint8), "file_name": b"human_1.jpg"},
        ]
    }
    monkeypatch.setattr(binary_dataset_module, "require", lambda module, pip_name=None: _FakeTFDS(fake_dataset))

    config = {
        "data": {
            "source": "tfds",
            "tfds_name": "celeb_a",
            "processed_dir": str(tmp_path / "processed"),
            "tfds_splits": ["train"],
            "max_edge": 32,
            "fixed_binary_label": 0,
            "filename_prefix": "celeb_a",
        }
    }
    samples = preprocess_binary_dataset(config)
    assert len(samples) == 2
    assert {sample.label for sample in samples} == {0}
    assert all("/not_cat/" in sample.path for sample in samples)


def test_preprocess_binary_dataset_supports_mixed_sources(tmp_path: Path, monkeypatch):
    import numpy as np
    from PIL import Image

    fake_dataset = {
        "train": [
            {"image": np.full((32, 32, 3), 20, dtype=np.uint8), "label": 0, "file_name": b"cat_0.jpg"},
            {"image": np.full((32, 32, 3), 220, dtype=np.uint8), "label": 1, "file_name": b"dog_0.jpg"},
        ]
    }
    monkeypatch.setattr(binary_dataset_module, "require", lambda module, pip_name=None: _FakeTFDS(fake_dataset))

    raw_dir = tmp_path / "raw"
    (raw_dir / "cat").mkdir(parents=True, exist_ok=True)
    (raw_dir / "not_cat").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(raw_dir / "not_cat" / "person.jpg")

    config = {
        "data": {
            "source": "mixed",
            "processed_dir": str(tmp_path / "processed"),
            "max_edge": 32,
            "sources": [
                {
                    "source": "tfds_cats_vs_dogs",
                    "positive_label_names": ["cat"],
                    "negative_label_names": ["dog"],
                    "tfds_splits": ["train"],
                    "filename_prefix": "cats_vs_dogs",
                },
                {
                    "source": "directory",
                    "raw_dir": str(raw_dir),
                    "filename_prefix": "local_dir",
                },
            ],
        }
    }
    samples = preprocess_binary_dataset(config)
    assert len(samples) == 3
    assert sum(sample.label for sample in samples) == 1


def test_parse_catflw_annotations_with_group_mapping(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": "cat_001.jpg",
        "labels": [[10, 20], [12, 22], [30, 40], [50, 60], [52, 62], [70, 80], [90, 100], [92, 102], [94, 104]],
        "bbox": [5, 10, 100, 110],
    }
    (labels_dir / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "data": {
            "source": "catflw",
            "labels_dir": str(labels_dir),
            "point_groups": [[0, 1], [2], [3, 4], [5], [6], [7], [8], [0], [1]],
        }
    }
    rows = parse_catflw_annotations(config)
    assert len(rows) == 1
    assert rows[0]["image_id"] == "cat_001.jpg"
    assert rows[0]["landmarks"][:2] == [11.0, 21.0]


def test_preprocess_landmark_dataset_supports_catflw_source(tmp_path: Path):
    from PIL import Image

    image_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 120), color=(255, 255, 255)).save(image_dir / "cat_001.jpg")
    payload = {
        "image": "cat_001.jpg",
        "labels": [[10, 20], [12, 22], [30, 40], [50, 60], [52, 62], [70, 80], [90, 100], [92, 102], [94, 104]],
        "bbox": {"xmin": 5, "ymin": 10, "xmax": 105, "ymax": 110},
    }
    (labels_dir / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "data": {
            "source": "catflw",
            "image_dir": str(image_dir),
            "labels_dir": str(labels_dir),
            "processed_dir": str(tmp_path / "processed"),
            "point_groups": [[0], [1], [2], [3], [4], [5], [6], [7], [8]],
        }
    }
    samples = preprocess_landmark_dataset(config, detector=None)
    assert len(samples) == 1
    assert Path(samples[0].image_path).exists()
    assert samples[0].bbox == (5, 10, 100, 100)


def test_parse_cat_dataset_annotations_supports_dot_cat_files(tmp_path: Path):
    image_dir = tmp_path / "cat_dataset"
    nested_dir = image_dir / "CAT_00"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "0001.jpg").write_bytes(b"jpg")
    (nested_dir / "0001.jpg.cat").write_text(
        "9 11 21 31 41 51 61 71 81 91 101 111 121 131 141 151 161 171 181",
        encoding="utf-8",
    )
    config = {
        "data": {
            "source": "cat_dataset",
            "image_dir": str(image_dir),
            "annotation_dir": str(image_dir),
            "annotation_suffix": ".cat",
            "image_extensions": [".jpg"],
        }
    }
    rows = parse_cat_dataset_annotations(config)
    assert len(rows) == 1
    assert rows[0]["image_id"] == "CAT_00/0001.jpg"
    assert rows[0]["landmarks"][:6] == [11.0, 21.0, 31.0, 41.0, 51.0, 61.0]


def test_preprocess_landmark_dataset_supports_cat_dataset_source(tmp_path: Path):
    from PIL import Image

    image_dir = tmp_path / "cat_dataset"
    nested_dir = image_dir / "CAT_01"
    nested_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (240, 200), color=(255, 255, 255)).save(nested_dir / "0001.jpg")
    (nested_dir / "0001.jpg.cat").write_text(
        "9 60 80 160 80 110 140 40 55 55 20 85 45 150 45 180 15 190 60",
        encoding="utf-8",
    )
    config = {
        "data": {
            "source": "cat_dataset",
            "image_dir": str(image_dir),
            "annotation_dir": str(image_dir),
            "annotation_suffix": ".cat",
            "processed_dir": str(tmp_path / "processed"),
            "image_extensions": [".jpg"],
            "use_landmark_bbox": True,
            "bbox_padding_ratio": 0.1,
        }
    }
    samples = preprocess_landmark_dataset(config, detector=None)
    assert len(samples) == 1
    assert Path(samples[0].image_path).exists()
    assert "CAT_01/0001.jpg" in samples[0].image_path
    assert samples[0].bbox is not None


def test_generate_gallery_metadata_from_cat_directories(tmp_path: Path):
    from PIL import Image

    gallery_root = tmp_path / "gallery"
    for cat_id in ("cat_a", "cat_b"):
        cat_dir = gallery_root / cat_id
        cat_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(cat_dir / "1.jpg")
    output_csv = tmp_path / "metadata.csv"
    result = generate_gallery_metadata(gallery_root, output_csv)
    assert result["row_count"] == 2
    content = output_csv.read_text(encoding="utf-8")
    assert "cat_id" in content
    assert "cat_a/1.jpg" in content

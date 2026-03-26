"""Landmark dataset preprocessing and tf.data builders."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from cat_rescue_ai.utils.coords import LANDMARK_COLUMNS, normalize_landmarks
from cat_rescue_ai.utils.deps import require
from cat_rescue_ai.utils.image import read_image, save_image


FLIP_INDEX_MAP = [1, 0, 2, 6, 7, 8, 3, 4, 5]


@dataclass(frozen=True)
class LandmarkSample:
    image_path: str
    landmarks: tuple[float, ...]
    split: str = "train"
    source_image: str | None = None
    bbox: tuple[int, int, int, int] | None = None


def _landmark_source(config: dict[str, Any]) -> str:
    return str(config.get("data", {}).get("source", "csv")).strip().lower()


def _row_to_landmark_vector(row: dict[str, str]) -> list[float]:
    if all(column in row for column in LANDMARK_COLUMNS):
        return [float(row[column]) for column in LANDMARK_COLUMNS]

    thesis_alias_columns = [
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
        "mouse_x",
        "mouse_y",
        "left_ear1_x",
        "left_ear1_y",
        "left_ear2_x",
        "left_ear2_y",
        "left_ear3_x",
        "left_ear3_y",
        "right_ear1_x",
        "right_ear1_y",
        "right_ear2_x",
        "right_ear2_y",
        "right_ear3_x",
        "right_ear3_y",
    ]
    if all(column in row for column in thesis_alias_columns):
        return [float(row[column]) for column in thesis_alias_columns]

    ignored = {"image_id", "id", "filename", "image_path"}
    numeric_values: list[float] = []
    for key, value in row.items():
        if key in ignored or value in ("", None):
            continue
        numeric_values.append(float(value))
    if len(numeric_values) < 18:
        raise ValueError("Landmark CSV row must contain at least 18 numeric coordinate values.")
    return numeric_values[:18]


def _normalize_bbox_payload(bbox: Any, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = image_size
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        if {"x", "y", "width", "height"}.issubset(bbox):
            left = int(float(bbox["x"]))
            top = int(float(bbox["y"]))
            box_width = int(float(bbox["width"]))
            box_height = int(float(bbox["height"]))
            return (left, top, box_width, box_height)
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(bbox):
            left = int(float(bbox["xmin"]))
            top = int(float(bbox["ymin"]))
            right = int(float(bbox["xmax"]))
            bottom = int(float(bbox["ymax"]))
            return (left, top, max(1, right - left), max(1, bottom - top))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        left, top, third, fourth = [float(item) for item in bbox]
        if third > left and fourth > top:
            return (int(left), int(top), max(1, int(third - left)), max(1, int(fourth - top)))
        return (int(left), int(top), max(1, int(third)), max(1, int(fourth)))
    return None


def _average_point_group(points: list[tuple[float, float]], indices: list[int]) -> tuple[float, float]:
    selected = [points[index] for index in indices]
    x = sum(item[0] for item in selected) / len(selected)
    y = sum(item[1] for item in selected) / len(selected)
    return (x, y)


def _bbox_from_landmark_vector(
    landmarks: Sequence[float],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    xs = [float(landmarks[index]) for index in range(0, len(landmarks), 2)]
    ys = [float(landmarks[index]) for index in range(1, len(landmarks), 2)]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    box_width = max(max_x - min_x, 1.0)
    box_height = max(max_y - min_y, 1.0)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    left = max(0, int(min_x - pad_x))
    top = max(0, int(min_y - pad_y))
    right = min(width, int(max_x + pad_x))
    bottom = min(height, int(max_y + pad_y))
    return (left, top, max(1, right - left), max(1, bottom - top))


def _catflw_group_mapping(data_cfg: dict[str, Any]) -> list[list[int]]:
    groups = data_cfg.get("point_groups", [])
    if not groups:
        raise ValueError("CatFLW source requires data.point_groups with 9 groups of source landmark indices.")
    if len(groups) != len(LANDMARK_COLUMNS) // 2:
        raise ValueError("CatFLW point_groups must contain exactly 9 groups.")
    parsed_groups: list[list[int]] = []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("Each CatFLW point group must be a non-empty list of indices.")
        parsed_groups.append([int(index) for index in group])
    return parsed_groups


def _flatten_catflw_landmarks(landmarks_payload: Any, groups: list[list[int]]) -> list[float]:
    if not isinstance(landmarks_payload, list):
        raise ValueError("CatFLW landmark payload must be a list.")
    points: list[tuple[float, float]] = []
    for item in landmarks_payload:
        if isinstance(item, dict):
            x = float(item.get("x"))
            y = float(item.get("y"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = float(item[0])
            y = float(item[1])
        else:
            raise ValueError("Unsupported CatFLW landmark point format.")
        points.append((x, y))
    mapped = [_average_point_group(points, group) for group in groups]
    flattened: list[float] = []
    for x, y in mapped:
        flattened.extend([x, y])
    return flattened


def _catflw_record_to_row(record: dict[str, Any], groups: list[list[int]]) -> dict[str, Any]:
    image_id = (
        record.get("image_id")
        or record.get("image_path")
        or record.get("filename")
        or record.get("file_name")
        or record.get("img")
        or record.get("image")
    )
    if not image_id:
        raise ValueError("CatFLW annotation record is missing image identifier.")
    landmarks_payload = record.get("labels") or record.get("landmarks") or record.get("points") or record.get("keypoints")
    if landmarks_payload is None:
        raise ValueError("CatFLW annotation record is missing landmark payload.")
    bbox_payload = record.get("bounding_boxes") or record.get("bounding_box") or record.get("bbox") or record.get("face_bbox")
    return {
        "image_id": str(image_id),
        "landmarks": _flatten_catflw_landmarks(landmarks_payload, groups),
        "bbox": bbox_payload,
    }


def parse_catflw_annotations(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    groups = _catflw_group_mapping(data_cfg)
    annotation_json = data_cfg.get("annotation_json")
    labels_dir = data_cfg.get("labels_dir")
    records: list[dict[str, Any]] = []

    if annotation_json:
        payload = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "annotations" in payload and isinstance(payload["annotations"], list):
                payload = payload["annotations"]
            else:
                payload = list(payload.values())
        if not isinstance(payload, list):
            raise ValueError("CatFLW annotation_json must contain a list of records.")
        for item in payload:
            if isinstance(item, dict):
                records.append(_catflw_record_to_row(item, groups))
        return records

    if labels_dir:
        for json_path in sorted(Path(labels_dir).glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append(_catflw_record_to_row(payload, groups))
        return records

    raise ValueError("CatFLW source requires either data.annotation_json or data.labels_dir.")


def parse_landmark_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("image_id") or row.get("filename") or row.get("image_path") or row.get("id")
            if not image_id:
                raise ValueError("Landmark CSV is missing image identifier column.")
            rows.append({"image_id": image_id, "landmarks": _row_to_landmark_vector(row)})
    return rows


def _parse_cat_annotation_file(annotation_path: Path) -> list[float]:
    tokens = annotation_path.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError(f"Empty CAT annotation file: {annotation_path}")
    point_count = int(float(tokens[0]))
    values = [float(value) for value in tokens[1:]]
    if len(values) < point_count * 2:
        raise ValueError(f"CAT annotation has insufficient coordinates: {annotation_path}")
    if point_count < len(LANDMARK_COLUMNS) // 2:
        raise ValueError(f"CAT annotation must contain at least 9 landmarks: {annotation_path}")
    return values[: len(LANDMARK_COLUMNS)]


def parse_cat_dataset_annotations(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    image_dir = Path(data_cfg["image_dir"])
    annotation_dir = Path(data_cfg.get("annotation_dir", image_dir))
    annotation_suffix = str(data_cfg.get("annotation_suffix", ".cat"))
    image_extensions = {suffix.lower() for suffix in data_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])}

    records: list[dict[str, Any]] = []
    for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in image_extensions):
        relative_path = image_path.relative_to(image_dir)
        annotation_path = annotation_dir / relative_path.parent / f"{image_path.name}{annotation_suffix}"
        if not annotation_path.exists():
            alt_annotation_path = image_path.with_name(f"{image_path.name}{annotation_suffix}")
            if alt_annotation_path.exists():
                annotation_path = alt_annotation_path
            else:
                continue
        records.append(
            {
                "image_id": str(relative_path),
                "landmarks": _parse_cat_annotation_file(annotation_path),
            }
        )

    if not records:
        raise ValueError(f"No CAT dataset annotations found under {annotation_dir}.")
    return records


def _landmark_bbox(image_size: tuple[int, int], detector: Any | None, image_path: Path) -> tuple[int, int, int, int]:
    if detector is not None:
        try:
            result = detector.detect_path(image_path)
            bbox = result.get("primary_bbox")
            if bbox:
                return tuple(int(v) for v in bbox)
        except Exception:
            pass
    width, height = image_size
    return (0, 0, width, height)


def preprocess_landmark_dataset(config: dict[str, Any], detector: Any | None = None) -> list[LandmarkSample]:
    data_cfg = config["data"]
    image_dir = Path(data_cfg["image_dir"])
    source = _landmark_source(config)
    if source == "catflw":
        csv_rows = parse_catflw_annotations(config)
    elif source in {"cat_dataset", "cat_dot_cat"}:
        csv_rows = parse_cat_dataset_annotations(config)
    else:
        csv_rows = parse_landmark_csv(data_cfg["csv_path"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    bbox_padding_ratio = float(data_cfg.get("bbox_padding_ratio", 0.15))

    samples: list[LandmarkSample] = []
    for row in csv_rows:
        source_path = image_dir / str(row["image_id"])
        if not source_path.exists():
            candidates = list(image_dir.glob(f"{Path(str(row['image_id'])).stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"Landmark image not found for row: {row['image_id']}")
            source_path = candidates[0]

        image = read_image(source_path)
        bbox = _normalize_bbox_payload(row.get("bbox"), image.size)
        if bbox is None and data_cfg.get("use_landmark_bbox", True):
            bbox = _bbox_from_landmark_vector(row["landmarks"], image.size, padding_ratio=bbox_padding_ratio)
        if bbox is None:
            bbox = _landmark_bbox(image.size, detector, source_path)
        left, top, width, height = bbox
        cropped = image.crop((left, top, left + width, top + height))
        try:
            relative_path = source_path.relative_to(image_dir)
            target_path = processed_dir / relative_path
        except ValueError:
            target_path = processed_dir / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(cropped, target_path)

        shifted: list[float] = []
        for index, value in enumerate(row["landmarks"]):
            shifted.append(float(value) - (left if index % 2 == 0 else top))
        normalized = normalize_landmarks(shifted, width, height)
        samples.append(
            LandmarkSample(
                image_path=str(target_path),
                landmarks=tuple(normalized),
                source_image=str(source_path),
                bbox=(left, top, width, height),
            )
        )
    return samples


def split_landmark_samples(samples: list[LandmarkSample], val_ratio: float, seed: int) -> tuple[list[LandmarkSample], list[LandmarkSample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    val_samples = [LandmarkSample(**{**sample.__dict__, "split": "val"}) for sample in shuffled[:val_size]]
    train_samples = [LandmarkSample(**{**sample.__dict__, "split": "train"}) for sample in shuffled[val_size:]]
    return train_samples, val_samples


def summarize_landmark_samples(train_samples: Iterable[LandmarkSample], val_samples: Iterable[LandmarkSample]) -> dict[str, Any]:
    train_samples = list(train_samples)
    val_samples = list(val_samples)
    return {"train_count": len(train_samples), "val_count": len(val_samples)}


def build_landmark_tf_datasets(
    train_samples: list[LandmarkSample],
    val_samples: list[LandmarkSample],
    config: dict[str, Any],
):
    tf = require("tensorflow")
    batch_size = int(config["training"].get("batch_size", 16))
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    aug_cfg = config.get("augmentation", {})
    train_cfg = config["training"]

    def _dataset_from_samples(samples: list[LandmarkSample], training: bool):
        paths = [sample.image_path for sample in samples]
        coords = [list(sample.landmarks) for sample in samples]
        dataset = tf.data.Dataset.from_tensor_slices((paths, coords))
        if training and samples:
            dataset = dataset.shuffle(buffer_size=len(samples), seed=int(train_cfg.get("random_seed", 42)))

        def _load(path, landmarks):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, input_size)
            return image, tf.cast(landmarks, tf.float32)

        dataset = dataset.map(_load, num_parallel_calls=int(train_cfg.get("num_parallel_calls", 4)))

        def _flip_landmarks(landmarks):
            xs = tf.gather(landmarks[0::2], FLIP_INDEX_MAP)
            ys = tf.gather(landmarks[1::2], FLIP_INDEX_MAP)
            xs = 1.0 - xs
            stacked = tf.reshape(tf.stack([xs, ys], axis=1), (-1,))
            return stacked

        if training:
            def _augment(image, landmarks):
                if aug_cfg.get("horizontal_flip", True):
                    should_flip = tf.random.uniform(()) > 0.5
                    image = tf.cond(should_flip, lambda: tf.image.flip_left_right(image), lambda: image)
                    landmarks = tf.cond(should_flip, lambda: _flip_landmarks(landmarks), lambda: landmarks)
                if aug_cfg.get("brightness_delta", 0.0):
                    image = tf.image.random_brightness(image, max_delta=float(aug_cfg["brightness_delta"]))
                image = tf.clip_by_value(image, 0.0, 1.0)
                return image, landmarks

            dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.batch(batch_size).prefetch(int(train_cfg.get("prefetch_buffer", 2)))

    return _dataset_from_samples(train_samples, True), _dataset_from_samples(val_samples, False)

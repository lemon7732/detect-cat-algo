"""Binary cat-vs-not-cat dataset utilities."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cat_rescue_ai.utils.image import iter_image_files, read_image, resize_keep_ratio, save_image
from cat_rescue_ai.utils.deps import require


@dataclass(frozen=True)
class BinarySample:
    path: str
    label: int
    split: str = "train"


def _binary_source(config: dict[str, Any]) -> str:
    return str(config.get("data", {}).get("source", "directory")).strip().lower()


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _dataset_prefix(data_cfg: dict[str, Any], fallback_source: str) -> str:
    explicit = str(data_cfg.get("filename_prefix", "")).strip()
    if explicit:
        return _sanitize_filename(explicit)
    dataset_name = str(data_cfg.get("tfds_name", "")).strip()
    if dataset_name:
        return _sanitize_filename(dataset_name)
    return _sanitize_filename(fallback_source)


def _crop_with_normalized_bbox(image, bbox_value):
    if bbox_value is None:
        return image
    if isinstance(bbox_value, dict):
        keys = {"ymin", "xmin", "ymax", "xmax"}
        if not keys.issubset(bbox_value.keys()):
            return image
        width, height = image.size
        left = max(0, min(width, int(float(bbox_value["xmin"]) * width)))
        top = max(0, min(height, int(float(bbox_value["ymin"]) * height)))
        right = max(left + 1, min(width, int(float(bbox_value["xmax"]) * width)))
        bottom = max(top + 1, min(height, int(float(bbox_value["ymax"]) * height)))
        return image.crop((left, top, right, bottom))
    return image


def _extract_tfds_filename(example: dict[str, Any], split_name: str, index: int) -> str:
    for key in ("file_name", "filename", "image/filename", "image_id", "id"):
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        text = str(value)
        if text:
            return _sanitize_filename(Path(text).name)
    return f"{split_name}_{index:06d}.jpg"


def _resolve_fixed_binary_label(data_cfg: dict[str, Any]) -> int | None:
    value = data_cfg.get("fixed_binary_label")
    if value is None:
        return None
    return 1 if int(value) == 1 else 0


def _target_binary_path(processed_dir: Path, binary_label: int, prefix: str, filename: str) -> Path:
    label_dir = "cat" if binary_label == 1 else "not_cat"
    return processed_dir / label_dir / prefix / filename


def _resolve_binary_label(label_name: str, data_cfg: dict[str, Any]) -> int | None:
    normalized = label_name.strip().lower()
    positive = {item.strip().lower() for item in data_cfg.get("positive_label_names", ["cat"])}
    negative = {item.strip().lower() for item in data_cfg.get("negative_label_names", ["dog", "not_cat", "non_cat"])}
    if normalized in positive:
        return 1
    if normalized in negative:
        return 0
    return None


def _preprocess_directory_binary_dataset(config: dict[str, Any]) -> list[BinarySample]:
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    max_edge = int(config["data"].get("max_edge", 300))
    extensions = config["data"].get("image_extensions", [".jpg", ".jpeg", ".png"])
    prefix = _dataset_prefix(config["data"], "directory")

    processed_samples: list[BinarySample] = []
    for sample in scan_binary_samples(raw_dir, extensions):
        source = Path(sample.path)
        target = _target_binary_path(processed_dir, sample.label, prefix, source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            image = read_image(source)
            image = resize_keep_ratio(image, max_edge)
            save_image(image, target)
        processed_samples.append(BinarySample(path=str(target), label=sample.label))
    return processed_samples


def _preprocess_tfds_binary_dataset(config: dict[str, Any]) -> list[BinarySample]:
    from PIL import Image

    tfds = require("tensorflow_datasets", "tensorflow-datasets")

    data_cfg = config["data"]
    source = _binary_source(config)
    default_dataset_names = {"tfds_cats_vs_dogs": "cats_vs_dogs", "tfds_oxford_iiit_pet": "oxford_iiit_pet"}
    dataset_name = str(data_cfg.get("tfds_name") or default_dataset_names.get(source, "")).strip()
    if not dataset_name:
        raise ValueError("TFDS binary dataset source requires data.tfds_name or a known preset source.")
    label_key = str(data_cfg.get("label_key") or ("species" if dataset_name == "oxford_iiit_pet" else "label"))
    image_key = str(data_cfg.get("image_key", "image"))
    bbox_key = str(data_cfg.get("bbox_key") or ("head_bbox" if data_cfg.get("use_head_bbox", False) else "")).strip()
    split_names = list(data_cfg.get("tfds_splits", ["train"]))
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    max_edge = int(data_cfg.get("max_edge", 300))
    max_samples = int(data_cfg.get("max_samples", 0))
    per_label_limit = int(data_cfg.get("max_samples_per_label", 0))
    fixed_binary_label = _resolve_fixed_binary_label(data_cfg)
    prefix = _dataset_prefix(data_cfg, source)

    builder = tfds.builder(dataset_name, data_dir=data_cfg.get("tfds_data_dir"))
    builder.download_and_prepare()
    label_feature = builder.info.features.get(label_key) if fixed_binary_label is None else None

    processed_samples: list[BinarySample] = []
    exported_per_label = {0: 0, 1: 0}
    for split_name in split_names:
        dataset = builder.as_dataset(split=split_name, shuffle_files=False)
        for index, example in enumerate(tfds.as_numpy(dataset)):
            if fixed_binary_label is not None:
                binary_label = fixed_binary_label
            else:
                raw_label = int(example[label_key])
                label_name = label_feature.int2str(raw_label) if hasattr(label_feature, "int2str") else str(raw_label)
                binary_label = _resolve_binary_label(label_name, data_cfg)
                if binary_label is None:
                    continue
            if per_label_limit and exported_per_label[binary_label] >= per_label_limit:
                continue
            image_array = example[image_key]
            image = Image.fromarray(image_array).convert("RGB")
            if bbox_key:
                image = _crop_with_normalized_bbox(image, example.get(bbox_key))
            image = resize_keep_ratio(image, max_edge)
            filename = _extract_tfds_filename(example, split_name, len(processed_samples))
            target = _target_binary_path(processed_dir, binary_label, prefix, filename)
            save_image(image, target)
            processed_samples.append(BinarySample(path=str(target), label=binary_label))
            exported_per_label[binary_label] += 1
            if max_samples and len(processed_samples) >= max_samples:
                return processed_samples
    return processed_samples


def _preprocess_mixed_binary_dataset(config: dict[str, Any]) -> list[BinarySample]:
    data_cfg = config["data"]
    source_items = list(data_cfg.get("sources", []))
    if not source_items:
        raise ValueError("Mixed binary dataset source requires data.sources.")

    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    merged_samples: list[BinarySample] = []
    for index, source_item in enumerate(source_items, start=1):
        item_cfg = dict(source_item)
        item_cfg.setdefault("processed_dir", str(processed_dir))
        item_cfg.setdefault("max_edge", data_cfg.get("max_edge", 300))
        item_cfg.setdefault("image_extensions", data_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
        item_cfg.setdefault("tfds_data_dir", data_cfg.get("tfds_data_dir"))
        item_cfg.setdefault("filename_prefix", item_cfg.get("filename_prefix") or f"source_{index:02d}")
        nested_config = {"data": item_cfg}
        source_name = str(item_cfg.get("source", "")).strip().lower()
        if source_name == "directory":
            merged_samples.extend(_preprocess_directory_binary_dataset(nested_config))
        elif source_name in {"tfds", "tfds_cats_vs_dogs", "tfds_oxford_iiit_pet"}:
            merged_samples.extend(_preprocess_tfds_binary_dataset(nested_config))
        else:
            raise ValueError(f"Unsupported mixed binary dataset source: {source_name}")
    return merged_samples


def scan_binary_samples(raw_dir: str | Path, extensions: list[str]) -> list[BinarySample]:
    root = Path(raw_dir)
    samples: list[BinarySample] = []
    for label_name, label in (("cat", 1), ("not_cat", 0)):
        label_root = root / label_name
        for image_path in iter_image_files(label_root, extensions):
            samples.append(BinarySample(path=str(image_path), label=label))
    return sorted(samples, key=lambda item: item.path)


def split_binary_samples(samples: list[BinarySample], val_ratio: float, seed: int) -> tuple[list[BinarySample], list[BinarySample]]:
    grouped = {
        1: [sample for sample in samples if sample.label == 1],
        0: [sample for sample in samples if sample.label == 0],
    }
    rng = random.Random(seed)
    train_samples: list[BinarySample] = []
    val_samples: list[BinarySample] = []
    for label, label_samples in grouped.items():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        if not shuffled:
            continue
        val_size = max(1, int(len(shuffled) * val_ratio))
        if val_size >= len(shuffled):
            val_size = len(shuffled) - 1 if len(shuffled) > 1 else 1
        val_chunk = shuffled[:val_size]
        train_chunk = shuffled[val_size:]
        if not train_chunk and val_chunk:
            train_chunk = val_chunk[:1]
            val_chunk = val_chunk[1:]
        val_samples.extend(BinarySample(path=item.path, label=label, split="val") for item in val_chunk)
        train_samples.extend(BinarySample(path=item.path, label=label, split="train") for item in train_chunk)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def preprocess_binary_dataset(config: dict[str, Any]) -> list[BinarySample]:
    source = _binary_source(config)
    if source == "directory":
        return _preprocess_directory_binary_dataset(config)
    if source in {"tfds", "tfds_cats_vs_dogs", "tfds_oxford_iiit_pet"}:
        return _preprocess_tfds_binary_dataset(config)
    if source == "mixed":
        return _preprocess_mixed_binary_dataset(config)
    raise ValueError(f"Unsupported binary dataset source: {source}")


def summarize_binary_samples(train_samples: list[BinarySample], val_samples: list[BinarySample]) -> dict[str, Any]:
    return {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_cat": sum(sample.label for sample in train_samples),
        "train_not_cat": sum(1 for sample in train_samples if sample.label == 0),
        "val_cat": sum(sample.label for sample in val_samples),
        "val_not_cat": sum(1 for sample in val_samples if sample.label == 0),
    }


def build_binary_tf_datasets(
    train_samples: list[BinarySample],
    val_samples: list[BinarySample],
    config: dict[str, Any],
):
    tf = require("tensorflow")
    model_cfg = config["model"]
    train_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})
    input_size = tuple(model_cfg.get("input_size", [224, 224]))

    def _dataset_from_samples(samples: list[BinarySample], training: bool):
        paths = [sample.path for sample in samples]
        labels = [sample.label for sample in samples]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training and samples:
            dataset = dataset.shuffle(buffer_size=len(samples), seed=int(train_cfg.get("random_seed", 42)))

        def _load(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_with_pad(image, input_size[0], input_size[1])
            return image, tf.cast(label, tf.float32)

        dataset = dataset.map(
            _load,
            num_parallel_calls=int(train_cfg.get("num_parallel_calls", 4)),
        )
        dataset = dataset.batch(int(train_cfg.get("batch_size", 32)))

        if training:
            augmenter_layers = []
            if aug_cfg.get("horizontal_flip", True):
                augmenter_layers.append(tf.keras.layers.RandomFlip("horizontal"))
            if aug_cfg.get("rotation_factor", 0.0):
                augmenter_layers.append(tf.keras.layers.RandomRotation(float(aug_cfg["rotation_factor"])))
            if aug_cfg.get("zoom_factor", 0.0):
                augmenter_layers.append(
                    tf.keras.layers.RandomZoom(
                        height_factor=float(aug_cfg["zoom_factor"]),
                        width_factor=float(aug_cfg["zoom_factor"]),
                    )
                )
            augmenter = tf.keras.Sequential(augmenter_layers) if augmenter_layers else None

            def _augment(images, labels):
                if augmenter is not None:
                    images = augmenter(images, training=True)
                if aug_cfg.get("brightness_delta", 0.0):
                    images = tf.image.random_brightness(images, max_delta=float(aug_cfg["brightness_delta"]))
                return images, labels

            dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.prefetch(int(train_cfg.get("prefetch_buffer", 2)))

    return _dataset_from_samples(train_samples, True), _dataset_from_samples(val_samples, False)

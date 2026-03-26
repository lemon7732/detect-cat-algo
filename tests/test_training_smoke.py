from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from cat_rescue_ai.training.binary_trainer import train_binary_model
from cat_rescue_ai.training.landmark_trainer import train_landmark_model


pytest.importorskip("tensorflow")


def _save_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def _make_square_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), color=color)
    image.save(path)


def _make_landmark_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (128, 128), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 40, 42, 52), fill="black")
    draw.ellipse((82, 40, 94, 52), fill="black")
    draw.ellipse((58, 74, 70, 86), fill="black")
    draw.polygon([(18, 24), (32, 8), (48, 30)], outline="black", fill=(220, 220, 220))
    draw.polygon([(80, 30), (96, 8), (110, 24)], outline="black", fill=(220, 220, 220))
    image.save(path)


def test_binary_training_smoke(tmp_path: Path):
    for index in range(4):
        _make_square_image(tmp_path / "data/binary/raw/cat" / f"cat_{index}.png", (20, 20 + index * 10, 20))
        _make_square_image(tmp_path / "data/binary/raw/not_cat" / f"not_{index}.png", (220, 220 - index * 10, 220))

    config = {
        "project_name": "test-binary",
        "logging": {"level": "INFO"},
        "seed": 42,
        "model": {"name": "resnet_reference", "input_size": [64, 64], "dropout": 0.1},
        "data": {
            "raw_dir": str(tmp_path / "data/binary/raw"),
            "processed_dir": str(tmp_path / "data/binary/processed"),
            "max_edge": 96,
            "image_extensions": [".png"],
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "val_ratio": 0.25,
            "random_seed": 42,
            "early_stopping_patience": 1,
            "reduce_lr_patience": 1,
            "num_parallel_calls": 1,
            "prefetch_buffer": 1,
        },
        "augmentation": {
            "horizontal_flip": False,
            "rotation_factor": 0.0,
            "zoom_factor": 0.0,
            "brightness_delta": 0.0,
        },
        "artifacts": {
            "output_dir": str(tmp_path / "artifacts/binary"),
            "best_weights": str(tmp_path / "artifacts/binary/best.weights.h5"),
            "final_weights": str(tmp_path / "artifacts/binary/final.weights.h5"),
            "history_csv": str(tmp_path / "artifacts/binary/history.csv"),
            "history_plot": str(tmp_path / "artifacts/binary/history.png"),
            "eval_json": str(tmp_path / "artifacts/binary/eval.json"),
            "predictions_csv": str(tmp_path / "artifacts/binary/test_predictions.csv"),
        },
    }
    config_path = _save_yaml(tmp_path / "binary.yaml", config)
    result = train_binary_model(config_path)
    assert "metrics" in result
    assert Path(config["artifacts"]["best_weights"]).exists()
    assert Path(config["artifacts"]["final_weights"]).exists()


def test_landmark_training_smoke(tmp_path: Path):
    image_dir = tmp_path / "data/landmarks/images"
    csv_path = tmp_path / "data/landmarks/train.csv"
    rows = []
    for index in range(4):
        image_name = f"landmark_{index}.png"
        _make_landmark_image(image_dir / image_name)
        rows.append(
            {
                "image_id": image_name,
                "left_eye_x": 36,
                "left_eye_y": 46,
                "right_eye_x": 88,
                "right_eye_y": 46,
                "mouth_x": 64,
                "mouth_y": 80,
                "left_ear1_x": 24,
                "left_ear1_y": 28,
                "left_ear2_x": 32,
                "left_ear2_y": 12,
                "left_ear3_x": 44,
                "left_ear3_y": 30,
                "right_ear1_x": 84,
                "right_ear1_y": 30,
                "right_ear2_x": 96,
                "right_ear2_y": 12,
                "right_ear3_x": 104,
                "right_ear3_y": 28,
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "project_name": "test-landmarks",
        "logging": {"level": "INFO"},
        "seed": 42,
        "model": {"name": "vgg_regressor", "input_size": [64, 64], "dense_units": [64], "dropout": 0.1},
        "data": {
            "image_dir": str(image_dir),
            "csv_path": str(csv_path),
            "processed_dir": str(tmp_path / "data/landmarks/processed"),
            "image_extensions": [".png"],
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "val_ratio": 0.25,
            "random_seed": 42,
            "early_stopping_patience": 1,
            "reduce_lr_patience": 1,
            "num_parallel_calls": 1,
            "prefetch_buffer": 1,
        },
        "augmentation": {"horizontal_flip": False, "brightness_delta": 0.0},
        "artifacts": {
            "output_dir": str(tmp_path / "artifacts/landmarks"),
            "best_weights": str(tmp_path / "artifacts/landmarks/best.weights.h5"),
            "final_weights": str(tmp_path / "artifacts/landmarks/final.weights.h5"),
            "history_csv": str(tmp_path / "artifacts/landmarks/history.csv"),
            "history_plot": str(tmp_path / "artifacts/landmarks/history.png"),
            "eval_json": str(tmp_path / "artifacts/landmarks/eval.json"),
            "predictions_csv": str(tmp_path / "artifacts/landmarks/val_predictions.csv"),
            "preview_dir": str(tmp_path / "artifacts/landmarks/previews"),
        },
    }
    config_path = _save_yaml(tmp_path / "landmarks.yaml", config)
    result = train_landmark_model(config_path, cascade_path="assets/cascades/missing.xml")
    assert "metrics" in result
    assert Path(config["artifacts"]["best_weights"]).exists()
    assert Path(config["artifacts"]["final_weights"]).exists()

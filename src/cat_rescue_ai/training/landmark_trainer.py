"""Training workflow for the VGG-style landmark regressor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cat_rescue_ai.config import load_config
from cat_rescue_ai.datasets.landmark_dataset import (
    build_landmark_tf_datasets,
    preprocess_landmark_dataset,
    split_landmark_samples,
    summarize_landmark_samples,
)
from cat_rescue_ai.detection.cat_face import CatFaceDetector
from cat_rescue_ai.logging_utils import setup_logging
from cat_rescue_ai.models.landmarks import build_vgg_landmark_model
from cat_rescue_ai.utils.coords import denormalize_landmarks, mean_absolute_error, root_mean_squared_error
from cat_rescue_ai.utils.deps import require
from cat_rescue_ai.utils.image import read_image
from cat_rescue_ai.utils.io import export_model_summary, export_training_curves, save_csv, save_json
from cat_rescue_ai.utils.seeding import set_global_seed


def _compile_landmark_model(model, config: dict[str, Any]):
    tf = require("tensorflow")
    learning_rate = float(config["training"].get("learning_rate", 5e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.MeanSquaredError(name="mse")],
    )


def _visualize_predictions(samples, predictions, preview_dir: Path) -> None:
    from PIL import ImageDraw

    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, predicted) in enumerate(zip(samples[:10], predictions[:10], strict=False), start=1):
        image = read_image(sample.image_path)
        width, height = image.size
        true_points = denormalize_landmarks(sample.landmarks, width, height)
        pred_points = denormalize_landmarks(predicted, width, height)
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for point_index in range(0, len(true_points), 2):
            tx, ty = true_points[point_index], true_points[point_index + 1]
            px, py = pred_points[point_index], pred_points[point_index + 1]
            draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill="green")
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="red")
        canvas.save(preview_dir / f"preview_{index:02d}.png")


def train_landmark_model(config_path: str | Path, cascade_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    logger = setup_logging(config)
    set_global_seed(int(config["training"].get("random_seed", config.get("seed", 42))))

    detector = None
    try:
        detector = CatFaceDetector(cascade_path=cascade_path)
    except Exception as exc:
        logger.warning("Landmark preprocessing will fall back to full images because detector is unavailable: %s", exc)

    samples = preprocess_landmark_dataset(config, detector=detector)
    train_samples, val_samples = split_landmark_samples(
        samples,
        val_ratio=float(config["training"].get("val_ratio", 0.2)),
        seed=int(config["training"].get("random_seed", 42)),
    )
    logger.info("Prepared landmark dataset: %s", summarize_landmark_samples(train_samples, val_samples))

    train_ds, val_ds = build_landmark_tf_datasets(train_samples, val_samples, config)
    model = build_vgg_landmark_model(config)
    _compile_landmark_model(model, config)

    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    export_model_summary(model, output_dir / "model_summary.txt")

    tf = require("tensorflow")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["artifacts"]["best_weights"],
            monitor="val_mae",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=int(config["training"].get("early_stopping_patience", 6)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(config["training"].get("reduce_lr_patience", 3)),
            factor=0.5,
        ),
        tf.keras.callbacks.CSVLogger(config["artifacts"]["history_csv"]),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=int(config["training"].get("epochs", 30)), callbacks=callbacks)
    model.save_weights(config["artifacts"]["final_weights"])

    predictions = model.predict(val_ds, verbose=0).tolist()
    rows = []
    maes = []
    rmses = []
    for sample, predicted in zip(val_samples, predictions, strict=False):
        mae = mean_absolute_error(predicted, sample.landmarks)
        rmse = root_mean_squared_error(predicted, sample.landmarks)
        maes.append(mae)
        rmses.append(rmse)
        rows.append({"image_path": sample.image_path, "mae": mae, "rmse": rmse})
    save_csv(config["artifacts"]["predictions_csv"], rows)
    _visualize_predictions(val_samples, predictions, Path(config["artifacts"]["preview_dir"]))

    history_dict = {key: list(map(float, values)) for key, values in history.history.items()}
    export_training_curves(history_dict, config["artifacts"]["history_plot"])
    metrics = {
        "mean_absolute_error": sum(maes) / len(maes) if maes else None,
        "root_mean_squared_error": sum(rmses) / len(rmses) if rmses else None,
    }
    save_json(config["artifacts"]["eval_json"], {"metrics": metrics})
    return {"metrics": metrics, "history": history_dict}


def evaluate_landmark_model(config_path: str | Path, cascade_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    detector = None
    try:
        detector = CatFaceDetector(cascade_path=cascade_path)
    except Exception:
        detector = None
    samples = preprocess_landmark_dataset(config, detector=detector)
    _, val_samples = split_landmark_samples(
        samples,
        val_ratio=float(config["training"].get("val_ratio", 0.2)),
        seed=int(config["training"].get("random_seed", 42)),
    )
    _, val_ds = build_landmark_tf_datasets([], val_samples, config)
    model = build_vgg_landmark_model(config)
    _compile_landmark_model(model, config)
    model.load_weights(config["artifacts"]["best_weights"])
    predictions = model.predict(val_ds, verbose=0).tolist()
    maes = [mean_absolute_error(prediction, sample.landmarks) for sample, prediction in zip(val_samples, predictions, strict=False)]
    metrics = {"mean_absolute_error": sum(maes) / len(maes) if maes else None}
    save_json(config["artifacts"]["eval_json"], {"metrics": metrics})
    return metrics

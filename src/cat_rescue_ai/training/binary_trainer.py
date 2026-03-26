"""Training and evaluation workflow for cat-vs-not-cat classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cat_rescue_ai.config import load_config
from cat_rescue_ai.datasets.binary_dataset import (
    build_binary_tf_datasets,
    preprocess_binary_dataset,
    split_binary_samples,
    summarize_binary_samples,
)
from cat_rescue_ai.logging_utils import setup_logging
from cat_rescue_ai.models.binary import build_binary_model
from cat_rescue_ai.utils.deps import require
from cat_rescue_ai.utils.io import export_model_summary, export_training_curves, save_csv, save_json
from cat_rescue_ai.utils.seeding import set_global_seed


def _compile_binary_model(model, config: dict[str, Any]):
    tf = require("tensorflow")
    learning_rate = float(config["training"].get("learning_rate", 3e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def train_binary_model(config_path: str | Path) -> dict[str, Any]:
    tf = require("tensorflow")
    config = load_config(config_path)
    logger = setup_logging(config)
    set_global_seed(int(config["training"].get("random_seed", config.get("seed", 42))))

    processed_samples = preprocess_binary_dataset(config)
    train_samples, val_samples = split_binary_samples(
        processed_samples,
        val_ratio=float(config["training"].get("val_ratio", 0.2)),
        seed=int(config["training"].get("random_seed", 42)),
    )
    logger.info("Prepared binary dataset: %s", summarize_binary_samples(train_samples, val_samples))

    train_ds, val_ds = build_binary_tf_datasets(train_samples, val_samples, config)
    model = build_binary_model(config)
    _compile_binary_model(model, config)

    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    export_model_summary(model, output_dir / "model_summary.txt")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["artifacts"]["best_weights"],
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=int(config["training"].get("early_stopping_patience", 4)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(config["training"].get("reduce_lr_patience", 2)),
            factor=0.5,
        ),
        tf.keras.callbacks.CSVLogger(config["artifacts"]["history_csv"]),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=int(config["training"].get("epochs", 10)), callbacks=callbacks)
    model.save_weights(config["artifacts"]["final_weights"])

    metrics = dict(zip(model.metrics_names, model.evaluate(val_ds, return_dict=False), strict=False))
    history_dict = {key: list(map(float, values)) for key, values in history.history.items()}
    export_training_curves(history_dict, config["artifacts"]["history_plot"])
    save_json(config["artifacts"]["eval_json"], {"metrics": metrics, "dataset": summarize_binary_samples(train_samples, val_samples)})

    prediction_rows = []
    probabilities = model.predict(val_ds, verbose=0).reshape(-1).tolist()
    labels = [sample.label for sample in val_samples]
    for sample, label, probability in zip(val_samples, labels, probabilities, strict=False):
        prediction_rows.append(
            {
                "path": sample.path,
                "label": label,
                "probability_cat": float(probability),
                "predicted_label": int(probability >= 0.5),
            }
        )
    save_csv(config["artifacts"]["predictions_csv"], prediction_rows)
    return {"metrics": metrics, "history": history_dict}


def evaluate_binary_model(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    processed_samples = preprocess_binary_dataset(config)
    _, val_samples = split_binary_samples(
        processed_samples,
        val_ratio=float(config["training"].get("val_ratio", 0.2)),
        seed=int(config["training"].get("random_seed", 42)),
    )
    _, val_ds = build_binary_tf_datasets([], val_samples, config)
    model = build_binary_model(config)
    _compile_binary_model(model, config)
    model.load_weights(config["artifacts"]["best_weights"])
    metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    save_json(config["artifacts"]["eval_json"], {"metrics": metrics})
    return metrics

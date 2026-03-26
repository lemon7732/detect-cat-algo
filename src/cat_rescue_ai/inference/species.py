"""Single-image species prediction helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cat_rescue_ai.config import load_config
from cat_rescue_ai.models.binary import build_binary_model
from cat_rescue_ai.utils.deps import require
from cat_rescue_ai.utils.image import letterbox, read_image


def _prepare_binary_image(config: dict[str, Any], image):
    tf = require("tensorflow")
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    image = letterbox(image, input_size)
    tensor = tf.keras.utils.img_to_array(image) / 255.0
    return tf.expand_dims(tensor, axis=0)


def load_binary_predictor(config_path: str | Path):
    config = load_config(config_path)
    model = build_binary_model(config)
    model.load_weights(config["artifacts"]["best_weights"])
    return config, model


def predict_species(config_path: str | Path, image_source: Any) -> dict[str, Any]:
    config, model = load_binary_predictor(config_path)
    image = read_image(image_source)
    tensor = _prepare_binary_image(config, image)
    probability = float(model.predict(tensor, verbose=0)[0][0])
    return {"is_cat": probability >= 0.5, "cat_probability": probability}

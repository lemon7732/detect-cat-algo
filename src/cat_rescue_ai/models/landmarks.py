"""VGG-style cat face landmark regressor."""

from __future__ import annotations

from typing import Any

from cat_rescue_ai.utils.deps import require


def _vgg_block(tf, x, filters: int, convs: int):
    for _ in range(convs):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)


def build_vgg_landmark_model(config: dict[str, Any]):
    tf = require("tensorflow")
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    dense_units = config["model"].get("dense_units", [512, 128])
    dropout = float(config["model"].get("dropout", 0.3))

    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    x = inputs
    for filters, convs in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
        x = _vgg_block(tf, x, filters, convs)
    x = tf.keras.layers.Flatten()(x)
    for units in dense_units:
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(18, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="vgg_landmark_regressor")

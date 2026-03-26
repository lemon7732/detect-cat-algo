"""Binary classification model factories."""

from __future__ import annotations

from typing import Any

from cat_rescue_ai.utils.deps import require


def _se_block(tf, x, filters: int, ratio: int = 8):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
    excitation = tf.keras.layers.Dense(max(filters // ratio, 8), activation="relu")(squeeze)
    excitation = tf.keras.layers.Dense(filters, activation="sigmoid")(excitation)
    excitation = tf.keras.layers.Reshape((1, 1, filters))(excitation)
    return tf.keras.layers.Multiply()([x, excitation])


def _reference_residual_block(tf, x, filters: int, stride: int = 1):
    shortcut = x
    y = tf.keras.layers.BatchNormalization()(x)
    y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
    y = tf.keras.layers.Add()([shortcut, y])
    return tf.keras.layers.ReLU()(y)


def _f_residual_block(tf, x, filters: int, stride: int = 1):
    shortcut = x
    y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = _se_block(tf, y, filters)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    y = tf.keras.layers.Add()([shortcut, y])
    return tf.keras.layers.ReLU()(y)


def build_resnet50_transfer(config: dict[str, Any]):
    tf = require("tensorflow")
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=config["model"].get("weights"),
        input_shape=(input_size[0], input_size[1], 3),
    )
    base.trainable = not bool(config["model"].get("freeze_backbone", False))
    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(float(config["model"].get("dropout", 0.3)))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="resnet50_transfer")


def build_resnet_reference(config: dict[str, Any]):
    tf = require("tensorflow")
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    x = tf.keras.layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    for filters, blocks in ((32, 2), (64, 2), (128, 2), (256, 2)):
        for block_index in range(blocks):
            x = _reference_residual_block(tf, x, filters, stride=2 if block_index == 0 and filters != 32 else 1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(float(config["model"].get("dropout", 0.3)))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="resnet_reference")


def build_f_resnet_se(config: dict[str, Any]):
    tf = require("tensorflow")
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    x = tf.keras.layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    for filters, blocks in ((32, 2), (64, 2), (128, 2), (256, 2)):
        for block_index in range(blocks):
            x = _f_residual_block(tf, x, filters, stride=2 if block_index == 0 and filters != 32 else 1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(float(config["model"].get("dropout", 0.3)))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="f_resnet_se")


def build_mini_cnn(config: dict[str, Any]):
    tf = require("tensorflow")
    input_size = tuple(config["model"].get("input_size", [224, 224]))
    dropout = float(config["model"].get("dropout", 0.2))
    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="mini_cnn")


def build_binary_model(config: dict[str, Any]):
    model_name = config["model"].get("name", "f_resnet_se")
    if model_name == "mini_cnn":
        return build_mini_cnn(config)
    if model_name == "resnet50_transfer":
        return build_resnet50_transfer(config)
    if model_name == "resnet_reference":
        return build_resnet_reference(config)
    if model_name == "f_resnet_se":
        return build_f_resnet_se(config)
    raise ValueError(f"Unsupported binary model: {model_name}")

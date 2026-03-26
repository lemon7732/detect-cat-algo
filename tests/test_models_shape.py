import pytest


tf = pytest.importorskip("tensorflow")

from cat_rescue_ai.models.binary import build_f_resnet_se, build_mini_cnn
from cat_rescue_ai.models.landmarks import build_vgg_landmark_model


def test_f_resnet_output_shape():
    model = build_f_resnet_se({"model": {"input_size": [224, 224], "dropout": 0.3}})
    assert model.output_shape == (None, 1)


def test_landmark_model_output_shape():
    model = build_vgg_landmark_model({"model": {"input_size": [224, 224], "dense_units": [128], "dropout": 0.2}})
    assert model.output_shape == (None, 18)


def test_mini_cnn_output_shape():
    model = build_mini_cnn({"model": {"input_size": [64, 64], "dropout": 0.2}})
    assert model.output_shape == (None, 1)

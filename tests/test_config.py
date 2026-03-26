from cat_rescue_ai.config import deep_merge
from cat_rescue_ai.datasets.binary_dataset import BinarySample, split_binary_samples


def test_deep_merge_overrides_nested_values():
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4})
    assert merged == {"a": {"b": 1, "c": 3}, "d": 4}


def test_split_binary_samples_is_stratified():
    samples = [BinarySample(f"cat_{i}.png", 1) for i in range(6)] + [BinarySample(f"not_{i}.png", 0) for i in range(6)]
    train_samples, val_samples = split_binary_samples(samples, val_ratio=0.25, seed=42)
    assert sum(sample.label for sample in val_samples) == 1
    assert sum(1 for sample in val_samples if sample.label == 0) == 1

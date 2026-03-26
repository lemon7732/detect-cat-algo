"""Random seed helpers."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except ModuleNotFoundError:
        pass

from __future__ import annotations

import json
import sys


def main() -> None:
    report = {
        "python_version": sys.version,
        "python_ok_for_full_tensorflow_training": sys.version_info < (3, 13),
        "modules": {},
    }
    for module_name in [
        "numpy",
        "cv2",
        "yaml",
        "PIL",
        "fastapi",
        "pydantic",
        "uvicorn",
        "pytest",
        "tensorflow",
        "tensorflow_datasets",
        "kaggle",
        "wildlife_datasets",
    ]:
        try:
            __import__(module_name)
            report["modules"][module_name] = "ok"
        except Exception as exc:
            report["modules"][module_name] = f"missing: {type(exc).__name__}"
    if sys.version_info >= (3, 13):
        report["note"] = (
            "Current interpreter is newer than the TensorFlow 2.19 supported range. "
            "Core detection/API utilities can run, but full training/inference with TensorFlow "
            "should use Python 3.10-3.12."
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

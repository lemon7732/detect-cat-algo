"""OpenCV Haar-based cat face detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cat_rescue_ai.exceptions import CatFaceNotFoundError, ConfigError
from cat_rescue_ai.utils.image import read_image


def resolve_cascade_path(preferred: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.append(Path("assets/cascades/haarcascade_frontalcatface.xml"))
    try:
        import cv2  # type: ignore

        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalcatface.xml")
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalcatface_extended.xml")
    except ModuleNotFoundError:
        pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ConfigError(
        "Unable to locate haarcascade_frontalcatface.xml. Place it under assets/cascades/ or configure cascade_path."
    )


class CatFaceDetector:
    def __init__(
        self,
        cascade_path: str | Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: tuple[int, int] = (32, 32),
    ) -> None:
        import cv2  # type: ignore

        self.cv2 = cv2
        self.cascade_path = resolve_cascade_path(cascade_path)
        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        if self.classifier.empty():
            raise ConfigError(f"Failed to load cascade classifier: {self.cascade_path}")

    def detect(self, image: Any) -> dict[str, Any]:
        import numpy as np  # type: ignore

        pil_image = read_image(image) if isinstance(image, (str, Path, bytes)) or hasattr(image, "read") else image
        rgb = np.array(pil_image)
        gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        boxes = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]
        primary = None
        if boxes:
            primary = max(boxes, key=lambda box: box["w"] * box["h"])
        return {"faces": boxes, "primary_bbox": [primary["x"], primary["y"], primary["w"], primary["h"]] if primary else None}

    def detect_path(self, image_path: str | Path) -> dict[str, Any]:
        return self.detect(str(image_path))

    def require_primary_face(self, image: Any) -> list[int]:
        result = self.detect(image)
        bbox = result.get("primary_bbox")
        if not bbox:
            raise CatFaceNotFoundError("No cat face detected in image.")
        return bbox

    def crop_primary_face(self, image: Any):
        pil_image = read_image(image) if isinstance(image, (str, Path, bytes)) or hasattr(image, "read") else image
        bbox = self.require_primary_face(pil_image)
        x, y, w, h = bbox
        return pil_image.crop((x, y, x + w, y + h)), bbox

    def draw_detections(self, image: Any):
        from PIL import ImageDraw

        pil_image = read_image(image) if isinstance(image, (str, Path, bytes)) or hasattr(image, "read") else image
        result = self.detect(pil_image)
        canvas = pil_image.copy()
        draw = ImageDraw.Draw(canvas)
        for box in result["faces"]:
            draw.rectangle(
                [(box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"])],
                outline="red",
                width=3,
            )
        return canvas, result

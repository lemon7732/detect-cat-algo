"""Image I/O and preprocessing helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Iterable

from cat_rescue_ai.exceptions import InvalidImageError


def _open_pillow_image(source: str | Path | bytes | BinaryIO):
    from PIL import Image, ImageOps  # lazy import

    try:
        if hasattr(source, "read"):
            payload = source.read()
            image = Image.open(io.BytesIO(payload))
        elif isinstance(source, bytes):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(Path(source))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:  # pragma: no cover - depends on Pillow internals
        raise InvalidImageError(f"Failed to decode image: {source}") from exc


def read_image(source: str | Path | bytes | BinaryIO):
    return _open_pillow_image(source)


def resize_keep_ratio(image, max_edge: int):
    width, height = image.size
    if max(width, height) <= max_edge:
        return image
    scale = float(max_edge) / float(max(width, height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size)


def letterbox(image, target_size: tuple[int, int], fill_color: tuple[int, int, int] = (0, 0, 0)):
    from PIL import Image

    target_w, target_h = target_size
    resized = resize_keep_ratio(image, max_edge=max(target_size))
    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    offset = ((target_w - resized.size[0]) // 2, (target_h - resized.size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def save_image(image, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def iter_image_files(root: str | Path, extensions: Iterable[str]) -> list[Path]:
    root_path = Path(root)
    allowed = {suffix.lower() for suffix in extensions}
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )

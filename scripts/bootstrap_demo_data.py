from __future__ import annotations

import csv
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw


def draw_cat_face(path: Path, variant: str, jitter: int, size: int = 128) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), color=(245, 241, 232))
    draw = ImageDraw.Draw(image)

    left_eye_x = 36 + jitter
    right_eye_x = 88 + jitter
    eye_y = 48 + jitter // 3
    mouth_x = 64 + jitter
    mouth_y = 82 + jitter // 4

    if variant == "tabby":
        fur = (206, 152, 98)
        stripe = (120, 84, 52)
        left_ear = [(22 + jitter, 30), (36 + jitter, 8), (52 + jitter, 34)]
        right_ear = [(76 + jitter, 34), (92 + jitter, 8), (108 + jitter, 30)]
    else:
        fur = (112, 112, 122)
        stripe = (72, 72, 82)
        left_ear = [(18 + jitter, 34), (34 + jitter, 10), (48 + jitter, 38)]
        right_ear = [(80 + jitter, 38), (96 + jitter, 10), (112 + jitter, 34)]

    draw.ellipse((18 + jitter, 24, 110 + jitter, 116), fill=fur, outline=stripe, width=3)
    draw.polygon(left_ear, fill=fur, outline=stripe)
    draw.polygon(right_ear, fill=fur, outline=stripe)
    draw.ellipse((left_eye_x - 6, eye_y - 5, left_eye_x + 6, eye_y + 5), fill=(18, 18, 18))
    draw.ellipse((right_eye_x - 6, eye_y - 5, right_eye_x + 6, eye_y + 5), fill=(18, 18, 18))
    draw.polygon([(mouth_x - 5, mouth_y - 2), (mouth_x + 5, mouth_y - 2), (mouth_x, mouth_y + 6)], fill=(40, 40, 40))
    draw.line((mouth_x, mouth_y + 6, mouth_x - 8, mouth_y + 14), fill=(60, 60, 60), width=2)
    draw.line((mouth_x, mouth_y + 6, mouth_x + 8, mouth_y + 14), fill=(60, 60, 60), width=2)

    if variant == "tabby":
        draw.line((54 + jitter, 24, 64 + jitter, 54), fill=stripe, width=4)
        draw.line((74 + jitter, 24, 64 + jitter, 54), fill=stripe, width=4)
    else:
        draw.arc((42 + jitter, 28, 86 + jitter, 76), start=180, end=360, fill=stripe, width=4)

    image.save(path)
    return {
        "image_id": path.name,
        "left_eye_x": left_eye_x,
        "left_eye_y": eye_y,
        "right_eye_x": right_eye_x,
        "right_eye_y": eye_y,
        "mouth_x": mouth_x,
        "mouth_y": mouth_y,
        "left_ear1_x": left_ear[0][0],
        "left_ear1_y": left_ear[0][1],
        "left_ear2_x": left_ear[1][0],
        "left_ear2_y": left_ear[1][1],
        "left_ear3_x": left_ear[2][0],
        "left_ear3_y": left_ear[2][1],
        "right_ear1_x": right_ear[0][0],
        "right_ear1_y": right_ear[0][1],
        "right_ear2_x": right_ear[1][0],
        "right_ear2_y": right_ear[1][1],
        "right_ear3_x": right_ear[2][0],
        "right_ear3_y": right_ear[2][1],
    }


def draw_non_cat(path: Path, seed: int, size: int = 128) -> None:
    rng = Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    if seed % 2 == 0:
        draw.rectangle((20, 20, 108, 108), fill=(90, 150, 220), outline=(40, 90, 160), width=4)
        draw.rectangle((44, 44, 84, 84), fill=(245, 245, 245))
    else:
        center = (64 + rng.randint(-4, 4), 64 + rng.randint(-4, 4))
        draw.ellipse((center[0] - 34, center[1] - 34, center[0] + 34, center[1] + 34), fill=(90, 190, 120))
        draw.line((24, 104, 104, 24), fill=(255, 255, 255), width=6)
        draw.line((24, 24, 104, 104), fill=(255, 255, 255), width=6)
    image.save(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path("data/demo")
    (root / "binary/raw/cat").mkdir(parents=True, exist_ok=True)
    (root / "binary/raw/not_cat").mkdir(parents=True, exist_ok=True)
    (root / "landmarks/images").mkdir(parents=True, exist_ok=True)
    (root / "gallery").mkdir(parents=True, exist_ok=True)
    (root / "query").mkdir(parents=True, exist_ok=True)

    landmark_rows: list[dict[str, object]] = []
    gallery_rows: list[dict[str, object]] = []

    for index in range(12):
        variant = "tabby" if index < 6 else "shadow"
        jitter = (index % 3) * 2 - 2
        binary_path = root / "binary/raw/cat" / f"{variant}_{index:02d}.png"
        row = draw_cat_face(binary_path, variant=variant, jitter=jitter, size=128)
        landmark_image_path = root / "landmarks/images" / binary_path.name
        draw_cat_face(landmark_image_path, variant=variant, jitter=jitter, size=128)
        landmark_rows.append({**row})

    for index in range(12):
        draw_non_cat(root / "binary/raw/not_cat" / f"not_cat_{index:02d}.png", seed=index)

    for cat_id, variant, offsets in (
        ("cat_tabby", "tabby", [-3, 0, 3]),
        ("cat_shadow", "shadow", [-4, 0, 4]),
    ):
        cat_dir = root / "gallery" / cat_id
        for index, jitter in enumerate(offsets, start=1):
            image_path = cat_dir / f"{index}.png"
            draw_cat_face(image_path, variant=variant, jitter=jitter)
            gallery_rows.append(
                {
                    "cat_id": cat_id,
                    "name": "Tabby" if variant == "tabby" else "Shadow",
                    "sex": "unknown",
                    "age": "adult",
                    "description": f"demo {variant} cat",
                    "image_path": f"{cat_id}/{index}.png",
                }
            )

    draw_cat_face(root / "query" / "query_tabby.png", variant="tabby", jitter=1)
    draw_cat_face(root / "query" / "query_shadow.png", variant="shadow", jitter=-1)

    write_csv(root / "landmarks/train.csv", landmark_rows)
    write_csv(root / "gallery/metadata.csv", gallery_rows)
    print("Demo data generated under data/demo")


if __name__ == "__main__":
    main()

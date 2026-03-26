from pathlib import Path

from cat_rescue_ai.gallery.index import match_against_gallery


def test_match_against_gallery_returns_best_entry(tmp_path: Path):
    payload = {
        "config": {
            "matching": {
                "mode": "image",
                "top_k": 3,
                "cosine_threshold": 0.5,
                "euclidean_threshold": 2.0,
            }
        },
        "entries": [
            {
                "cat_id": "cat-1",
                "name": "Mimi",
                "prototype_vector": [0.0, 0.0],
                "image_vectors": [[1.0, 0.0]],
                "image_paths": ["cat-1/a.jpg"],
            },
            {"cat_id": "cat-2", "name": "Nana", "prototype_vector": [0.0, 1.0]},
        ],
    }
    matched = match_against_gallery([1.0, 0.0], payload)
    assert matched["best"]["cat_id"] == "cat-1"
    assert matched["best"]["match_source"] == "image"
    assert matched["best"]["matched_image_path"] == "cat-1/a.jpg"

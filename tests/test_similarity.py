from cat_rescue_ai.features.similarity import cosine_similarity, euclidean_distance, rank_gallery


def test_euclidean_distance_zero():
    assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0


def test_cosine_similarity_identity():
    assert round(cosine_similarity([1, 0], [1, 0]), 6) == 1.0


def test_rank_gallery_prefers_closest_item():
    gallery = [
        {"cat_id": "a", "prototype_vector": [0.0, 0.0]},
        {"cat_id": "b", "prototype_vector": [1.0, 1.0]},
    ]
    ranked = rank_gallery([1.0, 1.0], gallery, top_k=2)
    assert ranked[0]["cat_id"] == "b"


def test_rank_gallery_can_match_against_image_vectors():
    gallery = [
        {
            "cat_id": "a",
            "prototype_vector": [0.0, 0.0],
            "image_vectors": [[1.0, 1.0]],
            "image_paths": ["a-1.jpg"],
        },
        {"cat_id": "b", "prototype_vector": [0.8, 0.2]},
    ]
    ranked = rank_gallery([1.0, 1.0], gallery, top_k=2, mode="image")
    assert ranked[0]["cat_id"] == "a"
    assert ranked[0]["match_source"] == "image"
    assert ranked[0]["matched_image_path"] == "a-1.jpg"

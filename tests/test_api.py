import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_missing_image_weight_is_renormalized():
    scores = {
        "image": None,
        "text": 0.8,
        "imageText": 0.55,
        "attribute": 0.6,
        "date": 0.7,
        "location": 0.5,
    }

    weights = {
        "image": 0.30,
        "text": 0.25,
        "imageText": 0.10,
        "attribute": 0.20,
        "date": 0.10,
        "location": 0.05,
    }

    result = main.combine_scores(scores, weights)

    expected = (
        0.8 * 0.25
        + 0.55 * 0.10
        + 0.6 * 0.20
        + 0.7 * 0.10
        + 0.5 * 0.05
    ) / 0.70

    assert result == pytest.approx(expected)


def test_average_available_scores():
    assert main.average_available_scores(
        [None, None]
    ) is None

    assert main.average_available_scores(
        [0.6, None]
    ) == pytest.approx(0.6)

    assert main.average_available_scores(
        [0.6, 0.8]
    ) == pytest.approx(0.7)


def test_full_ranking_without_real_models(monkeypatch):
    def fake_text_embedding(text: str):
        if "지갑" in text:
            return np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )

        return np.array(
            [0.0, 1.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        main.models,
        "text_embedding",
        fake_text_embedding,
    )

    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-TEST",
                "title": "검정색 지갑",
                "category": "지갑",
            },
            "candidates": [
                {
                    "itemId": "FOUND-WALLET",
                    "status": "ACTIVE",
                    "title": "검정색 반지갑",
                    "category": "지갑",
                },
                {
                    "itemId": "FOUND-UMBRELLA",
                    "status": "ACTIVE",
                    "title": "파란색 우산",
                    "category": "우산",
                },
            ],
            "excludedItemIds": [],
            "topK": 5,
            "responseLimit": 10,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["totalRankedCount"] == 2
    assert (
        result["topCandidates"][0]["itemId"]
        == "FOUND-WALLET"
    )


def test_excluded_item_is_not_displayed(monkeypatch):
    monkeypatch.setattr(
        main.models,
        "text_embedding",
        lambda text: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-TEST",
                "title": "지갑",
                "category": "지갑",
            },
            "candidates": [
                {
                    "itemId": "FOUND-001",
                    "status": "ACTIVE",
                    "title": "지갑",
                    "category": "지갑",
                },
                {
                    "itemId": "FOUND-002",
                    "status": "ACTIVE",
                    "title": "지갑",
                    "category": "지갑",
                },
            ],
            "excludedItemIds": [
                "FOUND-001"
            ],
            "topK": 5,
            "responseLimit": 10,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["totalRankedCount"] == 2
    assert result["excludedCount"] == 1
    assert len(result["topCandidates"]) == 1
    assert (
        result["topCandidates"][0]["itemId"]
        == "FOUND-002"
    )
    assert (
        result["topCandidates"][0]["globalRank"]
        == 2
    )
    assert (
        result["topCandidates"][0]["displayRank"]
        == 1
    )


def test_search_visual_text_ranks_candidate_images(
    monkeypatch,
):
    monkeypatch.setattr(
        main.models,
        "visual_text_embedding",
        lambda text: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    image_vectors = {
        "wallet.jpg": np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        "umbrella.jpg": np.array(
            [0.0, 1.0],
            dtype=np.float32,
        ),
    }

    monkeypatch.setattr(
        main.models,
        "image_embedding",
        lambda reference: image_vectors[reference],
    )

    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-TEXT-TO-IMAGE",
                "visualText": (
                    "금색 로고가 있는 검은색 카드지갑"
                ),
            },
            "candidates": [
                {
                    "itemId": "FOUND-WALLET",
                    "imagePath": "wallet.jpg",
                },
                {
                    "itemId": "FOUND-UMBRELLA",
                    "imagePath": "umbrella.jpg",
                },
            ],
        },
    )

    assert response.status_code == 200

    result = response.json()
    first = result["topCandidates"][0]

    assert first["itemId"] == "FOUND-WALLET"
    assert first["scores"]["imageText"] == pytest.approx(
        1.0
    )
    assert first["imageTextDetails"][
        "searchTextToCandidateImage"
    ] == pytest.approx(1.0)
    assert first["imageTextDetails"][
        "searchImageToCandidateText"
    ] is None


def test_search_image_ranks_candidate_visual_text(
    monkeypatch,
):
    monkeypatch.setattr(
        main.models,
        "image_embedding",
        lambda reference: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    def fake_visual_text_embedding(text: str):
        if "지갑" in text:
            return np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )

        return np.array(
            [0.0, 1.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        main.models,
        "visual_text_embedding",
        fake_visual_text_embedding,
    )

    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-IMAGE-TO-TEXT",
                "imagePath": "query.jpg",
            },
            "candidates": [
                {
                    "itemId": "FOUND-WALLET",
                    "visualText": "검은색 카드지갑",
                },
                {
                    "itemId": "FOUND-UMBRELLA",
                    "visualText": "파란색 장우산",
                },
            ],
        },
    )

    assert response.status_code == 200

    result = response.json()
    first = result["topCandidates"][0]

    assert first["itemId"] == "FOUND-WALLET"
    assert first["scores"]["imageText"] == pytest.approx(
        1.0
    )
    assert first["imageTextDetails"][
        "searchTextToCandidateImage"
    ] is None
    assert first["imageTextDetails"][
        "searchImageToCandidateText"
    ] == pytest.approx(1.0)


def test_invalid_image_keeps_other_scores_available(
    monkeypatch,
):
    monkeypatch.setattr(
        main.models,
        "text_embedding",
        lambda text: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    def fail_image_embedding(reference: str):
        raise FileNotFoundError(reference)

    monkeypatch.setattr(
        main.models,
        "image_embedding",
        fail_image_embedding,
    )

    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-MISSING-IMAGE",
                "title": "검은색 지갑",
            },
            "candidates": [
                {
                    "itemId": "FOUND-MISSING-IMAGE",
                    "title": "검은색 지갑",
                    "imagePath": "missing.jpg",
                },
            ],
        },
    )

    assert response.status_code == 200

    candidate = response.json()[
        "topCandidates"
    ][0]

    assert candidate["scores"]["text"] == pytest.approx(
        1.0
    )
    assert candidate["scores"]["image"] is None
    assert candidate["scores"]["imageText"] is None


def test_visual_text_does_not_load_siglip_without_images(
    monkeypatch,
):
    monkeypatch.setattr(
        main.models,
        "text_embedding",
        lambda text: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    calls: list[str] = []

    def track_calls(text: str):
        calls.append(text)

        return np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        main.models,
        "visual_text_embedding",
        track_calls,
    )

    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-TEXT-ONLY",
                "description": "검은색 지갑을 잃어버림",
                "visualText": "검은색 카드지갑",
            },
            "candidates": [
                {
                    "itemId": "FOUND-TEXT-ONLY",
                    "description": "검은색 지갑",
                    "visualText": "검은색 카드지갑",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["topCandidates"][0][
        "scores"
    ]["imageText"] is None
    assert calls == []


def test_inactive_search_is_rejected():
    response = client.post(
        "/v1/rankings",
        json={
            "search": {
                "searchId": "SEARCH-INACTIVE",
                "status": "INACTIVE",
            },
            "candidates": [],
            "excludedItemIds": [],
            "topK": 5,
            "responseLimit": 10,
        },
    )

    assert response.status_code == 409
    assert (
        response.json()["detail"]
        == "ACTIVE 상태의 탐색 카드만 순위를 계산할 수 있습니다."
    )

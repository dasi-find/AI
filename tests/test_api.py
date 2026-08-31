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
        "attribute": 0.6,
        "date": 0.7,
        "location": 0.5,
    }

    weights = {
        "image": 0.35,
        "text": 0.30,
        "attribute": 0.20,
        "date": 0.10,
        "location": 0.05,
    }

    result = main.combine_scores(scores, weights)

    expected = (
        0.8 * 0.30
        + 0.6 * 0.20
        + 0.7 * 0.10
        + 0.5 * 0.05
    ) / 0.65

    assert result == pytest.approx(expected)


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
from __future__ import annotations

import io
import math
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor


# =========================================================
# 환경 설정
# =========================================================

class Settings(BaseSettings):
    text_model_name: str = "BAAI/bge-m3"
    image_model_name: str = "google/siglip2-base-patch16-224"
    device: str = "auto"
    dataset_dir: str | None = None

    image_weight: float = 0.35
    text_weight: float = 0.30
    attribute_weight: float = 0.20
    date_weight: float = 0.10
    location_weight: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


DEVICE = resolve_device(settings.device)


# =========================================================
# API 요청 데이터 계약
# =========================================================

class SearchItem(BaseModel):
    searchId: str
    status: str = "ACTIVE"

    title: str | None = None
    description: str | None = None
    category: str | None = None
    attributes: dict[str, Any] = Field(
        default_factory=dict
    )

    lostAt: datetime | None = None

    locationText: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    imagePath: str | None = None
    imageUrl: str | None = None


class FoundItem(BaseModel):
    itemId: str
    status: str = "ACTIVE"

    title: str | None = None
    description: str | None = None
    category: str | None = None
    attributes: dict[str, Any] = Field(
        default_factory=dict
    )

    foundAt: datetime | None = None

    locationText: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    imagePath: str | None = None
    imageUrl: str | None = None

    source: str | None = None
    sourceUrl: str | None = None


class RankingRequest(BaseModel):
    search: SearchItem
    candidates: list[FoundItem]

    excludedItemIds: list[str] = Field(
        default_factory=list
    )

    topK: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    responseLimit: int = Field(
        default=10,
        ge=5,
        le=100,
    )


# =========================================================
# 텍스트 전처리
# =========================================================

def value_to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
            str(item)
            for item in value
        )

    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {value_to_text(item)}"
            for key, item in value.items()
            if item is not None
        )

    return str(value)


def build_search_text(
    item: SearchItem | FoundItem,
) -> str:
    parts: list[str] = []

    if item.category:
        parts.append(
            f"분류: {item.category}"
        )

    if item.title:
        parts.append(
            f"물품명: {item.title}"
        )

    if item.description:
        parts.append(
            f"설명: {item.description}"
        )

    if item.attributes:
        attribute_text = value_to_text(
            item.attributes
        )

        if attribute_text:
            parts.append(
                f"특징: {attribute_text}"
            )

    if item.locationText:
        parts.append(
            f"장소: {item.locationText}"
        )

    return "\n".join(parts).strip()


# =========================================================
# BGE-M3 + SigLIP2 모델
# =========================================================

class ModelBundle:
    def __init__(self) -> None:
        self._text_model: SentenceTransformer | None = None
        self._image_model = None
        self._image_processor = None

        # 여러 요청이 동시에 들어와도
        # 모델을 한 번만 로딩하도록 보호합니다.
        self._text_model_lock = threading.Lock()
        self._image_model_lock = threading.Lock()

    @property
    def text_model(self) -> SentenceTransformer:
        if self._text_model is None:
            with self._text_model_lock:
                if self._text_model is None:
                    text_model = SentenceTransformer(
                        settings.text_model_name,
                        device=DEVICE,
                    )

                    self._text_model = text_model

        return self._text_model

    def load_image_model(self) -> None:
        if self._image_model is not None:
            return

        with self._image_model_lock:
            if self._image_model is not None:
                return

            image_processor = (
                AutoProcessor.from_pretrained(
                    settings.image_model_name
                )
            )

            image_model = AutoModel.from_pretrained(
                settings.image_model_name
            )

            image_model.to(DEVICE)
            image_model.eval()

            # 완전히 로딩된 다음 공유 변수에 저장합니다.
            self._image_processor = image_processor
            self._image_model = image_model

    @lru_cache(maxsize=50000)
    def text_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        vector = self.text_model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            vector,
            dtype=np.float32,
        )

    def open_image(
        self,
        reference: str,
    ) -> Image.Image:
        if (
            reference.startswith("http://")
            or reference.startswith("https://")
        ):
            response = httpx.get(
                reference,
                timeout=10.0,
                follow_redirects=True,
            )

            response.raise_for_status()

            if len(response.content) > 10 * 1024 * 1024:
                raise ValueError(
                    "이미지 파일이 10MB를 초과했습니다."
                )

            return Image.open(
                io.BytesIO(response.content)
            ).convert("RGB")

        path = (
            Path(reference)
            .expanduser()
            .resolve()
        )

        return Image.open(path).convert("RGB")

    @lru_cache(maxsize=10000)
    def image_embedding(
        self,
        reference: str,
    ) -> np.ndarray:
        self.load_image_model()

        image = self.open_image(reference)

        inputs = self._image_processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            features = (
                self._image_model.get_image_features(
                    **inputs
                )
            )

        if hasattr(features, "pooler_output"):
            features = features.pooler_output

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-12)

        return (
            features[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


models = ModelBundle()


# =========================================================
# 공통 점수 함수
# =========================================================

def cosine_score(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    denominator = float(
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator == 0:
        return 0.0

    cosine = float(
        np.dot(first, second)
        / denominator
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    # 코사인 유사도의 -1~1 범위를
    # 서비스 점수용 0~1 범위로 변경합니다.
    return (cosine + 1.0) / 2.0


def normalize_value(value: Any) -> str:
    return (
        value_to_text(value)
        .lower()
        .replace(" ", "")
        .strip()
    )


def attribute_score(
    query: SearchItem,
    candidate: FoundItem,
) -> float | None:
    scores: list[float] = []

    if query.category and candidate.category:
        query_category = normalize_value(
            query.category
        )

        candidate_category = normalize_value(
            candidate.category
        )

        scores.append(
            1.0
            if query_category == candidate_category
            else 0.0
        )

    for key, query_value in query.attributes.items():
        candidate_value = (
            candidate.attributes.get(key)
        )

        if (
            query_value is None
            or candidate_value is None
        ):
            continue

        query_text = normalize_value(
            query_value
        )

        candidate_text = normalize_value(
            candidate_value
        )

        if not query_text or not candidate_text:
            continue

        if (
            query_text == candidate_text
            or query_text in candidate_text
            or candidate_text in query_text
        ):
            scores.append(1.0)
        else:
            scores.append(0.0)

    if not scores:
        return None

    return sum(scores) / len(scores)


def date_score(
    lost_at: datetime | None,
    found_at: datetime | None,
) -> float | None:
    if lost_at is None or found_at is None:
        return None

    difference_days = abs(
        (found_at - lost_at).total_seconds()
    ) / 86400

    # 날짜 차이가 커질수록 점수가 완만하게 감소합니다.
    # 날짜는 후보 제거 조건으로 사용하지 않습니다.
    return math.exp(
        -difference_days / 14.0
    )


def haversine_distance_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    earth_radius = 6371.0

    first_latitude_radian = math.radians(
        first_latitude
    )

    second_latitude_radian = math.radians(
        second_latitude
    )

    latitude_difference = math.radians(
        second_latitude - first_latitude
    )

    longitude_difference = math.radians(
        second_longitude - first_longitude
    )

    value = (
        math.sin(
            latitude_difference / 2
        ) ** 2
        + math.cos(first_latitude_radian)
        * math.cos(second_latitude_radian)
        * math.sin(
            longitude_difference / 2
        ) ** 2
    )

    return (
        earth_radius
        * 2
        * math.atan2(
            math.sqrt(value),
            math.sqrt(1 - value),
        )
    )


def location_score(
    query: SearchItem,
    candidate: FoundItem,
) -> float | None:
    coordinates_available = all(
        value is not None
        for value in [
            query.latitude,
            query.longitude,
            candidate.latitude,
            candidate.longitude,
        ]
    )

    if coordinates_available:
        distance = haversine_distance_km(
            query.latitude,
            query.longitude,
            candidate.latitude,
            candidate.longitude,
        )

        # 거리가 멀어질수록 점수가 감소합니다.
        # 위치는 후보 제거 조건으로 사용하지 않습니다.
        return math.exp(
            -distance / 20.0
        )

    if (
        query.locationText
        and candidate.locationText
    ):
        query_location = normalize_value(
            query.locationText
        )

        candidate_location = normalize_value(
            candidate.locationText
        )

        if (
            query_location in candidate_location
            or candidate_location in query_location
        ):
            return 1.0

        return 0.0

    return None


def image_reference(
    item: SearchItem | FoundItem,
) -> str | None:
    return item.imagePath or item.imageUrl


def safe_image_embedding(
    reference: str | None,
) -> np.ndarray | None:
    if not reference:
        return None

    try:
        return models.image_embedding(reference)
    except Exception:
        # 이미지 파일 누락, 손상 또는 다운로드 실패 시
        # 전체 요청을 실패시키지 않고 이미지 없음으로 처리합니다.
        return None


def combine_scores(
    scores: dict[str, float | None],
    weights: dict[str, float],
) -> float:
    weighted_sum = 0.0
    available_weight = 0.0

    for name, score in scores.items():
        if score is None:
            continue

        weight = weights[name]

        weighted_sum += score * weight
        available_weight += weight

    if available_weight == 0:
        return 0.0

    # 이미지 등의 정보가 없으면
    # 사용 가능한 가중치만으로 재정규화합니다.
    return weighted_sum / available_weight


def build_reasons(
    scores: dict[str, float | None],
    query: SearchItem,
    candidate: FoundItem,
) -> list[str]:
    reasons: list[str] = []

    if (
        scores["image"] is not None
        and scores["image"] >= 0.75
    ):
        reasons.append(
            "사진의 형태와 시각적 특징이 유사합니다."
        )

    if (
        scores["text"] is not None
        and scores["text"] >= 0.70
    ):
        reasons.append(
            "물품 설명과 주요 표현이 유사합니다."
        )

    if (
        scores["attribute"] is not None
        and scores["attribute"] >= 0.70
    ):
        reasons.append(
            "분류, 색상 또는 주요 특징이 유사합니다."
        )

    if (
        scores["date"] is not None
        and scores["date"] >= 0.70
        and query.lostAt
        and candidate.foundAt
    ):
        days = abs(
            (
                candidate.foundAt
                - query.lostAt
            ).total_seconds()
        ) / 86400

        reasons.append(
            "분실 시점과 습득 시점이 "
            f"약 {round(days, 1)}일 차이입니다."
        )

    if (
        scores["location"] is not None
        and scores["location"] >= 0.70
    ):
        reasons.append(
            "분실 장소와 가까운 지역에서 접수되었습니다."
        )

    if not reasons:
        reasons.append(
            "여러 조건을 종합하여 비교한 후보입니다."
        )

    return reasons[:4]


# =========================================================
# 전체 순위 계산
# =========================================================

def rank_all_items(
    request: RankingRequest,
) -> dict[str, Any]:
    query = request.search

    query_text = build_search_text(query)

    query_text_vector = (
        models.text_embedding(query_text)
        if query_text
        else None
    )

    query_image_vector = safe_image_embedding(
        image_reference(query)
    )

    weights = {
        "image": settings.image_weight,
        "text": settings.text_weight,
        "attribute": settings.attribute_weight,
        "date": settings.date_weight,
        "location": settings.location_weight,
    }

    ranked_items: list[dict[str, Any]] = []

    for candidate in request.candidates:
        if candidate.status.strip().upper() != "ACTIVE":
            continue

        candidate_text = build_search_text(
            candidate
        )

        text_similarity: float | None = None

        if (
            query_text_vector is not None
            and candidate_text
        ):
            candidate_text_vector = (
                models.text_embedding(
                    candidate_text
                )
            )

            text_similarity = cosine_score(
                query_text_vector,
                candidate_text_vector,
            )

        image_similarity: float | None = None

        candidate_image_vector = (
            safe_image_embedding(
                image_reference(candidate)
            )
        )

        if (
            query_image_vector is not None
            and candidate_image_vector is not None
        ):
            image_similarity = cosine_score(
                query_image_vector,
                candidate_image_vector,
            )

        scores = {
            "image": image_similarity,
            "text": text_similarity,
            "attribute": attribute_score(
                query,
                candidate,
            ),
            "date": date_score(
                query.lostAt,
                candidate.foundAt,
            ),
            "location": location_score(
                query,
                candidate,
            ),
        }

        final_score = combine_scores(
            scores,
            weights,
        )

        ranked_items.append(
            {
                "itemId": candidate.itemId,
                "finalScore": round(
                    final_score,
                    6,
                ),
                "scores": {
                    name: (
                        round(score, 6)
                        if score is not None
                        else None
                    )
                    for name, score in scores.items()
                },
                "reasons": build_reasons(
                    scores,
                    query,
                    candidate,
                ),
                "source": candidate.source,
                "sourceUrl": candidate.sourceUrl,
                "notice": (
                    "AI가 계산한 유사 후보이며 "
                    "동일 물품 또는 소유권을 "
                    "의미하지 않습니다."
                ),
            }
        )

    # 모든 ACTIVE 물품을 계산한 후 정렬합니다.
    ranked_items.sort(
        key=lambda item: (
            -item["finalScore"],
            item["itemId"],
        )
    )

    # 제외 여부와 관계없이 전체 순위를 먼저 부여합니다.
    for index, item in enumerate(
        ranked_items,
        start=1,
    ):
        item["globalRank"] = index

    excluded = set(
        request.excludedItemIds
    )

    visible_items: list[
        dict[str, Any]
    ] = []

    for item in ranked_items:
        if item["itemId"] in excluded:
            continue

        visible_item = dict(item)

        visible_item["displayRank"] = (
            len(visible_items) + 1
        )

        visible_item["isDisplayTopK"] = (
            visible_item["displayRank"]
            <= request.topK
        )

        visible_items.append(
            visible_item
        )

        if (
            len(visible_items)
            >= request.responseLimit
        ):
            break

    excluded_count = sum(
        1
        for item in ranked_items
        if item["itemId"] in excluded
    )

    return {
        "searchId": request.search.searchId,
        "totalRankedCount": len(
            ranked_items
        ),
        "excludedCount": excluded_count,
        "topK": request.topK,
        "topCandidates": visible_items,
        "modelVersion": (
            f"siglip2:{settings.image_model_name}"
            f"|bge-m3:{settings.text_model_name}"
            "|fixed-cosine-v0.1"
        ),
    }


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="다시찾음 AI Matching Service",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE,
        "textModel": settings.text_model_name,
        "imageModel": settings.image_model_name,
    }


@app.post("/v1/rankings")
def create_ranking(
    request: RankingRequest,
) -> dict[str, Any]:
    if (
        request.search.status.strip().upper()
        != "ACTIVE"
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "ACTIVE 상태의 탐색 카드만 "
                "순위를 계산할 수 있습니다."
            ),
        )

    return rank_all_items(request)
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


def load_items(jsonl_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    with jsonl_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{line_number}번째 줄의 JSON 형식이 잘못됐습니다."
                ) from error

    return items


def build_image_path(
    dataset_dir: Path,
    item: dict[str, Any],
) -> str | None:
    if not item.get("has_image"):
        return None

    relative_path = item.get("image_file")

    if not relative_path:
        return None

    image_path = (
        dataset_dir / relative_path
    ).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"이미지 파일을 찾을 수 없습니다: {image_path}"
        )

    return str(image_path)


def build_found_at(
    found_date: str | None,
) -> str | None:
    if not found_date:
        return None

    return f"{found_date}T12:00:00+09:00"


def convert_candidate(
    dataset_dir: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "itemId": item["source_id"],
        "status": "ACTIVE",
        "title": item.get("item_name"),
        "description": (
            item.get("title")
            or item.get("text_for_embedding")
        ),
        "category": (
            item.get("category_l2")
            or item.get("category_l1")
        ),
        "attributes": {
            "color": item.get("color"),
            "categoryL1": item.get("category_l1"),
        },
        "foundAt": build_found_at(
            item.get("found_date")
        ),

        # storage_place는 보관기관이므로
        # 실제 습득 위치 점수로 사용하지 않습니다.
        "locationText": None,
        "latitude": None,
        "longitude": None,

        "imagePath": build_image_path(
            dataset_dir,
            item,
        ),
        "imageUrl": None,
        "source": item.get("source"),

        # 현재 데이터에는 상세 페이지 URL이 없고
        # 이미지 URL만 있으므로 임의로 만들지 않습니다.
        "sourceUrl": None,
    }


def main() -> None:
    dataset_dir_value = os.getenv("DATASET_DIR")

    if not dataset_dir_value:
        raise RuntimeError(
            ".env에 DATASET_DIR을 입력해야 합니다."
        )

    dataset_dir = Path(
        dataset_dir_value
    ).expanduser().resolve()

    jsonl_path = dataset_dir / "all_items_60.jsonl"

    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {jsonl_path}"
        )

    items = load_items(jsonl_path)

    if len(items) != 60:
        raise ValueError(
            f"60건이어야 하지만 {len(items)}건입니다."
        )

    candidates = [
        convert_candidate(dataset_dir, item)
        for item in items
    ]

    # 첫 번째 이미지 물품을 파이프라인 확인용 검색물로 사용
    query_item = next(
        item
        for item in items
        if item.get("has_image")
    )

    query_image_path = build_image_path(
        dataset_dir,
        query_item,
    )

    request_data = {
        "search": {
            "searchId": "SEARCH-60-SMOKE",
            "status": "ACTIVE",
            "title": query_item.get("item_name"),
            "description": query_item.get("title"),
            "category": (
                query_item.get("category_l2")
                or query_item.get("category_l1")
            ),
            "attributes": {
                "color": query_item.get("color"),
                "categoryL1": query_item.get(
                    "category_l1"
                ),
            },
            "lostAt": build_found_at(
                query_item.get("found_date")
            ),
            "locationText": None,
            "latitude": None,
            "longitude": None,
            "imagePath": query_image_path,
            "imageUrl": None,
        },
        "excludedItemIds": [],
        "topK": 5,
        "responseLimit": 10,
        "candidates": candidates,
    }

    output_path = (
        Path(__file__).resolve().parent
        / "sample_request_60.local.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            request_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    image_count = sum(
        candidate["imagePath"] is not None
        for candidate in candidates
    )

    print(f"생성 완료: {output_path}")
    print(f"전체 후보: {len(candidates)}")
    print(f"이미지 있음: {image_count}")
    print(f"이미지 없음: {len(candidates) - image_count}")
    print(f"테스트 검색물: {query_item['sample_id']}")


if __name__ == "__main__":
    main()
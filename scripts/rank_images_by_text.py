from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.models.siglip import (
    SigLIP2Encoder,
    cosine_similarity,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def normalized_cosine_score(
    raw_cosine: float,
) -> float:
    return (max(-1.0, min(1.0, raw_cosine)) + 1.0) / 2.0


def find_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"이미지 폴더를 찾을 수 없습니다: {image_dir}"
        )

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rank_images(
    visual_text: str,
    image_paths: list[Path],
    encoder: SigLIP2Encoder,
) -> tuple[list[dict], list[dict]]:
    text_embedding = encoder.encode_text(
        visual_text
    )

    ranked: list[dict] = []
    skipped: list[dict] = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image_embedding = encoder.encode_image(
                    image
                )
        except Exception as error:
            skipped.append(
                {
                    "filename": image_path.name,
                    "reason": str(error),
                }
            )
            continue

        raw_cosine = cosine_similarity(
            text_embedding,
            image_embedding,
        )

        ranked.append(
            {
                "filename": image_path.name,
                "imagePath": str(image_path.resolve()),
                "rawCosine": round(raw_cosine, 6),
                "normalizedScore": round(
                    normalized_cosine_score(raw_cosine),
                    6,
                ),
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["normalizedScore"],
            item["filename"],
        )
    )

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "전처리된 visualText 한 개로 폴더 안의 "
            "후보 이미지들을 순위화합니다."
        )
    )
    parser.add_argument(
        "--text",
        required=True,
        help="직접 작성한 전처리 visualText",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="비교할 후보 이미지 폴더",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="터미널에 표시할 상위 후보 수",
    )
    parser.add_argument(
        "--output",
        default="results/text_to_image_ranking.json",
        help="전체 순위를 저장할 JSON 경로",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "xpu"],
        default=None,
        help="기본값은 사용 가능한 장치를 자동 선택",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k는 1 이상이어야 합니다.")

    visual_text = args.text.strip()

    if not visual_text:
        parser.error("--text는 빈 문자열일 수 없습니다.")

    image_dir = Path(args.image_dir).expanduser().resolve()
    image_paths = find_images(image_dir)

    if not image_paths:
        raise RuntimeError(
            f"비교할 이미지가 없습니다: {image_dir}"
        )

    print("=" * 72)
    print("Text -> Image Candidate Ranking")
    print("=" * 72)
    print("visualText:", visual_text)
    print("후보 이미지:", len(image_paths))
    print("SigLIP2 모델을 불러오는 중...")

    encoder = SigLIP2Encoder(
        device=args.device
    )

    ranked, skipped = rank_images(
        visual_text=visual_text,
        image_paths=image_paths,
        encoder=encoder,
    )

    print("사용 장치:", encoder.device)
    print()
    print("===== Top Candidates =====")

    for item in ranked[: args.top_k]:
        print(
            f'{item["rank"]:2d}위 | '
            f'{item["normalizedScore"]:.6f} | '
            f'{item["filename"]}'
        )

    if skipped:
        print()
        print("건너뛴 이미지:", len(skipped))

        for item in skipped:
            print(
                f'- {item["filename"]}: '
                f'{item["reason"]}'
            )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "visualText": visual_text,
        "model": encoder.model_name,
        "device": encoder.device,
        "candidateCount": len(image_paths),
        "rankedCount": len(ranked),
        "skippedCount": len(skipped),
        "candidates": ranked,
        "skipped": skipped,
        "notice": (
            "normalizedScore는 동일 물품 확률이 아니라 "
            "후보 간 상대 순위를 위한 보조 점수입니다."
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("전체 결과 저장:", output_path.resolve())


if __name__ == "__main__":
    main()

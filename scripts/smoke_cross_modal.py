from __future__ import annotations

import argparse
import json

from src.models.siglip import SigLIP2Encoder
from src.scoring.cross_modal import CrossModalScorer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="이미지 경로",
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Rule/Template로 생성된 SigLIP용 visualText",
    )

    parser.add_argument(
        "--direction",
        choices=[
            "image_to_text",
            "text_to_image",
        ],
        default="image_to_text",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("SigLIP2 Cross-modal v1 Smoke Test")
    print("Image <-> visualText")
    print("=" * 72)

    print("Loading SigLIP2...")
    siglip = SigLIP2Encoder()

    scorer = CrossModalScorer(
        siglip=siglip,
    )

    if args.direction == "image_to_text":
        result = scorer.score_image_to_text(
            image_path=args.image,
            visual_text=args.text,
        )
    else:
        result = scorer.score_text_to_image(
            visual_text=args.text,
            image_path=args.image,
        )

    print()
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

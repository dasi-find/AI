from __future__ import annotations

import argparse
import json

from src.models.siglip import SigLIP2ImageEncoder
from src.scoring.image import ImageImageScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    print("=" * 72)
    print("Image↔Image v1 Smoke Test")
    print("Pipeline: Original Image -> SigLIP2 -> Cosine Similarity")
    print("=" * 72)

    print("Loading SigLIP2...")
    siglip = SigLIP2ImageEncoder()

    scorer = ImageImageScorer(siglip=siglip)

    print("Running Image↔Image comparison...")

    result = scorer.score_paths(
        query_image_path=args.query,
        candidate_image_path=args.candidate,
    )

    print()
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

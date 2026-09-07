from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch

from src.models.siglip import SigLIP2ImageEncoder, cosine_similarity


@dataclass
class PreparedImage:
    embedding: torch.Tensor
    source_path: str | None = None
    preprocessing: str = "original"


class ImageImageScorer:
    """
    Final v1 Image↔Image scorer.

    Original Image
    -> SigLIP2 Image Encoder
    -> normalized embedding
    -> cosine similarity
    -> imageImageScore

    Florence-2 is not part of this scoring path.
    """

    def __init__(self, siglip: SigLIP2ImageEncoder) -> None:
        self.siglip = siglip

    def prepare_image(self, image: Image.Image) -> PreparedImage:
        image = image.convert("RGB")
        embedding = self.siglip.encode_image(image)

        return PreparedImage(
            embedding=embedding,
            preprocessing="original",
        )

    def prepare_image_path(self, image_path: str | Path) -> PreparedImage:
        image_path = Path(image_path)

        with Image.open(image_path) as image:
            prepared = self.prepare_image(image)

        prepared.source_path = str(image_path)
        return prepared

    def score_prepared(
        self,
        query: PreparedImage,
        candidate: PreparedImage,
    ) -> dict:
        similarity = cosine_similarity(
            query.embedding,
            candidate.embedding,
        )

        return {
            "imageImageScore": similarity,
            "preprocessing": "original",
            "querySource": query.source_path,
            "candidateSource": candidate.source_path,
            "imageModel": self.siglip.model_name,
        }

    def score_paths(
        self,
        query_image_path: str | Path,
        candidate_image_path: str | Path,
    ) -> dict:
        query = self.prepare_image_path(query_image_path)
        candidate = self.prepare_image_path(candidate_image_path)

        return self.score_prepared(
            query=query,
            candidate=candidate,
        )

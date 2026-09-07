from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch

from src.models.siglip import SigLIP2Encoder, cosine_similarity


@dataclass
class PreparedSigLIPImage:
    embedding: torch.Tensor
    source_path: str | None = None


@dataclass
class PreparedSigLIPText:
    embedding: torch.Tensor
    visual_text: str


class CrossModalScorer:
    """
    SigLIP2 Cross-modal scorer.

    Text -> Image and Image -> Text are NOT separate models.
    They use the same SigLIP2 shared embedding space.

    Same (image, text) pair:
        cosine(image_embedding, text_embedding)

    The numerical pair score is the same regardless of which side is called
    the query. Retrieval direction only changes which candidate set is ranked.

    v1 role:
        auxiliary matching score
        NOT ownership probability
        NOT a standalone final decision
    """

    def __init__(
        self,
        siglip: SigLIP2Encoder,
    ) -> None:
        self.siglip = siglip

    def prepare_image(
        self,
        image: Image.Image,
    ) -> PreparedSigLIPImage:
        return PreparedSigLIPImage(
            embedding=self.siglip.encode_image(image),
        )

    def prepare_image_path(
        self,
        image_path: str | Path,
    ) -> PreparedSigLIPImage:
        image_path = Path(image_path)

        with Image.open(image_path) as image:
            prepared = self.prepare_image(image)

        prepared.source_path = str(image_path)
        return prepared

    def prepare_text(
        self,
        visual_text: str,
    ) -> PreparedSigLIPText:
        visual_text = str(visual_text).strip()

        return PreparedSigLIPText(
            embedding=self.siglip.encode_text(visual_text),
            visual_text=visual_text,
        )

    def score_prepared(
        self,
        image: PreparedSigLIPImage,
        text: PreparedSigLIPText,
        direction: str,
    ) -> dict:
        if direction not in {
            "image_to_text",
            "text_to_image",
        }:
            raise ValueError(
                "direction must be 'image_to_text' or 'text_to_image'"
            )

        similarity = cosine_similarity(
            image.embedding,
            text.embedding,
        )

        return {
            "imageTextScore": similarity,
            "direction": direction,
            "visualText": text.visual_text,
            "imageSource": image.source_path,
            "imageModel": self.siglip.model_name,
            "textModel": self.siglip.model_name,
            "role": "auxiliary_matching_score",
        }

    def score_image_to_text(
        self,
        image_path: str | Path,
        visual_text: str,
    ) -> dict:
        return self.score_prepared(
            image=self.prepare_image_path(image_path),
            text=self.prepare_text(visual_text),
            direction="image_to_text",
        )

    def score_text_to_image(
        self,
        visual_text: str,
        image_path: str | Path,
    ) -> dict:
        return self.score_prepared(
            image=self.prepare_image_path(image_path),
            text=self.prepare_text(visual_text),
            direction="text_to_image",
        )

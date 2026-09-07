from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"


def _extract_embedding(output: object) -> torch.Tensor:
    """
    Transformers-version compatibility.

    SigLIP2 get_image_features()/get_text_features() may return either:
    - torch.Tensor
    - BaseModelOutputWithPooling (use .pooler_output)
    """
    if torch.is_tensor(output):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    raise TypeError(
        f"Unexpected SigLIP output type: {type(output)}"
    )


class SigLIP2Encoder:
    """
    Shared SigLIP2 encoder for:
      - Image <-> Image
      - Image <-> Text
      - Text <-> Image

    One model instance should be reused across the AI service.
    """

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or self._resolve_device()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(
            images=[image.convert("RGB")],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        output = self.model.get_image_features(**inputs)
        embedding = _extract_embedding(output)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding[0].detach().cpu()

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        text = str(text).strip()

        if not text:
            raise ValueError("visualText must not be empty.")

        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        output = self.model.get_text_features(**inputs)
        embedding = _extract_embedding(output)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding[0].detach().cpu()


# Backward compatibility:
# existing Image↔Image scorer can continue importing this old class name.
SigLIP2ImageEncoder = SigLIP2Encoder


def cosine_similarity(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:
    a = F.normalize(
        embedding_a.float().view(1, -1),
        p=2,
        dim=-1,
    )

    b = F.normalize(
        embedding_b.float().view(1, -1),
        p=2,
        dim=-1,
    )

    return float(
        (a @ b.T).item()
    )

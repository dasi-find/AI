from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.rank_images_by_text import (
    find_images,
    normalized_cosine_score,
    rank_images,
)


class FakeEncoder:
    def encode_text(self, text: str) -> torch.Tensor:
        assert text

        return torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        )

    def encode_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        red, _, blue = image.convert("RGB").getpixel(
            (0, 0)
        )

        if red > blue:
            return torch.tensor(
                [1.0, 0.0],
                dtype=torch.float32,
            )

        return torch.tensor(
            [0.0, 1.0],
            dtype=torch.float32,
        )


def save_color_image(
    path: Path,
    color: tuple[int, int, int],
) -> None:
    Image.new(
        "RGB",
        (2, 2),
        color=color,
    ).save(path)


def test_normalized_cosine_score_is_bounded():
    assert normalized_cosine_score(-2.0) == 0.0
    assert normalized_cosine_score(0.0) == 0.5
    assert normalized_cosine_score(2.0) == 1.0


def test_rank_images_orders_best_candidate_first(
    tmp_path,
):
    wallet_path = tmp_path / "wallet.jpg"
    umbrella_path = tmp_path / "umbrella.jpg"

    save_color_image(
        wallet_path,
        (255, 0, 0),
    )
    save_color_image(
        umbrella_path,
        (0, 0, 255),
    )

    (tmp_path / "notes.txt").write_text(
        "이미지가 아닌 파일",
        encoding="utf-8",
    )

    image_paths = find_images(tmp_path)

    assert len(image_paths) == 2

    ranked, skipped = rank_images(
        visual_text="검은색 카드지갑",
        image_paths=image_paths,
        encoder=FakeEncoder(),
    )

    assert skipped == []
    assert ranked[0]["filename"] == "wallet.jpg"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["normalizedScore"] == pytest.approx(
        1.0
    )

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.models.siglip import SigLIP2Encoder, cosine_similarity


VISUAL_TEXTS = [
    {
        "id": "민준_01",
        "label": "민준",
        "text": "검정색 SAINT_LAURENT 카드지갑, 앞면 금색 로고",
    },
    {
        "id": "민준_02",
        "label": "민준",
        "text": "검정색 작은 가죽 카드홀더, 앞면 금색 브랜드 장식",
    },
    {
        "id": "민준_03",
        "label": "민준",
        "text": "검정색 카드지갑, 사선 퀼팅 무늬, 금색 로고",
    },
    {
        "id": "민준_04",
        "label": "민준",
        "text": "검정색 카드 케이스, 앞면 큰 금속 로고, 뒷면 카드 수납칸",
    },
    {
        "id": "민준_05",
        "label": "민준",
        "text": "검정색 가죽 카드지갑",
    },
    {
        "id": "병주_01",
        "label": "병주",
        "text": "검정색 가죽 반지갑, 모서리 작은 빨강 흰색 남색 로고",
    },
    {
        "id": "병주_02",
        "label": "병주",
        "text": "검정색 남성용 반지갑, 모서리 작은 직사각형 브랜드 마크",
    },
    {
        "id": "병주_03",
        "label": "병주",
        "text": "검정색 가죽 지갑, 매끄러운 표면, 작은 삼색 로고",
    },
    {
        "id": "병주_04",
        "label": "병주",
        "text": "검정색 반지갑, 사용감과 긁힌 흔적",
    },
    {
        "id": "병주_05",
        "label": "병주",
        "text": "검정색 가죽 지갑",
    },
]


def load_images(image_dir: str):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    image_data = []

    for path in paths:
        name = path.name

        if "민준" in name:
            label = "민준"
        elif "병주" in name:
            label = "병주"
        else:
            label = "UNKNOWN"

        image_data.append({
            "id": path.stem,
            "label": label,
            "path": str(path),
        })

    unknown = [x for x in image_data if x["label"] == "UNKNOWN"]
    if unknown:
        raise RuntimeError(
            "파일명에 '민준' 또는 '병주'가 없는 이미지가 있습니다:\n"
            + "\n".join(x["path"] for x in unknown)
        )

    return image_data


def build_similarity_matrix(siglip: SigLIP2Encoder, image_data):
    print()
    print("Encoding 10 visual texts...")
    text_embeddings = [
        siglip.encode_text(item["text"])
        for item in VISUAL_TEXTS
    ]

    print("Encoding images...")
    image_embeddings = []

    from PIL import Image

    for item in image_data:
        with Image.open(item["path"]) as image:
            image_embeddings.append(
                siglip.encode_image(image)
            )

    similarity_matrix = torch.empty(
        len(text_embeddings),
        len(image_embeddings),
        dtype=torch.float32,
    )

    for i, text_embedding in enumerate(text_embeddings):
        for j, image_embedding in enumerate(image_embeddings):
            similarity_matrix[i, j] = cosine_similarity(
                text_embedding,
                image_embedding,
            )

    return similarity_matrix


def evaluate(results: pd.DataFrame, positive_count: int):
    top1 = []
    recall3 = []
    recall5 = []
    reciprocal_ranks = []

    for query in results["query"].unique():
        qdf = results[results["query"] == query].sort_values("rank")

        top1.append(int(bool(qdf.iloc[0]["correct"])))
        recall3.append(
            float(qdf[qdf["rank"] <= 3]["correct"].sum()) / positive_count
        )
        recall5.append(
            float(qdf[qdf["rank"] <= 5]["correct"].sum()) / positive_count
        )

        first_positive = qdf[qdf["correct"]].iloc[0]
        reciprocal_ranks.append(
            1.0 / int(first_positive["rank"])
        )

    return {
        "Top-1 Accuracy": sum(top1) / len(top1),
        "Mean Recall@3": sum(recall3) / len(recall3),
        "Mean Recall@5": sum(recall5) / len(recall5),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def print_metrics(title: str, metrics: dict):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    for key, value in metrics.items():
        print(f"{key:24s}: {value:.4f}")


def print_hardest_negatives(df: pd.DataFrame, title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    for query in df["query"].unique():
        qdf = df[df["query"] == query]
        negatives = qdf[~qdf["correct"]].sort_values(
            "score",
            ascending=False,
        )

        if negatives.empty:
            continue

        row = negatives.iloc[0]

        print(
            f"{query:18s} -> "
            f"{row['candidate']:22s} | "
            f"score={row['score']:.4f} | "
            f"rank={int(row['rank'])}"
        )


def text_to_image(similarity_matrix, image_data):
    rows = []

    print()
    print("#" * 78)
    print("TEXT -> IMAGE")
    print("#" * 78)

    for i, query in enumerate(VISUAL_TEXTS):
        scores = similarity_matrix[i]
        ranking = torch.argsort(
            scores,
            descending=True,
        ).tolist()

        print()
        print("=" * 78)
        print(f"Query: {query['id']}")
        print(f"Text : {query['text']}")
        print("-" * 78)

        for rank, idx in enumerate(ranking, start=1):
            candidate = image_data[idx]
            score = float(scores[idx])
            correct = candidate["label"] == query["label"]

            print(
                f"{rank:2d}위 | "
                f"{candidate['id']:22s} | "
                f"{score:.4f} | "
                f"{'정답' if correct else '오답'}"
            )

            rows.append({
                "query": query["id"],
                "query_label": query["label"],
                "candidate": candidate["id"],
                "candidate_label": candidate["label"],
                "rank": rank,
                "score": score,
                "correct": correct,
            })

    return pd.DataFrame(rows)


def image_to_text(similarity_matrix, image_data):
    rows = []

    print()
    print("#" * 78)
    print("IMAGE -> TEXT")
    print("#" * 78)

    for image_idx, query in enumerate(image_data):
        scores = similarity_matrix[:, image_idx]
        ranking = torch.argsort(
            scores,
            descending=True,
        ).tolist()

        print()
        print("=" * 78)
        print(f"Query Image: {query['id']}")
        print("-" * 78)

        for rank, idx in enumerate(ranking, start=1):
            candidate = VISUAL_TEXTS[idx]
            score = float(scores[idx])
            correct = candidate["label"] == query["label"]

            print(
                f"{rank:2d}위 | "
                f"{candidate['id']:12s} | "
                f"{score:.4f} | "
                f"{'정답' if correct else '오답'}"
            )

            rows.append({
                "query": query["id"],
                "query_label": query["label"],
                "candidate": candidate["id"],
                "candidate_label": candidate["label"],
                "rank": rank,
                "score": score,
                "correct": correct,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        required=True,
        help="민준/병주 원본 이미지 10장이 있는 폴더",
    )

    parser.add_argument(
        "--output-dir",
        default="results/cross_modal_v1_ranking",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_data = load_images(args.image_dir)

    print("=" * 78)
    print("SigLIP2 Cross-modal v1 Multi-candidate Ranking")
    print("=" * 78)
    print("Images:", len(image_data))

    if len(image_data) != 10:
        print(
            "[WARNING] 현재 평가 스크립트는 민준 5장 + 병주 5장, "
            "총 10장을 기준으로 설계했습니다."
        )

    counts = pd.Series(
        [x["label"] for x in image_data]
    ).value_counts()

    print()
    print("Image labels:")
    print(counts.to_string())

    if set(counts.index) != {"민준", "병주"}:
        raise RuntimeError(
            "민준/병주 두 label이 모두 필요합니다."
        )

    if counts.nunique() != 1:
        raise RuntimeError(
            "현재 평가는 민준/병주 이미지 수가 같다는 전제입니다."
        )

    positive_count = int(counts.iloc[0])

    print()
    print("Loading SigLIP2 ONCE...")
    siglip = SigLIP2Encoder()

    similarity_matrix = build_similarity_matrix(
        siglip,
        image_data,
    )

    print()
    print(
        "Similarity matrix:",
        tuple(similarity_matrix.shape),
    )

    t2i_df = text_to_image(
        similarity_matrix,
        image_data,
    )

    i2t_df = image_to_text(
        similarity_matrix,
        image_data,
    )

    t2i_metrics = evaluate(
        t2i_df,
        positive_count=positive_count,
    )

    i2t_metrics = evaluate(
        i2t_df,
        positive_count=5,
    )

    print_metrics(
        "Text -> Image Metrics",
        t2i_metrics,
    )

    print_metrics(
        "Image -> Text Metrics",
        i2t_metrics,
    )

    print_hardest_negatives(
        t2i_df,
        "Text -> Image Hardest Negatives",
    )

    print_hardest_negatives(
        i2t_df,
        "Image -> Text Hardest Negatives",
    )

    t2i_df.to_csv(
        output_dir / "text_to_image_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    i2t_df.to_csv(
        output_dir / "image_to_text_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matrix_df = pd.DataFrame(
        similarity_matrix.numpy(),
        index=[x["id"] for x in VISUAL_TEXTS],
        columns=[x["id"] for x in image_data],
    )

    matrix_df.to_csv(
        output_dir / "similarity_matrix.csv",
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame([
        {
            "direction": "text_to_image",
            **t2i_metrics,
        },
        {
            "direction": "image_to_text",
            **i2t_metrics,
        },
    ])

    summary_df.to_csv(
        output_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("Saved")
    print("=" * 78)
    print(output_dir)


if __name__ == "__main__":
    main()

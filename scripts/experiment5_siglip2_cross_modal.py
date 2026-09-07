import os
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "google/siglip2-base-patch16-224"


TEXT_DATA = [
    {
        "id": "민준_01",
        "label": "민준",
        "text": "검정색 카드지갑이고 앞면에 큰 금색 YSL 로고가 있습니다.",
    },
    {
        "id": "민준_02",
        "label": "민준",
        "text": "검은색 작은 가죽 카드홀더이며 앞쪽에 금색 브랜드 장식이 있습니다.",
    },
    {
        "id": "민준_03",
        "label": "민준",
        "text": "블랙 카드지갑이고 표면에 사선 퀼팅 무늬와 금색 로고가 있습니다.",
    },
    {
        "id": "민준_04",
        "label": "민준",
        "text": "검정색 카드 케이스이며 앞면에 큰 금속 로고가 있고 뒷면에 카드 수납칸이 있습니다.",
    },
    {
        "id": "민준_05",
        "label": "민준",
        "text": "검은색 가죽 카드지갑입니다.",
    },
    {
        "id": "병주_01",
        "label": "병주",
        "text": "검정색 가죽 반지갑이고 한쪽 모서리에 작은 빨강 흰색 남색 로고가 있습니다.",
    },
    {
        "id": "병주_02",
        "label": "병주",
        "text": "검은색 남성용 반지갑이며 모서리에 작은 직사각형 브랜드 마크가 있습니다.",
    },
    {
        "id": "병주_03",
        "label": "병주",
        "text": "블랙 가죽 지갑이고 표면이 매끄러우며 작은 삼색 로고가 붙어 있습니다.",
    },
    {
        "id": "병주_04",
        "label": "병주",
        "text": "검정색 반지갑이며 표면에 사용감과 긁힌 흔적이 있습니다.",
    },
    {
        "id": "병주_05",
        "label": "병주",
        "text": "검은색 가죽 지갑입니다.",
    },
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_images(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    paths = sorted(
        [
            p
            for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
    )

    image_data = []

    for path in paths:
        filename = path.name

        if "민준" in filename:
            label = "민준"
        elif "병주" in filename:
            label = "병주"
        else:
            label = "UNKNOWN"

        image_data.append(
            {
                "id": path.stem,
                "label": label,
                "path": str(path),
            }
        )

    return image_data


def evaluate(results, num_positives):
    df = pd.DataFrame(results)

    metrics = {
        "top1": [],
        "recall3": [],
        "recall5": [],
        "rr": [],
        "hard_negative_rank": [],
    }

    for query in df["query"].unique():
        qdf = df[df["query"] == query].sort_values("rank")

        metrics["top1"].append(int(qdf.iloc[0]["correct"]))

        top3_correct = qdf[qdf["rank"] <= 3]["correct"].sum()
        top5_correct = qdf[qdf["rank"] <= 5]["correct"].sum()

        metrics["recall3"].append(top3_correct / num_positives)
        metrics["recall5"].append(top5_correct / num_positives)

        positives = qdf[qdf["correct"]]
        first_positive_rank = int(positives.iloc[0]["rank"])

        metrics["rr"].append(1 / first_positive_rank)

        negatives = qdf[~qdf["correct"]]
        hardest_negative_rank = int(
            negatives.sort_values("score", ascending=False).iloc[0]["rank"]
        )

        metrics["hard_negative_rank"].append(hardest_negative_rank)

    return {
        "Top-1 Accuracy": sum(metrics["top1"]) / len(metrics["top1"]),
        "Mean Recall@3": sum(metrics["recall3"]) / len(metrics["recall3"]),
        "Mean Recall@5": sum(metrics["recall5"]) / len(metrics["recall5"]),
        "MRR": sum(metrics["rr"]) / len(metrics["rr"]),
        "Mean Hardest Negative Rank": (
            sum(metrics["hard_negative_rank"])
            / len(metrics["hard_negative_rank"])
        ),
    }


def print_metrics(title, metrics):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key:28s}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        required=True,
        help="민준/병주 지갑 이미지 폴더",
    )

    parser.add_argument(
        "--output-dir",
        default="results/experiment5_cross_modal",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()

    print("=" * 70)
    print("Experiment 5 - SigLIP2 Cross-modal Retrieval")
    print("=" * 70)
    print("Model :", MODEL_NAME)
    print("Device:", device)
    print()

    # --------------------------------------------------
    # Image dataset
    # --------------------------------------------------

    image_data = load_images(args.image_dir)

    print("Images:", len(image_data))

    for item in image_data:
        print(
            f"- {item['id']} | "
            f"label={item['label']}"
        )

    if len(image_data) == 0:
        raise RuntimeError("이미지를 찾지 못했습니다.")

    unknown = [
        item
        for item in image_data
        if item["label"] == "UNKNOWN"
    ]

    if unknown:
        print()
        print("[ERROR] label을 판단하지 못한 이미지가 있습니다.")

        for item in unknown:
            print(item["path"])

        raise RuntimeError(
            "이미지 파일명에 '민준' 또는 '병주'가 들어가야 합니다."
        )

    label_counts = pd.Series(
        [x["label"] for x in image_data]
    ).value_counts()

    print()
    print("Label counts:")
    print(label_counts)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print("Loading SigLIP2...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    print("Model loaded.")

    # --------------------------------------------------
    # Text embeddings
    # --------------------------------------------------

    texts = [item["text"] for item in TEXT_DATA]

    text_inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.no_grad():
        text_outputs = model.get_text_features(
            **text_inputs
        )
    text_embeddings = text_outputs.pooler_output

    text_embeddings = F.normalize(
        text_embeddings,
        p=2,
        dim=-1,
    )

    print(
        "Text embeddings:",
        tuple(text_embeddings.shape),
    )

    # --------------------------------------------------
    # Image embeddings
    # --------------------------------------------------

    images = [
        Image.open(item["path"]).convert("RGB")
        for item in image_data
    ]

    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(device)
        for key, value in image_inputs.items()
    }

    with torch.no_grad():
        image_outputs = model.get_image_features(
            **image_inputs
        )
    image_embeddings = image_outputs.pooler_output
    
    image_embeddings = F.normalize(
        image_embeddings,
        p=2,
        dim=-1,
    )

    print(
        "Image embeddings:",
        tuple(image_embeddings.shape),
    )

    # --------------------------------------------------
    # Cross-modal similarity
    # --------------------------------------------------

    similarity_matrix = (
        text_embeddings @ image_embeddings.T
    )

    similarity_matrix = similarity_matrix.cpu()

    print(
        "Similarity matrix:",
        tuple(similarity_matrix.shape),
    )

    # ==================================================
    # Experiment A
    # Text -> Image
    # ==================================================

    text_to_image_results = []

    print()
    print("#" * 70)
    print("TEXT -> IMAGE")
    print("#" * 70)

    for i, query in enumerate(TEXT_DATA):
        scores = similarity_matrix[i]

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        print()
        print("=" * 70)
        print(f"Query: {query['id']}")
        print(f"Text : {query['text']}")
        print(f"Label: {query['label']}")
        print("-" * 70)

        for rank, idx in enumerate(
            ranking.tolist(),
            start=1,
        ):
            candidate = image_data[idx]
            score = float(scores[idx])

            correct = (
                candidate["label"]
                == query["label"]
            )

            print(
                f"{rank:2d}위 | "
                f"{candidate['id']:20s} | "
                f"{score:.4f} | "
                f"{'정답' if correct else '오답'}"
            )

            text_to_image_results.append(
                {
                    "query": query["id"],
                    "query_label": query["label"],
                    "candidate": candidate["id"],
                    "candidate_label": candidate["label"],
                    "rank": rank,
                    "score": score,
                    "correct": correct,
                }
            )

    # ==================================================
    # Experiment B
    # Image -> Text
    # ==================================================

    image_to_text_results = []

    print()
    print("#" * 70)
    print("IMAGE -> TEXT")
    print("#" * 70)

    for i, query in enumerate(image_data):
        scores = similarity_matrix[:, i]

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        print()
        print("=" * 70)
        print(f"Query Image: {query['id']}")
        print(f"Label      : {query['label']}")
        print("-" * 70)

        for rank, idx in enumerate(
            ranking.tolist(),
            start=1,
        ):
            candidate = TEXT_DATA[idx]
            score = float(scores[idx])

            correct = (
                candidate["label"]
                == query["label"]
            )

            print(
                f"{rank:2d}위 | "
                f"{candidate['id']:12s} | "
                f"{score:.4f} | "
                f"{'정답' if correct else '오답'}"
            )

            image_to_text_results.append(
                {
                    "query": query["id"],
                    "query_label": query["label"],
                    "candidate": candidate["id"],
                    "candidate_label": candidate["label"],
                    "rank": rank,
                    "score": score,
                    "correct": correct,
                }
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    image_label_counts = {
        label: int(count)
        for label, count in label_counts.items()
    }

    # 현재 민준/병주 모두 같은 개수라는 전제
    image_positive_count = min(
        image_label_counts.values()
    )

    text_label_counts = pd.Series(
        [x["label"] for x in TEXT_DATA]
    ).value_counts()

    text_positive_count = int(
        text_label_counts.min()
    )

    text_to_image_metrics = evaluate(
        text_to_image_results,
        num_positives=image_positive_count,
    )

    image_to_text_metrics = evaluate(
        image_to_text_results,
        num_positives=text_positive_count,
    )

    print_metrics(
        "Text -> Image Metrics",
        text_to_image_metrics,
    )

    print_metrics(
        "Image -> Text Metrics",
        image_to_text_metrics,
    )

    # --------------------------------------------------
    # Hard negatives
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Text -> Image Hardest Negatives")
    print("=" * 70)

    t2i_df = pd.DataFrame(
        text_to_image_results
    )

    for query in t2i_df["query"].unique():
        qdf = t2i_df[
            t2i_df["query"] == query
        ]

        negative = (
            qdf[~qdf["correct"]]
            .sort_values(
                "score",
                ascending=False,
            )
            .iloc[0]
        )

        print(
            f"{query:12s} | "
            f"{negative['candidate']:20s} | "
            f"score={negative['score']:.4f} | "
            f"rank={int(negative['rank'])}"
        )

    print()
    print("=" * 70)
    print("Image -> Text Hardest Negatives")
    print("=" * 70)

    i2t_df = pd.DataFrame(
        image_to_text_results
    )

    for query in i2t_df["query"].unique():
        qdf = i2t_df[
            i2t_df["query"] == query
        ]

        negative = (
            qdf[~qdf["correct"]]
            .sort_values(
                "score",
                ascending=False,
            )
            .iloc[0]
        )

        print(
            f"{query:20s} | "
            f"{negative['candidate']:12s} | "
            f"score={negative['score']:.4f} | "
            f"rank={int(negative['rank'])}"
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    t2i_path = os.path.join(
        args.output_dir,
        "text_to_image_results.csv",
    )

    i2t_path = os.path.join(
        args.output_dir,
        "image_to_text_results.csv",
    )

    matrix_path = os.path.join(
        args.output_dir,
        "similarity_matrix.csv",
    )

    t2i_df.to_csv(
        t2i_path,
        index=False,
        encoding="utf-8-sig",
    )

    i2t_df.to_csv(
        i2t_path,
        index=False,
        encoding="utf-8-sig",
    )

    matrix_df = pd.DataFrame(
        similarity_matrix.numpy(),
        index=[x["id"] for x in TEXT_DATA],
        columns=[x["id"] for x in image_data],
    )

    matrix_df.to_csv(
        matrix_path,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Saved")
    print("=" * 70)
    print(t2i_path)
    print(i2t_path)
    print(matrix_path)


if __name__ == "__main__":
    main()
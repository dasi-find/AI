import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "google/siglip2-base-patch16-224"

TEXT_FIELDS = [
    "item_name",
    "title",
    "text_for_embedding",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def extract_embedding(output):
    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if torch.is_tensor(output):
        return output

    raise TypeError(
        f"Unexpected model output type: {type(output)}"
    )


def resolve_image_path(dataset_dir, image_file):
    if not image_file:
        return None

    dataset_dir = Path(dataset_dir)

    image_path = Path(image_file)

    # 절대경로일 경우
    if image_path.is_absolute() and image_path.exists():
        return image_path

    # dataset_dir / image_file
    candidate = dataset_dir / image_file

    if candidate.exists():
        return candidate

    # dataset_dir / images / filename
    candidate = dataset_dir / "images" / image_path.name

    if candidate.exists():
        return candidate

    return None


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)

    jsonl_path = dataset_dir / "all_items_60.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {jsonl_path}"
        )

    items = []

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if not item.get("has_image"):
                continue

            image_path = resolve_image_path(
                dataset_dir,
                item.get("image_file"),
            )

            if image_path is None:
                continue

            item["_image_path"] = str(image_path)

            items.append(item)

    return items


def get_text(item, field):
    value = item.get(field)

    if value is None:
        return ""

    return str(value).strip()


def encode_images(
    items,
    processor,
    model,
    device,
):
    images = []

    for item in items:
        image = Image.open(
            item["_image_path"]
        ).convert("RGB")

        images.append(image)

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_image_features(
            **inputs
        )

    embeddings = extract_embedding(outputs)

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    return embeddings


def encode_texts(
    texts,
    processor,
    model,
    device,
):
    inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_text_features(
            **inputs
        )

    embeddings = extract_embedding(outputs)

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    return embeddings


def evaluate_direction(
    similarity_matrix,
    item_ids,
    direction,
):
    results = []

    n = len(item_ids)

    if direction == "text_to_image":

        for query_idx in range(n):
            scores = similarity_matrix[query_idx]

            ranking = torch.argsort(
                scores,
                descending=True,
            ).tolist()

            for rank, candidate_idx in enumerate(
                ranking,
                start=1,
            ):
                correct = (
                    candidate_idx == query_idx
                )

                results.append(
                    {
                        "query_id": item_ids[query_idx],
                        "candidate_id": item_ids[candidate_idx],
                        "rank": rank,
                        "score": float(
                            scores[candidate_idx]
                        ),
                        "correct": correct,
                    }
                )

    elif direction == "image_to_text":

        for query_idx in range(n):
            scores = similarity_matrix[:, query_idx]

            ranking = torch.argsort(
                scores,
                descending=True,
            ).tolist()

            for rank, candidate_idx in enumerate(
                ranking,
                start=1,
            ):
                correct = (
                    candidate_idx == query_idx
                )

                results.append(
                    {
                        "query_id": item_ids[query_idx],
                        "candidate_id": item_ids[candidate_idx],
                        "rank": rank,
                        "score": float(
                            scores[candidate_idx]
                        ),
                        "correct": correct,
                    }
                )

    return pd.DataFrame(results)


def calculate_metrics(df):
    top1 = []
    hit5 = []
    reciprocal_ranks = []

    correct_ranks = []

    for query_id in df["query_id"].unique():

        qdf = (
            df[df["query_id"] == query_id]
            .sort_values("rank")
        )

        correct_row = qdf[
            qdf["correct"]
        ].iloc[0]

        rank = int(
            correct_row["rank"]
        )

        correct_ranks.append(rank)

        top1.append(
            int(rank == 1)
        )

        hit5.append(
            int(rank <= 5)
        )

        reciprocal_ranks.append(
            1 / rank
        )

    return {
        "Top-1 Accuracy": sum(top1) / len(top1),
        "Hit@5": sum(hit5) / len(hit5),
        "MRR": (
            sum(reciprocal_ranks)
            / len(reciprocal_ranks)
        ),
        "Mean Correct Rank": (
            sum(correct_ranks)
            / len(correct_ranks)
        ),
        "Median Correct Rank": float(
            pd.Series(correct_ranks).median()
        ),
    }


def print_metrics(
    title,
    metrics,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for key, value in metrics.items():
        print(
            f"{key:24s}: {value:.4f}"
        )


def print_failures(
    df,
    items_by_id,
    top_n=10,
):
    failures = []

    for query_id in df["query_id"].unique():

        qdf = (
            df[df["query_id"] == query_id]
            .sort_values("rank")
        )

        correct_row = qdf[
            qdf["correct"]
        ].iloc[0]

        rank = int(
            correct_row["rank"]
        )

        if rank > 1:
            failures.append(
                (
                    query_id,
                    rank,
                )
            )

    failures.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("Worst retrieval cases")
    print("-" * 70)

    for query_id, rank in failures[:top_n]:

        item = items_by_id[
            query_id
        ]

        print(
            f"Correct rank={rank:2d} | "
            f"{query_id}"
        )

        print(
            "  item_name:",
            item.get("item_name"),
        )

        print(
            "  title:",
            item.get("title"),
        )

        print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="results/experiment6_police_cross_modal",
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 70)
    print(
        "Experiment 6 - "
        "Real Police Text/Image Cross-modal Retrieval"
    )
    print("=" * 70)

    print("Model :", MODEL_NAME)
    print("Device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    items = load_dataset(
        args.dataset_dir
    )

    print()
    print(
        "Valid image items:",
        len(items),
    )

    if len(items) == 0:
        raise RuntimeError(
            "사용 가능한 이미지 데이터가 없습니다."
        )

    for item in items:
        print(
            "-",
            item.get("sample_id"),
            "|",
            item.get("item_name"),
        )

    # sample_id를 실험용 ID로 사용
    item_ids = [
        item["sample_id"]
        for item in items
    ]

    items_by_id = {
        item["sample_id"]: item
        for item in items
    }

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print("Loading SigLIP2...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    print("Model loaded.")

    # --------------------------------------------------
    # Image Embedding
    # 한 번만 생성
    # --------------------------------------------------

    print()
    print("Encoding images...")

    image_embeddings = encode_images(
        items,
        processor,
        model,
        device,
    )

    print(
        "Image embeddings:",
        tuple(image_embeddings.shape),
    )

    summary_rows = []

    # ==================================================
    # Text field 별 비교
    # ==================================================

    for field in TEXT_FIELDS:

        print()
        print("#" * 70)
        print(
            f"TEXT FIELD: {field}"
        )
        print("#" * 70)

        texts = [
            get_text(
                item,
                field,
            )
            for item in items
        ]

        empty_count = sum(
            1
            for text in texts
            if not text
        )

        print(
            "Empty texts:",
            empty_count,
        )

        # 빈 값은 SigLIP에 넣어도 의미가 없으므로
        # 최소한의 placeholder 사용
        processed_texts = [
            text if text else " "
            for text in texts
        ]

        text_embeddings = encode_texts(
            processed_texts,
            processor,
            model,
            device,
        )

        similarity_matrix = (
            text_embeddings
            @ image_embeddings.T
        ).cpu()

        # ----------------------------------------------
        # Text -> Image
        # ----------------------------------------------

        t2i_df = evaluate_direction(
            similarity_matrix,
            item_ids,
            "text_to_image",
        )

        t2i_metrics = calculate_metrics(
            t2i_df
        )

        print_metrics(
            f"{field} | Text -> Image",
            t2i_metrics,
        )

        print_failures(
            t2i_df,
            items_by_id,
        )

        # ----------------------------------------------
        # Image -> Text
        # ----------------------------------------------

        i2t_df = evaluate_direction(
            similarity_matrix,
            item_ids,
            "image_to_text",
        )

        i2t_metrics = calculate_metrics(
            i2t_df
        )

        print_metrics(
            f"{field} | Image -> Text",
            i2t_metrics,
        )

        print_failures(
            i2t_df,
            items_by_id,
        )

        # ----------------------------------------------
        # Save
        # ----------------------------------------------

        t2i_df.to_csv(
            output_dir
            / f"{field}_text_to_image.csv",
            index=False,
            encoding="utf-8-sig",
        )

        i2t_df.to_csv(
            output_dir
            / f"{field}_image_to_text.csv",
            index=False,
            encoding="utf-8-sig",
        )

        matrix_df = pd.DataFrame(
            similarity_matrix.numpy(),
            index=item_ids,
            columns=item_ids,
        )

        matrix_df.to_csv(
            output_dir
            / f"{field}_similarity_matrix.csv",
            encoding="utf-8-sig",
        )

        summary_rows.append(
            {
                "text_field": field,
                "direction": "text_to_image",
                **t2i_metrics,
            }
        )

        summary_rows.append(
            {
                "text_field": field,
                "direction": "image_to_text",
                **i2t_metrics,
            }
        )

    # ==================================================
    # Summary
    # ==================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    summary_df.to_csv(
        output_dir
        / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Saved:",
        output_dir,
    )


if __name__ == "__main__":
    main()
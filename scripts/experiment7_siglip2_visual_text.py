import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "google/siglip2-base-patch16-224"


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

    raise TypeError(f"Unexpected model output type: {type(output)}")


def resolve_image_path(dataset_dir, image_file):
    if not image_file:
        return None

    dataset_dir = Path(dataset_dir)
    image_path = Path(image_file)

    if image_path.is_absolute() and image_path.exists():
        return image_path

    candidate = dataset_dir / image_path
    if candidate.exists():
        return candidate

    candidate = dataset_dir / "images" / image_path.name
    if candidate.exists():
        return candidate

    return None


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    jsonl_path = dataset_dir / "all_items_60.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    items = []

    with jsonl_path.open("r", encoding="utf-8") as f:
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


def load_llm_visual_text(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = {"sample_id", "llm_visual_text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"LLM CSV missing columns: {sorted(missing)}"
        )

    return {
        str(row["sample_id"]): str(row["llm_visual_text"]).strip()
        for _, row in df.iterrows()
    }


def build_rule_visual_text(item):
    """
    LLM 없이 구조화 필드만 단순 연결하는 Control.
    원본 정보 외의 특징은 추가하지 않는다.
    """
    parts = []

    color = str(item.get("color") or "").strip()
    category_l2 = str(item.get("category_l2") or "").strip()
    item_name = str(item.get("item_name") or "").strip()

    if color:
        parts.append(color)

    if category_l2:
        parts.append(category_l2)

    if item_name:
        parts.append(item_name)

    return " ".join(parts).strip()


def encode_images(items, processor, model, device):
    images = [
        Image.open(item["_image_path"]).convert("RGB")
        for item in items
    ]

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    embeddings = extract_embedding(outputs)

    return F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )


def encode_texts(texts, processor, model, device):
    processed = [
        text if str(text).strip() else " "
        for text in texts
    ]

    inputs = processor(
        text=processed,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    embeddings = extract_embedding(outputs)

    return F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )


def build_global_results(similarity_matrix, items, direction):
    rows = []
    n = len(items)

    if direction == "text_to_image":
        for query_idx in range(n):
            scores = similarity_matrix[query_idx]
            ranking = torch.argsort(
                scores,
                descending=True,
            ).tolist()

            for rank, candidate_idx in enumerate(ranking, start=1):
                rows.append({
                    "query_id": items[query_idx]["sample_id"],
                    "query_category": items[query_idx]["category_l1"],
                    "candidate_id": items[candidate_idx]["sample_id"],
                    "candidate_category": items[candidate_idx]["category_l1"],
                    "rank": rank,
                    "score": float(scores[candidate_idx]),
                    "correct": candidate_idx == query_idx,
                })

    elif direction == "image_to_text":
        for query_idx in range(n):
            scores = similarity_matrix[:, query_idx]
            ranking = torch.argsort(
                scores,
                descending=True,
            ).tolist()

            for rank, candidate_idx in enumerate(ranking, start=1):
                rows.append({
                    "query_id": items[query_idx]["sample_id"],
                    "query_category": items[query_idx]["category_l1"],
                    "candidate_id": items[candidate_idx]["sample_id"],
                    "candidate_category": items[candidate_idx]["category_l1"],
                    "rank": rank,
                    "score": float(scores[candidate_idx]),
                    "correct": candidate_idx == query_idx,
                })

    else:
        raise ValueError(direction)

    return pd.DataFrame(rows)


def build_category_filtered_results(similarity_matrix, items, direction):
    """
    같은 category_l1 후보 안에서만 검색.
    현재 30개 샘플의 category_l1별 크기가 2~5개라 Hit@5는 의미가 없으므로
    Top-1 / Hit@2 / MRR / Mean Rank를 본다.
    """
    rows = []
    n = len(items)

    for query_idx in range(n):
        query_category = items[query_idx]["category_l1"]

        candidate_indices = [
            idx
            for idx in range(n)
            if items[idx]["category_l1"] == query_category
        ]

        if direction == "text_to_image":
            raw_scores = similarity_matrix[query_idx]
        elif direction == "image_to_text":
            raw_scores = similarity_matrix[:, query_idx]
        else:
            raise ValueError(direction)

        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: float(raw_scores[idx]),
            reverse=True,
        )

        for rank, candidate_idx in enumerate(ranked_indices, start=1):
            rows.append({
                "query_id": items[query_idx]["sample_id"],
                "query_category": query_category,
                "candidate_id": items[candidate_idx]["sample_id"],
                "candidate_category": items[candidate_idx]["category_l1"],
                "candidate_pool_size": len(candidate_indices),
                "rank": rank,
                "score": float(raw_scores[candidate_idx]),
                "correct": candidate_idx == query_idx,
            })

    return pd.DataFrame(rows)


def calculate_global_metrics(df):
    top1 = []
    hit5 = []
    reciprocal_ranks = []
    ranks = []

    for query_id in df["query_id"].unique():
        qdf = df[df["query_id"] == query_id].sort_values("rank")
        correct = qdf[qdf["correct"]].iloc[0]
        rank = int(correct["rank"])

        ranks.append(rank)
        top1.append(int(rank == 1))
        hit5.append(int(rank <= 5))
        reciprocal_ranks.append(1 / rank)

    return {
        "Top-1 Accuracy": sum(top1) / len(top1),
        "Hit@5": sum(hit5) / len(hit5),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "Mean Correct Rank": sum(ranks) / len(ranks),
        "Median Correct Rank": float(pd.Series(ranks).median()),
    }


def calculate_category_metrics(df):
    top1 = []
    hit2 = []
    reciprocal_ranks = []
    ranks = []

    for query_id in df["query_id"].unique():
        qdf = df[df["query_id"] == query_id].sort_values("rank")
        correct = qdf[qdf["correct"]].iloc[0]
        rank = int(correct["rank"])

        ranks.append(rank)
        top1.append(int(rank == 1))
        hit2.append(int(rank <= 2))
        reciprocal_ranks.append(1 / rank)

    return {
        "Top-1 Accuracy": sum(top1) / len(top1),
        "Hit@2": sum(hit2) / len(hit2),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "Mean Correct Rank": sum(ranks) / len(ranks),
        "Median Correct Rank": float(pd.Series(ranks).median()),
    }


def print_metrics(title, metrics):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    for key, value in metrics.items():
        print(f"{key:24s}: {value:.4f}")


def print_worst_cases(df, items_by_id, top_n=8):
    failures = []

    for query_id in df["query_id"].unique():
        qdf = df[df["query_id"] == query_id].sort_values("rank")
        correct = qdf[qdf["correct"]].iloc[0]
        rank = int(correct["rank"])

        if rank > 1:
            failures.append((query_id, rank))

    failures.sort(key=lambda x: x[1], reverse=True)

    print()
    print("Worst retrieval cases")
    print("-" * 78)

    if not failures:
        print("No failures: every correct pair is rank 1.")
        return

    for query_id, rank in failures[:top_n]:
        item = items_by_id[query_id]

        print(
            f"Correct rank={rank:2d} | "
            f"{query_id} | "
            f"category={item.get('category_l1')} | "
            f"item={item.get('item_name')}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="all_items_60.jsonl과 images 폴더가 있는 데이터셋 경로",
    )

    parser.add_argument(
        "--llm-text-csv",
        required=True,
        help="sample_id,llm_visual_text 컬럼이 있는 CSV",
    )

    parser.add_argument(
        "--output-dir",
        default="results/experiment7_visual_text",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 78)
    print("Experiment 7 - Raw vs Rule vs LLM Visual Text")
    print("=" * 78)
    print("Model :", MODEL_NAME)
    print("Device:", device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    items = load_dataset(args.dataset_dir)

    print()
    print("Valid image items:", len(items))

    if len(items) == 0:
        raise RuntimeError("사용 가능한 이미지 데이터가 없습니다.")

    items_by_id = {
        item["sample_id"]: item
        for item in items
    }

    category_counts = pd.Series(
        [item["category_l1"] for item in items]
    ).value_counts()

    print()
    print("Category counts:")
    print(category_counts.to_string())

    # ------------------------------------------------------------------
    # LLM visual text
    # ------------------------------------------------------------------

    llm_text_map = load_llm_visual_text(args.llm_text_csv)

    missing_llm = [
        item["sample_id"]
        for item in items
        if item["sample_id"] not in llm_text_map
    ]

    if missing_llm:
        raise RuntimeError(
            "LLM visual text가 없는 sample_id: "
            + ", ".join(missing_llm)
        )

    text_variants = {
        "item_name": [
            str(item.get("item_name") or "").strip()
            for item in items
        ],
        "rule_visual_text": [
            build_rule_visual_text(item)
            for item in items
        ],
        "llm_visual_text": [
            llm_text_map[item["sample_id"]]
            for item in items
        ],
    }

    print()
    print("Text examples:")
    for idx in range(min(5, len(items))):
        print("-" * 78)
        print("ID   :", items[idx]["sample_id"])
        print("RAW  :", text_variants["item_name"][idx])
        print("RULE :", text_variants["rule_visual_text"][idx])
        print("LLM  :", text_variants["llm_visual_text"][idx])

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print()
    print("Loading SigLIP2...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    print("Model loaded.")

    # ------------------------------------------------------------------
    # Image embeddings: exactly once
    # ------------------------------------------------------------------

    print()
    print("Encoding images...")

    image_embeddings = encode_images(
        items,
        processor,
        model,
        device,
    )

    print("Image embeddings:", tuple(image_embeddings.shape))

    summary_rows = []

    # ------------------------------------------------------------------
    # Compare text variants
    # ------------------------------------------------------------------

    for variant_name, texts in text_variants.items():
        print()
        print("#" * 78)
        print(f"TEXT VARIANT: {variant_name}")
        print("#" * 78)

        text_embeddings = encode_texts(
            texts,
            processor,
            model,
            device,
        )

        similarity_matrix = (
            text_embeddings @ image_embeddings.T
        ).cpu()

        matrix_df = pd.DataFrame(
            similarity_matrix.numpy(),
            index=[item["sample_id"] for item in items],
            columns=[item["sample_id"] for item in items],
        )

        matrix_df.to_csv(
            output_dir / f"{variant_name}_similarity_matrix.csv",
            encoding="utf-8-sig",
        )

        for direction in [
            "text_to_image",
            "image_to_text",
        ]:
            # ==========================================================
            # Global retrieval
            # ==========================================================
            global_df = build_global_results(
                similarity_matrix,
                items,
                direction,
            )

            global_metrics = calculate_global_metrics(global_df)

            print_metrics(
                f"{variant_name} | {direction} | GLOBAL",
                global_metrics,
            )

            print_worst_cases(
                global_df,
                items_by_id,
            )

            global_df.to_csv(
                output_dir
                / f"{variant_name}_{direction}_global.csv",
                index=False,
                encoding="utf-8-sig",
            )

            summary_rows.append({
                "text_variant": variant_name,
                "direction": direction,
                "evaluation": "global",
                **global_metrics,
            })

            # ==========================================================
            # Same-category retrieval
            # ==========================================================
            category_df = build_category_filtered_results(
                similarity_matrix,
                items,
                direction,
            )

            category_metrics = calculate_category_metrics(category_df)

            print_metrics(
                f"{variant_name} | {direction} | SAME CATEGORY",
                category_metrics,
            )

            print_worst_cases(
                category_df,
                items_by_id,
            )

            category_df.to_csv(
                output_dir
                / f"{variant_name}_{direction}_same_category.csv",
                index=False,
                encoding="utf-8-sig",
            )

            summary_rows.append({
                "text_variant": variant_name,
                "direction": direction,
                "evaluation": "same_category",
                **category_metrics,
            })

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(
        output_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Text variant dump
    text_dump = pd.DataFrame({
        "sample_id": [item["sample_id"] for item in items],
        "category_l1": [item["category_l1"] for item in items],
        "item_name": text_variants["item_name"],
        "rule_visual_text": text_variants["rule_visual_text"],
        "llm_visual_text": text_variants["llm_visual_text"],
    })

    text_dump.to_csv(
        output_dir / "text_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Saved:", output_dir)
    print()
    print("Decision guide:")
    print(
        "- LLM을 채택하려면 Raw/Rule 대비 Global Hit@5가 의미 있게 개선되고,\n"
        "  Same-category Top-1/MRR도 함께 좋아지는지 확인하세요."
    )
    print(
        "- Rule과 LLM 성능이 거의 같으면 더 단순한 Rule 방식을 우선합니다."
    )
    print(
        "- LLM 후에도 Cross-modal이 불안정하면 SigLIP Text↔Image는 보조 점수로 둡니다."
    )


if __name__ == "__main__":
    main()
"""Offline BM25 + dense-vector baseline for experiments on a bounded corpus.

For the complete 1.75M-document corpus, use pipeline.py with a local Typesense
server. rank-bm25 scores every document for every query and is intentionally
kept as a transparent quality baseline, not the production serving path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from text_utils import clean_scalar, load_json_map, normalize_text


ROOT = Path(__file__).resolve().parent


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size <= k:
        return np.argsort(-scores)
    selected = np.argpartition(scores, -k)[-k:]
    return selected[np.argsort(-scores[selected])]


def load_documents(path: Path, max_documents: int | None) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
            if max_documents is not None and len(documents) >= max_documents:
                break
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents


def load_models(model_name: str):
    try:
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
    return BM25Okapi, SentenceTransformer(model_name)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    abbreviations = load_json_map(ROOT / "abbreviations.json")
    documents = load_documents(args.jsonl, args.max_documents)
    BM25Okapi, encoder = load_models(config["embedding"]["model"])

    corpus_tokens = [str(document["allTextNormalized"]).split() for document in documents]
    bm25 = BM25Okapi(corpus_tokens)
    if all("embedding" in document for document in documents):
        vectors = np.asarray([document["embedding"] for document in documents], dtype=np.float32)
    else:
        texts = [
            " ".join((
                str(document.get("brandNameNormalized", "")),
                str(document.get("categoryNameNormalized", "")),
                str(document.get("productNameNormalized", "")),
                str(document.get("productSpecificationNormalized", "")),
            )).strip()
            for document in documents
        ]
        vectors = np.asarray(encoder.encode(
            texts,
            batch_size=int(config["embedding"]["batch_size"]),
            normalize_embeddings=True,
            show_progress_bar=True,
        ), dtype=np.float32)

    evaluation = pd.read_csv(args.evaluation_file, dtype=str, keep_default_na=False)
    if not {"query_text", "materialId"}.issubset(evaluation.columns):
        raise ValueError("Evaluation CSV must contain query_text and materialId")
    evaluation["query_text"] = evaluation["query_text"].map(clean_scalar)
    evaluation["materialId"] = evaluation["materialId"].map(clean_scalar)
    grouped = evaluation.groupby("query_text", sort=False)["materialId"].agg(lambda values: sorted(set(values)))
    if args.max_queries is not None:
        grouped = grouped.iloc[: args.max_queries]

    material_ids = np.asarray([str(document["materialId"]) for document in documents], dtype=object)
    corpus_ids = set(material_ids.tolist())
    lexical_weight = 1.0 - float(config["search"]["vector_alpha"])
    vector_weight = float(config["search"]["vector_alpha"])
    candidate_k = int(config["search"]["candidate_k"])
    rows: list[dict[str, Any]] = []

    for number, (query, expected_list) in enumerate(grouped.items(), start=1):
        normalized = normalize_text(query, abbreviations=abbreviations, remove_commercial_noise=True)
        lexical_scores = np.asarray(bm25.get_scores(normalized.split()), dtype=np.float32)
        query_vector = np.asarray(encoder.encode([normalized], normalize_embeddings=True)[0], dtype=np.float32)
        semantic_scores = vectors @ query_vector
        lexical_top = top_indices(lexical_scores, candidate_k)
        vector_top = top_indices(semantic_scores, candidate_k)
        lexical_rank = {int(index): rank for rank, index in enumerate(lexical_top, start=1)}
        vector_rank = {int(index): rank for rank, index in enumerate(vector_top, start=1)}
        candidate_indices = set(lexical_rank) | set(vector_rank)
        fused = []
        for index in candidate_indices:
            score = (
                lexical_weight / (60 + lexical_rank.get(index, candidate_k + 1))
                + vector_weight / (60 + vector_rank.get(index, candidate_k + 1))
            )
            fused.append((score, index))
        fused.sort(reverse=True)
        predicted = [str(material_ids[index]) for _, index in fused[:10]]
        expected = set(expected_list)
        first_rank = next((rank for rank, value in enumerate(predicted, start=1) if value in expected), 0)
        rows.append({
            "query_text": query,
            "expected_material_ids": ";".join(expected_list),
            "corpus_reachable": bool(expected & corpus_ids),
            "top1_correct": int(bool(predicted) and predicted[0] in expected),
            "recall_at_5": len(expected & set(predicted[:5])) / len(expected),
            "recall_at_10": len(expected & set(predicted[:10])) / len(expected),
            "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
            **{f"predicted_material_id_{rank}": predicted[rank - 1] if len(predicted) >= rank else "" for rank in range(1, 11)},
        })
        if number % 100 == 0 or number == len(grouped):
            print(f"evaluated={number:,}/{len(grouped):,}", file=sys.stderr)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    reachable = output[output["corpus_reachable"]]

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"queries": 0}
        return {
            "queries": len(frame),
            "top1_accuracy": float(frame["top1_correct"].mean()),
            "recall_at_5": float(frame["recall_at_5"].mean()),
            "recall_at_10": float(frame["recall_at_10"].mean()),
            "mrr_at_10": float(frame["reciprocal_rank"].mean()),
        }

    summary = {
        "documents": len(documents),
        "queries": len(output),
        "warning": "Subset metrics are not comparable to full-corpus production metrics" if args.max_documents else None,
        "overall": metrics(output),
        "reachable_only": metrics(reachable),
        "output": str(args.output.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BM25 + dense-vector material retrieval baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--jsonl", type=Path, default=Path("build/materials.jsonl"))
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("offline_output/evaluation_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("offline_output/evaluation_summary.json"))
    parser.add_argument("--max-documents", type=int, default=100000)
    parser.add_argument("--max-queries", type=int)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

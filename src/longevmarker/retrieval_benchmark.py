from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text('', encoding='utf-8')
        return
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(query_embeddings: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    query_norm = normalize_rows(query_embeddings)
    corpus_norm = normalize_rows(corpus_embeddings)
    return query_norm @ corpus_norm.T


def load_embeddings(prefix: str | Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    prefix = Path(prefix)
    embeddings = np.load(prefix.with_suffix('.embeddings.npy'))
    metadata = load_csv(prefix.with_suffix('.metadata.csv'))
    return embeddings, metadata


def build_qrels_map(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        qrels[row['query_id']][row['doc_id']] = int(row['relevance'])
    return qrels


def filter_queries_by_split(
    query_embeddings: np.ndarray,
    query_meta: list[dict[str, str]],
    split_name: str | None,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    if not split_name:
        return query_embeddings, query_meta
    indices = [i for i, row in enumerate(query_meta) if row.get('split', '') == split_name]
    return query_embeddings[indices], [query_meta[i] for i in indices]


def score_rankings(query_meta, corpus_meta, similarities, qrels_map, top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rankings = []
    metrics_acc = defaultdict(list)
    corpus_ids = [row['row_id'] if 'row_id' in row else row.get('doc_id', '') for row in corpus_meta]
    doc_id_candidates = [row.get('row_id', '') for row in corpus_meta]
    for i, query_row in enumerate(query_meta):
        query_id = query_row['row_id'] if 'row_id' in query_row else query_row.get('query_id', '')
        target_qrels = qrels_map.get(query_id, {})
        scores = similarities[i]
        ranked_idx = np.argsort(-scores)[:top_k]
        reciprocal_rank = 0.0
        dcg = 0.0
        ideal_rels = sorted(target_qrels.values(), reverse=True)
        idcg = sum((2**rel - 1) / np.log2(rank + 2) for rank, rel in enumerate(ideal_rels[:top_k])) or 1.0
        hits5 = 0
        for rank, idx in enumerate(ranked_idx, start=1):
            raw_doc_id = corpus_ids[idx]
            if not raw_doc_id.startswith(('pair:', 'surrogate:', 'intervention:')):
                doc_id = raw_doc_id
            else:
                doc_id = raw_doc_id
            if doc_id not in target_qrels and doc_id_candidates[idx]:
                for prefix in ('pair:', 'surrogate:', 'intervention:'):
                    candidate = prefix + doc_id_candidates[idx]
                    if candidate in target_qrels:
                        doc_id = candidate
                        break
            rel = target_qrels.get(doc_id, 0)
            if reciprocal_rank == 0.0 and rel > 0:
                reciprocal_rank = 1.0 / rank
            if rank <= 5 and rel > 0:
                hits5 = 1
            dcg += (2**rel - 1) / np.log2(rank + 1)
            rankings.append(
                {
                    'query_id': query_id,
                    'rank': rank,
                    'doc_id': doc_id,
                    'score': round(float(scores[idx]), 6),
                    'relevance': rel,
                }
            )
        metrics_acc['mrr@10'].append(reciprocal_rank)
        metrics_acc['recall@5'].append(float(hits5))
        metrics_acc['ndcg@10'].append(dcg / idcg)
    metrics = [
        {'metric': metric, 'value': round(float(np.mean(values)) if values else 0.0, 6)}
        for metric, values in metrics_acc.items()
    ]
    return rankings, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run retrieval benchmark from precomputed embeddings.')
    parser.add_argument('--query-prefix', required=True)
    parser.add_argument('--corpus-prefix', required=True)
    parser.add_argument('--qrels-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--only-split')
    args = parser.parse_args(argv)

    query_embeddings, query_meta = load_embeddings(args.query_prefix)
    corpus_embeddings, corpus_meta = load_embeddings(args.corpus_prefix)
    query_embeddings, query_meta = filter_queries_by_split(query_embeddings, query_meta, args.only_split)
    qrels_map = build_qrels_map(load_csv(args.qrels_csv))
    sims = cosine_similarity_matrix(query_embeddings, corpus_embeddings)
    rankings, metrics = score_rankings(query_meta, corpus_meta, sims, qrels_map, top_k=args.top_k)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / 'rankings.csv', rankings)
    write_csv(output_dir / 'metrics.csv', metrics)
    for row in metrics:
        prefix = f"{args.only_split} " if args.only_split else ''
        print(f"{prefix}{row['metric']}: {row['value']}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

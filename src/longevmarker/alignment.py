from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .retrieval_benchmark import build_qrels_map, load_csv, score_rankings, write_csv


def load_embeddings(prefix: str | Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    prefix = Path(prefix)
    embeddings = np.load(prefix.with_suffix('.embeddings.npy'))
    metadata = load_csv(prefix.with_suffix('.metadata.csv'))
    return embeddings, metadata


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_doc_index(corpus_meta: list[dict[str, str]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, row in enumerate(corpus_meta):
        row_id = row.get('row_id', '')
        if not row_id:
            continue
        index[row_id] = i
        for prefix in ('pair:', 'surrogate:', 'intervention:'):
            index[prefix + row_id] = i
    return index


def build_training_targets(
    query_embeddings: np.ndarray,
    query_meta: list[dict[str, str]],
    corpus_embeddings: np.ndarray,
    qrels_map: dict[str, dict[str, int]],
    doc_index: dict[str, int],
    train_split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_rows = []
    y_rows = []
    used_queries: list[str] = []
    for i, row in enumerate(query_meta):
        if row.get('split', '') != train_split:
            continue
        query_id = row.get('row_id', '')
        positives = qrels_map.get(query_id, {})
        weighted = []
        weights = []
        for doc_id, rel in positives.items():
            idx = doc_index.get(doc_id)
            if idx is None or rel <= 0:
                continue
            weighted.append(corpus_embeddings[idx])
            weights.append(float(rel))
        if not weighted:
            continue
        target = np.average(np.vstack(weighted), axis=0, weights=np.array(weights, dtype=np.float32))
        x_rows.append(query_embeddings[i])
        y_rows.append(target)
        used_queries.append(query_id)
    if not x_rows:
        raise ValueError(f'No training rows found for split={train_split!r}')
    return np.vstack(x_rows), np.vstack(y_rows), used_queries


def fit_ridge_projection(x_train: np.ndarray, y_train: np.ndarray, regularization: float) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    identity = np.eye(x_aug.shape[1], dtype=np.float32)
    identity[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + regularization * identity
    rhs = x_aug.T @ y_train
    return np.linalg.solve(lhs, rhs)


def project_embeddings(query_embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    x_aug = np.concatenate([query_embeddings, np.ones((query_embeddings.shape[0], 1), dtype=np.float32)], axis=1)
    projected = x_aug @ weights
    return normalize_rows(projected)


def build_alignment_metadata(
    query_meta: list[dict[str, str]],
    corpus_meta: list[dict[str, str]],
    train_query_ids: list[str],
    train_split: str,
    regularization: float,
) -> list[dict[str, object]]:
    corpus_model = corpus_meta[0].get('model_name', '') if corpus_meta else ''
    rows = []
    train_query_set = set(train_query_ids)
    for row in query_meta:
        item = dict(row)
        item['alignment_train_split'] = train_split
        item['alignment_regularization'] = regularization
        item['alignment_target_model'] = corpus_model
        item['used_for_alignment_train'] = 'yes' if row.get('row_id', '') in train_query_set else 'no'
        rows.append(item)
    return rows


def parse_splits(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Learn a ridge alignment from molecular embeddings to text embedding space.')
    parser.add_argument('--query-prefix', required=True)
    parser.add_argument('--corpus-prefix', required=True)
    parser.add_argument('--qrels-csv', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--eval-splits', default='train,val,test')
    parser.add_argument('--regularization', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args(argv)

    query_embeddings, query_meta = load_embeddings(args.query_prefix)
    corpus_embeddings, corpus_meta = load_embeddings(args.corpus_prefix)
    qrels_map = build_qrels_map(load_csv(args.qrels_csv))
    doc_index = build_doc_index(corpus_meta)

    x_train, y_train, train_query_ids = build_training_targets(
        query_embeddings=query_embeddings,
        query_meta=query_meta,
        corpus_embeddings=corpus_embeddings,
        qrels_map=qrels_map,
        doc_index=doc_index,
        train_split=args.train_split,
    )
    weights = fit_ridge_projection(x_train, y_train, regularization=args.regularization)
    projected = project_embeddings(query_embeddings, weights)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix('.embeddings.npy'), projected)
    write_csv(
        prefix.with_suffix('.metadata.csv'),
        build_alignment_metadata(query_meta, corpus_meta, train_query_ids, args.train_split, args.regularization),
    )
    np.savez(
        prefix.with_suffix('.alignment.npz'),
        weights=weights,
        train_queries=np.array(train_query_ids, dtype=object),
        train_split=np.array([args.train_split], dtype=object),
        regularization=np.array([args.regularization], dtype=np.float32),
    )

    metrics_rows = []
    for split in parse_splits(args.eval_splits):
        subset_meta = [row for row in query_meta if row.get('split', '') == split]
        if not subset_meta:
            continue
        subset_idx = [i for i, row in enumerate(query_meta) if row.get('split', '') == split]
        subset_embeddings = projected[subset_idx]
        similarities = subset_embeddings @ normalize_rows(corpus_embeddings).T
        _, metrics = score_rankings(subset_meta, corpus_meta, similarities, qrels_map, top_k=args.top_k)
        for metric in metrics:
            metrics_rows.append({'split': split, 'metric': metric['metric'], 'value': metric['value']})
            print(f'{split} {metric["metric"]}: {metric["value"]}')
    write_csv(prefix.with_suffix('.metrics.csv'), metrics_rows)

    print(f'train_queries: {len(train_query_ids)}')
    print(f'projection_input_dim: {query_embeddings.shape[1] if query_embeddings.size else 0}')
    print(f'projection_output_dim: {projected.shape[1] if projected.size else 0}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

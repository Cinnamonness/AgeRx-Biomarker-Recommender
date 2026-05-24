from __future__ import annotations

import argparse
from pathlib import Path

from .twin_tower import (
    MOLECULAR_ENCODER_PRESETS,
    MolecularEncoderConfig,
    TrainConfig,
    load_csv,
    train_twin_tower,
    write_csv,
)


def metrics_map(model_dir: str | Path) -> dict[tuple[str, str], float]:
    rows = load_csv(Path(model_dir) / 'metrics.csv')
    return {(row['split'], row['metric']): float(row['value']) for row in rows}


def run_benchmark(
    output_dir: str | Path,
    molecular_queries_csv: str | Path,
    biomarker_candidates_csv: str | Path,
    qrels_csv: str | Path,
    triplets_csv: str | Path,
    device: str,
    epochs: int,
    early_stopping_patience: int,
    hidden_dim: int,
    embedding_dim: int,
    text_max_features: int,
    batch_size: int,
    top_k: int,
    seed: int,
) -> list[dict[str, object]]:
    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for preset_name, preset in MOLECULAR_ENCODER_PRESETS.items():
        run_dir = benchmark_dir / preset_name
        config = TrainConfig(
            molecular_queries_csv=str(molecular_queries_csv),
            biomarker_candidates_csv=str(biomarker_candidates_csv),
            qrels_csv=str(qrels_csv),
            triplets_csv=str(triplets_csv),
            output_dir=str(run_dir),
            molecular_encoder=MolecularEncoderConfig(
                preset=preset_name,
                model_name=preset['model_name'],
                input_column=preset['input_column'],
                trust_remote_code=bool(preset['trust_remote_code']),
                representation=preset['representation'],
                device=device,
            ),
            text_max_features=text_max_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            batch_size=batch_size,
            top_k=top_k,
            seed=seed,
        )
        status = 'ok'
        error_message = ''
        try:
            summary = train_twin_tower(config)
            metric_lookup = metrics_map(run_dir)
        except Exception as exc:  # pragma: no cover
            status = 'failed'
            error_message = str(exc)
            summary = {'best_val_mrr@10': 0.0, 'epochs_trained': 0}
            metric_lookup = {}
        rows.append(
            {
                'encoder': preset['label'],
                'preset': preset_name,
                'model_name': preset['model_name'],
                'input_column': preset['input_column'],
                'status': status,
                'error': error_message,
                'epochs_trained': summary.get('epochs_trained', 0),
                'best_val_mrr@10': summary.get('best_val_mrr@10', 0.0),
                'train_mrr@10': metric_lookup.get(('train', 'mrr@10'), 0.0),
                'train_recall@5': metric_lookup.get(('train', 'recall@5'), 0.0),
                'train_ndcg@10': metric_lookup.get(('train', 'ndcg@10'), 0.0),
                'val_mrr@10': metric_lookup.get(('val', 'mrr@10'), 0.0),
                'val_recall@5': metric_lookup.get(('val', 'recall@5'), 0.0),
                'val_ndcg@10': metric_lookup.get(('val', 'ndcg@10'), 0.0),
                'test_mrr@10': metric_lookup.get(('test', 'mrr@10'), 0.0),
                'test_recall@5': metric_lookup.get(('test', 'recall@5'), 0.0),
                'test_ndcg@10': metric_lookup.get(('test', 'ndcg@10'), 0.0),
                'run_dir': str(run_dir),
            }
        )
    write_csv(benchmark_dir / 'encoder_comparison.csv', rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Benchmark chemical encoders in the Twin Tower recommender.')
    parser.add_argument('--output-dir', default=str(root / 'outputs' / 'encoder_benchmark'))
    parser.add_argument('--molecular-queries-csv', default=str(root / 'data' / 'embedding' / 'molecular_interventions.csv'))
    parser.add_argument('--biomarker-candidates-csv', default=str(root / 'data' / 'embedding' / 'biomarker_semantic_texts.csv'))
    parser.add_argument('--qrels-csv', default=str(root / 'data' / 'embedding' / 'molecular_biomarker_semantic_qrels.csv'))
    parser.add_argument('--triplets-csv', default=str(root / 'data' / 'embedding' / 'molecule_to_biomarker_semantic_triplets.csv'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stopping-patience', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--text-max-features', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    rows = run_benchmark(
        output_dir=args.output_dir,
        molecular_queries_csv=args.molecular_queries_csv,
        biomarker_candidates_csv=args.biomarker_candidates_csv,
        qrels_csv=args.qrels_csv,
        triplets_csv=args.triplets_csv,
        device=args.device,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        text_max_features=args.text_max_features,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    for row in rows:
        print(f"{row['encoder']}: status={row['status']} test_mrr@10={row['test_mrr@10']}")
    return 0

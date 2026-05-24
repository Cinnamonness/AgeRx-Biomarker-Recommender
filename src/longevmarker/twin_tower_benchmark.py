from __future__ import annotations

import argparse
from pathlib import Path

from .twin_tower import (
    MOLECULAR_ENCODER_PRESETS,
    MolecularEncoderConfig,
    TrainConfig,
    cross_validate,
    write_csv,
)


def read_cv_summary(run_dir: Path) -> dict[str, float]:
    rows_path = run_dir / 'cross_validation_summary.csv'
    values: dict[str, float] = {}
    if not rows_path.exists():
        return values
    import csv
    with rows_path.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            values[row['metric']] = float(row['mean'])
            values[f"{row['metric']}_std"] = float(row['std'])
    return values


def run_benchmark(
    output_dir: str | Path,
    curated_csv: str | Path,
    compound_registry_csv: str | Path,
    compound_components_csv: str | Path,
    seed_interventions_csv: str | Path,
    device: str,
    epochs: int,
    early_stopping_patience: int,
    hidden_dim: int,
    embedding_dim: int,
    query_text_max_features: int,
    candidate_text_max_features: int,
    top_k_metrics: int,
    predict_top_k: int,
    seed: int,
    num_folds: int,
    unfreeze_last_n_layers: int,
) -> list[dict[str, object]]:
    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for preset_name, preset in MOLECULAR_ENCODER_PRESETS.items():
        run_dir = benchmark_dir / preset_name
        config = TrainConfig(
            curated_csv=str(curated_csv),
            compound_registry_csv=str(compound_registry_csv),
            compound_components_csv=str(compound_components_csv),
            seed_interventions_csv=str(seed_interventions_csv),
            output_dir=str(run_dir),
            molecular_encoder=MolecularEncoderConfig(
                preset=preset_name,
                model_name=preset['model_name'],
                representation=preset['representation'],
                trust_remote_code=bool(preset['trust_remote_code']),
                device=device,
                unfreeze_last_n_layers=unfreeze_last_n_layers,
            ),
            query_text_max_features=query_text_max_features,
            candidate_text_max_features=candidate_text_max_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            top_k_metrics=top_k_metrics,
            predict_top_k=predict_top_k,
            seed=seed,
            num_folds=num_folds,
        )
        status = 'ok'
        error_message = ''
        try:
            cv_summary = cross_validate(config)
            summary = {'best_epoch_mean': cv_summary.get('best_epoch_mean', 0)}
            metrics = read_cv_summary(run_dir)
        except Exception as exc:  # pragma: no cover
            status = 'failed'
            error_message = str(exc)
            summary = {}
            metrics = {}
        rows.append(
            {
                'encoder': preset['label'],
                'preset': preset_name,
                'model_name': preset['model_name'],
                'status': status,
                'error': error_message,
                'best_epoch_mean': summary.get('best_epoch_mean', 0),
                'cv_test_mrr@10_mean': metrics.get('test_mrr@10', 0.0),
                'cv_test_mrr@10_std': metrics.get('test_mrr@10_std', 0.0),
                'cv_test_recall@5_mean': metrics.get('test_recall@5', 0.0),
                'cv_test_recall@5_std': metrics.get('test_recall@5_std', 0.0),
                'cv_test_ndcg@10_mean': metrics.get('test_ndcg@10', 0.0),
                'cv_test_ndcg@10_std': metrics.get('test_ndcg@10_std', 0.0),
                'run_dir': str(run_dir),
            }
        )
    rows.sort(key=lambda row: (-float(row['cv_test_mrr@10_mean']), -float(row['cv_test_ndcg@10_mean']), row['encoder']))
    write_csv(benchmark_dir / 'encoder_comparison.csv', rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Benchmark chemical encoders in the Twin Tower recommender with cross-validation.')
    parser.add_argument('--output-dir', default=str(root / 'outputs' / 'encoder_benchmark'))
    parser.add_argument('--curated-csv', default=str(root / 'data' / 'processed' / 'curated_biomarkers.csv'))
    parser.add_argument('--compound-registry-csv', default=str(root / 'data' / 'processed' / 'compound_registry.csv'))
    parser.add_argument('--compound-components-csv', default=str(root / 'data' / 'processed' / 'compound_components.csv'))
    parser.add_argument('--seed-interventions-csv', default=str(root / 'data' / 'seed_interventions.csv'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--early-stopping-patience', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--query-text-max-features', type=int, default=2048)
    parser.add_argument('--candidate-text-max-features', type=int, default=4096)
    parser.add_argument('--top-k-metrics', type=int, default=10)
    parser.add_argument('--predict-top-k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--unfreeze-last-n-layers', type=int, default=1)
    args = parser.parse_args(argv)

    rows = run_benchmark(
        output_dir=args.output_dir,
        curated_csv=args.curated_csv,
        compound_registry_csv=args.compound_registry_csv,
        compound_components_csv=args.compound_components_csv,
        seed_interventions_csv=args.seed_interventions_csv,
        device=args.device,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        query_text_max_features=args.query_text_max_features,
        candidate_text_max_features=args.candidate_text_max_features,
        top_k_metrics=args.top_k_metrics,
        predict_top_k=args.predict_top_k,
        seed=args.seed,
        num_folds=args.num_folds,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    )
    for row in rows:
        print(f"{row['encoder']}: status={row['status']} cv_test_mrr@10_mean={row['cv_test_mrr@10_mean']}")
    return 0

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .embedding_data import (
    build_biomarker_semantic_texts,
    build_pair_texts,
    build_surrogate_texts,
    load_curated_rows,
    pick_negative_pair,
    pick_negative_surrogate,
)


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


def build_molecular_interventions(
    curated_rows: list[dict[str, object]],
    registry_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    grouped_pairs: dict[str, list[dict[str, object]]] = {}
    for row in curated_rows:
        grouped_pairs.setdefault(str(row['intervention']), []).append(row)
    results: list[dict[str, object]] = []
    for registry in registry_rows:
        intervention = registry['intervention']
        pairs = sorted(
            grouped_pairs[intervention],
            key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row['surrogate'])),
        )
        results.append(
            {
                'query_id': f"mol_query_{registry['intervention_id']}",
                'intervention_id': registry['intervention_id'],
                'intervention': intervention,
                'split': pairs[0]['split'],
                'molecule_type': registry['molecule_type'],
                'component_count': registry['component_count'],
                'canonical_smiles': registry['canonical_smiles'],
                'connectivity_smiles': registry['connectivity_smiles'],
                'component_canonical_smiles': registry['component_canonical_smiles'],
                'component_connectivity_smiles': registry['component_connectivity_smiles'],
                'mechanism_hint': pairs[0]['mechanism_hint'],
                'expected_surrogates': '|'.join(row['surrogate'] for row in pairs),
                'expected_pair_ids': '|'.join(row['pair_id'] for row in pairs),
                'primary_pathways': '|'.join(_top_unique(pairs, 'pathway', 4)),
                'primary_hallmarks': '|'.join(_top_unique(pairs, 'hallmark', 4)),
                'selfies_source_smiles': registry['canonical_smiles'],
                'selfies_sequence': '',
                'selfies_conversion_status': 'pending_selfies_conversion',
            }
        )
    return results


def build_molecular_pair_qrels(
    molecular_queries: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped_pairs: dict[str, list[dict[str, object]]] = {}
    for row in pair_rows:
        grouped_pairs.setdefault(str(row['intervention']), []).append(row)
    qrels: list[dict[str, object]] = []
    for query in molecular_queries:
        pairs = sorted(
            grouped_pairs[str(query['intervention'])],
            key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row['surrogate'])),
        )
        for rank, pair in enumerate(pairs, start=1):
            qrels.append(
                {
                    'query_id': query['query_id'],
                    'doc_id': f"pair:{pair['pair_id']}",
                    'relevance': 3 if rank <= 3 else 2,
                    'target_intervention': query['intervention'],
                }
            )
    return qrels


def build_molecular_surrogate_qrels(
    molecular_queries: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped_pairs: dict[str, list[dict[str, object]]] = {}
    for row in pair_rows:
        grouped_pairs.setdefault(str(row['intervention']), []).append(row)
    qrels: list[dict[str, object]] = []
    for query in molecular_queries:
        pairs = sorted(
            grouped_pairs[str(query['intervention'])],
            key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row['surrogate'])),
        )
        for rank, pair in enumerate(pairs, start=1):
            qrels.append(
                {
                    'query_id': query['query_id'],
                    'doc_id': f"surrogate:{pair['surrogate_id']}",
                    'relevance': 3 if rank <= 3 else 2,
                    'target_intervention': query['intervention'],
                }
            )
    return qrels


def build_molecular_biomarker_semantic_qrels(
    molecular_queries: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped_pairs: dict[str, list[dict[str, object]]] = {}
    for row in pair_rows:
        grouped_pairs.setdefault(str(row['intervention']), []).append(row)
    qrels: list[dict[str, object]] = []
    for query in molecular_queries:
        pairs = sorted(
            grouped_pairs[str(query['intervention'])],
            key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row['surrogate'])),
        )
        for rank, pair in enumerate(pairs, start=1):
            qrels.append(
                {
                    'query_id': query['query_id'],
                    'doc_id': f"surrogate:{pair['surrogate_id']}",
                    'relevance': 3 if rank <= 3 else 2,
                    'target_intervention': query['intervention'],
                }
            )
    return qrels


def build_molecule_to_pair_triplets(
    molecular_queries: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    pair_groups: dict[str, list[dict[str, object]]] = {}
    for row in pair_rows:
        pair_groups.setdefault(str(row['intervention']), []).append(row)
    rows: list[dict[str, object]] = []
    for query in molecular_queries:
        for pair in pair_groups[str(query['intervention'])]:
            negative_pair, negative_strategy = pick_negative_pair(pair, pair_rows)
            rows.append(
                {
                    'triplet_id': f"mol_pair_triplet_{pair['pair_id']}",
                    'split': query['split'],
                    'intervention': query['intervention'],
                    'canonical_smiles': query['canonical_smiles'],
                    'connectivity_smiles': query['connectivity_smiles'],
                    'molecule_type': query['molecule_type'],
                    'component_count': query['component_count'],
                    'positive_id': f"pair:{pair['pair_id']}",
                    'positive_text': pair['embedding_text'],
                    'negative_id': f"pair:{negative_pair['pair_id']}",
                    'negative_text': negative_pair['embedding_text'],
                    'negative_strategy': negative_strategy,
                    'target_surrogate': pair['surrogate'],
                }
            )
    return rows


def build_molecule_to_surrogate_triplets(
    molecular_queries: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    surrogate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    pair_groups: dict[str, list[dict[str, object]]] = {}
    for row in pair_rows:
        pair_groups.setdefault(str(row['intervention']), []).append(row)
    relevant_surrogates_by_intervention = {
        intervention: {pair['surrogate'] for pair in pairs}
        for intervention, pairs in pair_groups.items()
    }
    rows: list[dict[str, object]] = []
    surrogate_map = {row['surrogate']: row for row in surrogate_rows}
    for query in molecular_queries:
        for pair in pair_groups[str(query['intervention'])]:
            negative_surrogate, negative_strategy = pick_negative_surrogate(pair, surrogate_rows, relevant_surrogates_by_intervention)
            positive_surrogate = surrogate_map[pair['surrogate']]
            rows.append(
                {
                    'triplet_id': f"mol_surrogate_triplet_{pair['pair_id']}",
                    'split': query['split'],
                    'intervention': query['intervention'],
                    'canonical_smiles': query['canonical_smiles'],
                    'connectivity_smiles': query['connectivity_smiles'],
                    'molecule_type': query['molecule_type'],
                    'component_count': query['component_count'],
                    'positive_id': f"surrogate:{positive_surrogate['surrogate_id']}",
                    'positive_text': positive_surrogate['embedding_text'],
                    'negative_id': f"surrogate:{negative_surrogate['surrogate_id']}",
                    'negative_text': negative_surrogate['embedding_text'],
                    'negative_strategy': negative_strategy,
                    'target_surrogate': pair['surrogate'],
                }
            )
    return rows


def build_molecule_to_biomarker_semantic_triplets(
    molecular_queries: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    biomarker_semantic_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    pair_groups: dict[str, list[dict[str, object]]] = {}
    for row in pair_rows:
        pair_groups.setdefault(str(row['intervention']), []).append(row)
    relevant_surrogates_by_intervention = {
        intervention: {pair['surrogate'] for pair in pairs}
        for intervention, pairs in pair_groups.items()
    }
    rows: list[dict[str, object]] = []
    biomarker_map = {row['surrogate']: row for row in biomarker_semantic_rows}
    for query in molecular_queries:
        for pair in pair_groups[str(query['intervention'])]:
            negative_biomarker, negative_strategy = pick_negative_surrogate(pair, biomarker_semantic_rows, relevant_surrogates_by_intervention)
            positive_biomarker = biomarker_map[pair['surrogate']]
            rows.append(
                {
                    'triplet_id': f"mol_biomarker_semantic_triplet_{pair['pair_id']}",
                    'split': query['split'],
                    'intervention': query['intervention'],
                    'canonical_smiles': query['canonical_smiles'],
                    'connectivity_smiles': query['connectivity_smiles'],
                    'molecule_type': query['molecule_type'],
                    'component_count': query['component_count'],
                    'positive_id': f"surrogate:{positive_biomarker['surrogate_id']}",
                    'positive_text': positive_biomarker['embedding_text'],
                    'negative_id': f"surrogate:{negative_biomarker['surrogate_id']}",
                    'negative_text': negative_biomarker['embedding_text'],
                    'negative_strategy': negative_strategy,
                    'target_surrogate': pair['surrogate'],
                }
            )
    return rows


def build_encoder_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in ['ChemBERT', 'MolFormer']:
        for smiles_column in ['canonical_smiles', 'connectivity_smiles']:
            rows.append(
                {
                    'encoder_family': family,
                    'input_representation': 'smiles',
                    'intervention_input_column': smiles_column,
                    'combination_strategy': 'single_direct_or_component_mean_pool',
                    'pair_task_query_table': 'molecular_queries.csv',
                    'pair_task_qrels_table': 'molecular_pair_qrels.csv',
                    'pair_task_corpus_table': 'pair_texts.csv',
                    'surrogate_task_query_table': 'molecular_queries.csv',
                    'surrogate_task_qrels_table': 'molecular_surrogate_qrels.csv',
                    'surrogate_task_corpus_table': 'surrogate_texts.csv',
                    'biomarker_semantic_task_query_table': 'molecular_queries.csv',
                    'biomarker_semantic_task_qrels_table': 'molecular_biomarker_semantic_qrels.csv',
                    'biomarker_semantic_task_corpus_table': 'biomarker_semantic_texts.csv',
                    'training_triplets_pair': 'molecule_to_pair_triplets.csv',
                    'training_triplets_surrogate': 'molecule_to_surrogate_triplets.csv',
                    'training_triplets_biomarker_semantic': 'molecule_to_biomarker_semantic_triplets.csv',
                    'recommended_metrics': 'MRR@10|Recall@5|nDCG@10',
                }
            )
    rows.append(
        {
            'encoder_family': 'Selformer',
            'input_representation': 'selfies',
            'intervention_input_column': 'selfies_sequence',
            'combination_strategy': 'single_direct_or_component_mean_pool',
            'pair_task_query_table': 'molecular_queries.csv',
            'pair_task_qrels_table': 'molecular_pair_qrels.csv',
            'pair_task_corpus_table': 'pair_texts.csv',
            'surrogate_task_query_table': 'molecular_queries.csv',
            'surrogate_task_qrels_table': 'molecular_surrogate_qrels.csv',
            'surrogate_task_corpus_table': 'surrogate_texts.csv',
            'biomarker_semantic_task_query_table': 'molecular_queries.csv',
            'biomarker_semantic_task_qrels_table': 'molecular_biomarker_semantic_qrels.csv',
            'biomarker_semantic_task_corpus_table': 'biomarker_semantic_texts.csv',
            'training_triplets_pair': 'molecule_to_pair_triplets.csv',
            'training_triplets_surrogate': 'molecule_to_surrogate_triplets.csv',
            'training_triplets_biomarker_semantic': 'molecule_to_biomarker_semantic_triplets.csv',
            'recommended_metrics': 'MRR@10|Recall@5|nDCG@10',
        }
    )
    return rows


def build_model_comparison_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    task_specs = {
        'smiles_to_pair_retrieval': ('pair_texts.csv', 'molecular_pair_qrels.csv', 'molecule_to_pair_triplets.csv'),
        'smiles_to_surrogate_retrieval': ('surrogate_texts.csv', 'molecular_surrogate_qrels.csv', 'molecule_to_surrogate_triplets.csv'),
        'smiles_to_biomarker_semantic_retrieval': ('biomarker_semantic_texts.csv', 'molecular_biomarker_semantic_qrels.csv', 'molecule_to_biomarker_semantic_triplets.csv'),
    }
    for family in ['ChemBERT', 'MolFormer']:
        for smiles_column in ['canonical_smiles', 'connectivity_smiles']:
            for task, (corpus_table, qrels_table, training_table) in task_specs.items():
                rows.append(
                    {
                        'run_id': f"{family.lower()}_{smiles_column}_{task}",
                        'encoder_family': family,
                        'input_representation': 'smiles',
                        'intervention_input_column': smiles_column,
                        'task': task,
                        'query_table': 'molecular_queries.csv',
                        'corpus_table': corpus_table,
                        'qrels_table': qrels_table,
                        'training_table': training_table,
                        'primary_metric': 'MRR@10',
                        'secondary_metrics': 'Recall@5|nDCG@10',
                    }
                )
    for task, (corpus_table, qrels_table, training_table) in task_specs.items():
        rows.append(
            {
                'run_id': f"selformer_selfies_sequence_{task}",
                'encoder_family': 'Selformer',
                'input_representation': 'selfies',
                'intervention_input_column': 'selfies_sequence',
                'task': task,
                'query_table': 'molecular_queries.csv',
                'corpus_table': corpus_table,
                'qrels_table': qrels_table,
                'training_table': training_table,
                'primary_metric': 'MRR@10',
                'secondary_metrics': 'Recall@5|nDCG@10',
            }
        )
    return rows


def build_manifest(record_counts: dict[str, int]) -> list[dict[str, object]]:
    return [
        {
            'file_name': file_name,
            'record_count': record_count,
            'purpose': purpose,
        }
        for file_name, record_count, purpose in [
            ('molecular_interventions.csv', record_counts['molecular_interventions.csv'], 'one row per molecular query intervention'),
            ('molecular_pair_qrels.csv', record_counts['molecular_pair_qrels.csv'], 'pair retrieval relevance labels'),
            ('molecular_surrogate_qrels.csv', record_counts['molecular_surrogate_qrels.csv'], 'surrogate retrieval relevance labels'),
            ('molecular_biomarker_semantic_qrels.csv', record_counts['molecular_biomarker_semantic_qrels.csv'], 'biomarker semantic retrieval relevance labels'),
            ('molecule_to_pair_triplets.csv', record_counts['molecule_to_pair_triplets.csv'], 'cross-modal training triplets for pair retrieval'),
            ('molecule_to_surrogate_triplets.csv', record_counts['molecule_to_surrogate_triplets.csv'], 'cross-modal training triplets for surrogate retrieval'),
            ('molecule_to_biomarker_semantic_triplets.csv', record_counts['molecule_to_biomarker_semantic_triplets.csv'], 'cross-modal training triplets for biomarker semantic retrieval'),
            ('encoder_candidates.csv', record_counts['encoder_candidates.csv'], 'encoder comparison candidate grid'),
            ('model_comparison_runs.csv', record_counts['model_comparison_runs.csv'], 'benchmark run plan across encoders and tasks'),
        ]
    ]


def _top_unique(rows: list[dict[str, object]], key: str, limit: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.add(value)
            ordered.append(value)
        if len(ordered) >= limit:
            break
    return ordered


def prepare_molecular_embedding_tables(
    curated_path: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    curated_rows = load_curated_rows(curated_path)
    pair_rows = build_pair_texts(curated_rows)
    surrogate_rows = build_surrogate_texts(curated_rows)
    biomarker_semantic_rows = build_biomarker_semantic_texts(curated_rows)
    registry_rows = load_csv(registry_path)

    molecular_interventions = build_molecular_interventions(curated_rows, registry_rows)
    molecular_pair_qrels = build_molecular_pair_qrels(molecular_interventions, pair_rows)
    molecular_surrogate_qrels = build_molecular_surrogate_qrels(molecular_interventions, pair_rows)
    molecular_biomarker_semantic_qrels = build_molecular_biomarker_semantic_qrels(molecular_interventions, pair_rows)
    molecule_to_pair_triplets = build_molecule_to_pair_triplets(molecular_interventions, pair_rows)
    molecule_to_surrogate_triplets = build_molecule_to_surrogate_triplets(molecular_interventions, pair_rows, surrogate_rows)
    molecule_to_biomarker_semantic_triplets = build_molecule_to_biomarker_semantic_triplets(molecular_interventions, pair_rows, biomarker_semantic_rows)
    encoder_candidates = build_encoder_candidates()
    model_comparison_runs = build_model_comparison_runs()

    output = Path(output_dir)
    write_csv(output / 'molecular_interventions.csv', molecular_interventions)
    write_csv(output / 'molecular_pair_qrels.csv', molecular_pair_qrels)
    write_csv(output / 'molecular_surrogate_qrels.csv', molecular_surrogate_qrels)
    write_csv(output / 'molecular_biomarker_semantic_qrels.csv', molecular_biomarker_semantic_qrels)
    write_csv(output / 'molecule_to_pair_triplets.csv', molecule_to_pair_triplets)
    write_csv(output / 'molecule_to_surrogate_triplets.csv', molecule_to_surrogate_triplets)
    write_csv(output / 'molecule_to_biomarker_semantic_triplets.csv', molecule_to_biomarker_semantic_triplets)
    write_csv(output / 'encoder_candidates.csv', encoder_candidates)
    write_csv(output / 'model_comparison_runs.csv', model_comparison_runs)

    record_counts = {
        'molecular_interventions.csv': len(molecular_interventions),
        'molecular_pair_qrels.csv': len(molecular_pair_qrels),
        'molecular_surrogate_qrels.csv': len(molecular_surrogate_qrels),
        'molecular_biomarker_semantic_qrels.csv': len(molecular_biomarker_semantic_qrels),
        'molecule_to_pair_triplets.csv': len(molecule_to_pair_triplets),
        'molecule_to_surrogate_triplets.csv': len(molecule_to_surrogate_triplets),
        'molecule_to_biomarker_semantic_triplets.csv': len(molecule_to_biomarker_semantic_triplets),
        'encoder_candidates.csv': len(encoder_candidates),
        'model_comparison_runs.csv': len(model_comparison_runs),
    }
    write_csv(output / 'molecular_manifest.csv', build_manifest(record_counts))
    return record_counts


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Prepare molecular embedding tables for chemical encoders.')
    parser.add_argument('--curated-path', default=str(root / 'data' / 'processed' / 'curated_biomarkers.csv'))
    parser.add_argument('--registry-path', default=str(root / 'data' / 'processed' / 'compound_registry.csv'))
    parser.add_argument('--output-dir', default=str(root / 'data' / 'embedding'))
    args = parser.parse_args(argv)
    counts = prepare_molecular_embedding_tables(args.curated_path, args.registry_path, args.output_dir)
    for file_name, record_count in counts.items():
        print(f'{file_name}: {record_count}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

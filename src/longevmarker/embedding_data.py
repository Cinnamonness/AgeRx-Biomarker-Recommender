from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Iterable

SPLIT_MAP = {
    'Metformin': 'train',
    'Acarbose': 'train',
    'Liraglutide': 'train',
    'Empagliflozin': 'train',
    'Fisetin': 'train',
    'Nicotinamide Riboside': 'train',
    'Resveratrol': 'train',
    'Spermidine': 'train',
    'Rapamycin': 'val',
    'NMN': 'val',
    'Semaglutide': 'test',
    'Dasatinib + Quercetin': 'test',
}


def load_curated_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key in ['evidence_score', 'surrogate_confidence', 'literature_support']:
                parsed[key] = float(row[key])
            parsed['article_count'] = int(row['article_count'])
            parsed['split'] = split_for_intervention(row['intervention'])
            parsed['intervention_id'] = slugify(row['intervention'])
            parsed['surrogate_id'] = slugify(row['surrogate'])
            rows.append(parsed)
    return rows


def split_for_intervention(intervention: str) -> str:
    return SPLIT_MAP.get(intervention, 'train')


def slugify(text: str) -> str:
    normalized = re.sub(r'[^a-z0-9]+', '_', text.lower())
    return normalized.strip('_')


def top_ranked_unique(rows: Iterable[dict[str, object]], key: str, limit: int = 4) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    sorted_rows = sorted(
        rows,
        key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row[key])),
    )
    for row in sorted_rows:
        value = str(row[key]).strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
        if len(ordered) >= limit:
            break
    return ordered


def split_pipe(text: str) -> list[str]:
    return [item.strip() for item in str(text).split('|') if item.strip()]


def mechanism_themes_from_hints(hints: Iterable[str], limit: int = 6) -> list[str]:
    theme_rules = [
        ('ampk', 'AMPK activation'),
        ('mtor', 'mTOR inhibition'),
        ('glp-1', 'GLP-1 receptor agonism'),
        ('sglt2', 'SGLT2 inhibition'),
        ('senolytic', 'senolytic activity'),
        ('autophagy', 'autophagy induction'),
        ('nad', 'NAD+ restoration'),
        ('riboside', 'NAD+ restoration'),
        ('nmn', 'NAD+ restoration'),
        ('mitochond', 'mitochondrial bioenergetics'),
        ('glycosuria', 'glycosuric glucose lowering'),
        ('glucose', 'glucose regulation'),
        ('insulin', 'insulin sensitization'),
        ('satiety', 'satiety signaling'),
        ('polyamine', 'polyamine signaling'),
        ('spermidine', 'polyamine signaling'),
        ('sirtuin', 'sirtuin activation'),
        ('polyphenol', 'polyphenol stress response'),
        ('stress response', 'integrated stress response'),
    ]
    themes: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        lowered = hint.lower()
        for needle, label in theme_rules:
            if needle in lowered and label not in seen:
                seen.add(label)
                themes.append(label)
                if len(themes) >= limit:
                    return themes
    return themes


def top_reasons(rows: Iterable[dict[str, object]], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    sorted_rows = sorted(
        rows,
        key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row['recommendation_reason'])),
    )
    for row in sorted_rows:
        value = str(row['recommendation_reason']).strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
        if len(ordered) >= limit:
            break
    return ordered


def group_by(rows: Iterable[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def build_intervention_texts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = group_by(rows, 'intervention')
    results: list[dict[str, object]] = []
    for intervention, intervention_rows in sorted(grouped.items()):
        first = intervention_rows[0]
        top_surrogates = top_ranked_unique(intervention_rows, 'surrogate', limit=5)
        pathways = top_ranked_unique(intervention_rows, 'pathway', limit=4)
        hallmarks = top_ranked_unique(intervention_rows, 'hallmark', limit=4)
        categories = top_ranked_unique(intervention_rows, 'surrogate_category', limit=4)
        similar = split_pipe(str(first['similar_interventions']))
        embedding_text = (
            f"Drug intervention: {intervention}. "
            f"Mechanism: {first['mechanism_hint']}. "
            f"Top surrogate biomarkers: {', '.join(top_surrogates)}. "
            f"Biomarker categories: {', '.join(categories)}. "
            f"Pathways: {', '.join(pathways)}. "
            f"Aging hallmarks: {', '.join(hallmarks)}. "
            f"Similar interventions: {', '.join(similar)}."
        )
        results.append(
            {
                'intervention_id': first['intervention_id'],
                'intervention': intervention,
                'split': first['split'],
                'mechanism_hint': first['mechanism_hint'],
                'top_surrogates': '|'.join(top_surrogates),
                'surrogate_categories': '|'.join(categories),
                'pathways': '|'.join(pathways),
                'hallmarks': '|'.join(hallmarks),
                'similar_interventions': first['similar_interventions'],
                'pair_count': len(intervention_rows),
                'embedding_text': embedding_text,
            }
        )
    return results


def build_surrogate_texts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = group_by(rows, 'surrogate')
    results: list[dict[str, object]] = []
    for surrogate, surrogate_rows in sorted(grouped.items()):
        first = surrogate_rows[0]
        linked_interventions = top_ranked_unique(surrogate_rows, 'intervention', limit=8)
        pathways = top_ranked_unique(surrogate_rows, 'pathway', limit=4)
        hallmarks = top_ranked_unique(surrogate_rows, 'hallmark', limit=4)
        mean_evidence = sum(float(row['evidence_score']) for row in surrogate_rows) / len(surrogate_rows)
        mean_surrogate_conf = sum(float(row['surrogate_confidence']) for row in surrogate_rows) / len(surrogate_rows)
        embedding_text = (
            f"Surrogate biomarker: {surrogate}. "
            f"Category: {first['surrogate_category']}. "
            f"Pathways: {', '.join(pathways)}. "
            f"Aging hallmarks: {', '.join(hallmarks)}. "
            f"Observed across interventions: {', '.join(linked_interventions)}."
        )
        results.append(
            {
                'surrogate_id': first['surrogate_id'],
                'surrogate': surrogate,
                'surrogate_category': first['surrogate_category'],
                'pathways': '|'.join(pathways),
                'hallmarks': '|'.join(hallmarks),
                'linked_interventions': '|'.join(linked_interventions),
                'source_pair_count': len(surrogate_rows),
                'mean_evidence_score': round(mean_evidence, 3),
                'mean_surrogate_confidence': round(mean_surrogate_conf, 3),
                'embedding_text': embedding_text,
            }
        )
    return results


def build_biomarker_semantic_texts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = group_by(rows, 'surrogate')
    results: list[dict[str, object]] = []
    for surrogate, surrogate_rows in sorted(grouped.items()):
        first = surrogate_rows[0]
        linked_interventions = top_ranked_unique(surrogate_rows, 'intervention', limit=8)
        pathways = top_ranked_unique(surrogate_rows, 'pathway', limit=4)
        hallmarks = top_ranked_unique(surrogate_rows, 'hallmark', limit=4)
        mechanism_hints = top_ranked_unique(surrogate_rows, 'mechanism_hint', limit=6)
        mechanism_themes = mechanism_themes_from_hints(mechanism_hints, limit=6)
        reasons = top_reasons(surrogate_rows, limit=3)
        mean_evidence = sum(float(row['evidence_score']) for row in surrogate_rows) / len(surrogate_rows)
        mean_surrogate_conf = sum(float(row['surrogate_confidence']) for row in surrogate_rows) / len(surrogate_rows)
        mean_literature = sum(float(row['literature_support']) for row in surrogate_rows) / len(surrogate_rows)
        embedding_text = (
            f"Biomarker semantic profile: {surrogate}. "
            f"Category: {first['surrogate_category']}. "
            f"Pathways: {', '.join(pathways)}. "
            f"Aging hallmarks: {', '.join(hallmarks)}. "
            f"Mechanism-of-action contexts: {', '.join(mechanism_hints)}. "
            f"Mechanism themes: {', '.join(mechanism_themes) if mechanism_themes else 'not specified'}. "
            f"Linked drug interventions: {', '.join(linked_interventions)}. "
            f"Why informative: {' '.join(reasons)} "
            f"Mean evidence score: {mean_evidence:.3f}. "
            f"Mean surrogate confidence: {mean_surrogate_conf:.3f}. "
            f"Mean literature support: {mean_literature:.3f}."
        )
        results.append(
            {
                'surrogate_id': first['surrogate_id'],
                'surrogate': surrogate,
                'surrogate_category': first['surrogate_category'],
                'pathways': '|'.join(pathways),
                'hallmarks': '|'.join(hallmarks),
                'linked_interventions': '|'.join(linked_interventions),
                'mechanism_hints': '|'.join(mechanism_hints),
                'mechanism_themes': '|'.join(mechanism_themes),
                'top_reasons': '|'.join(reasons),
                'source_pair_count': len(surrogate_rows),
                'mean_evidence_score': round(mean_evidence, 3),
                'mean_surrogate_confidence': round(mean_surrogate_conf, 3),
                'mean_literature_support': round(mean_literature, 3),
                'embedding_text': embedding_text,
            }
        )
    return results


def build_pair_texts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (str(item['intervention']), -float(item['evidence_score']), str(item['surrogate']))):
        embedding_text = (
            f"Drug intervention: {row['intervention']}. "
            f"Mechanism: {row['mechanism_hint']}. "
            f"Recommended surrogate: {row['surrogate']}. "
            f"Biomarker category: {row['surrogate_category']}. "
            f"Pathway: {row['pathway']}. "
            f"Aging hallmark: {row['hallmark']}. "
            f"Evidence score: {row['evidence_score']:.3f}. "
            f"Surrogate confidence: {row['surrogate_confidence']:.3f}. "
            f"Literature support: {row['literature_support']:.3f}. "
            f"Reason: {row['recommendation_reason']}"
        )
        results.append(
            {
                'pair_id': row['pair_id'],
                'intervention_id': row['intervention_id'],
                'surrogate_id': row['surrogate_id'],
                'split': row['split'],
                'intervention': row['intervention'],
                'mechanism_hint': row['mechanism_hint'],
                'surrogate': row['surrogate'],
                'surrogate_category': row['surrogate_category'],
                'pathway': row['pathway'],
                'hallmark': row['hallmark'],
                'evidence_score': row['evidence_score'],
                'surrogate_confidence': row['surrogate_confidence'],
                'literature_support': row['literature_support'],
                'article_count': row['article_count'],
                'similar_interventions': row['similar_interventions'],
                'recommendation_reason': row['recommendation_reason'],
                'embedding_text': embedding_text,
            }
        )
    return results


def build_retrieval_corpus(
    intervention_rows: list[dict[str, object]],
    surrogate_rows: list[dict[str, object]],
    biomarker_semantic_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in intervention_rows:
        results.append(
            {
                'doc_id': f"intervention:{row['intervention_id']}",
                'doc_type': 'intervention',
                'title': row['intervention'],
                'text': row['embedding_text'],
                'split': row['split'],
                'source_id': row['intervention_id'],
                'ranking_weight': 1.0,
            }
        )
    for row in surrogate_rows:
        results.append(
            {
                'doc_id': f"surrogate:{row['surrogate_id']}",
                'doc_type': 'surrogate',
                'title': row['surrogate'],
                'text': row['embedding_text'],
                'split': 'shared',
                'source_id': row['surrogate_id'],
                'ranking_weight': round(float(row['mean_evidence_score']), 3),
            }
        )
    for row in biomarker_semantic_rows:
        results.append(
            {
                'doc_id': f"biomarker_semantic:{row['surrogate_id']}",
                'doc_type': 'biomarker_semantic',
                'title': f"{row['surrogate']} semantic profile",
                'text': row['embedding_text'],
                'split': 'shared',
                'source_id': row['surrogate_id'],
                'ranking_weight': round(float(row['mean_evidence_score']), 3),
            }
        )
    for row in pair_rows:
        results.append(
            {
                'doc_id': f"pair:{row['pair_id']}",
                'doc_type': 'pair',
                'title': f"{row['intervention']} -> {row['surrogate']}",
                'text': row['embedding_text'],
                'split': row['split'],
                'source_id': row['pair_id'],
                'ranking_weight': round(0.6 * float(row['evidence_score']) + 0.4 * float(row['surrogate_confidence']), 3),
            }
        )
    return results


def build_queries(
    pair_rows: list[dict[str, object]],
    intervention_rows: list[dict[str, object]],
    demo_query_path: str | Path,
) -> list[dict[str, object]]:
    pair_groups = group_by(pair_rows, 'intervention')
    intervention_map = {row['intervention']: row for row in intervention_rows}
    queries: list[dict[str, object]] = []

    for intervention, intervention_pairs in sorted(pair_groups.items()):
        intervention_row = intervention_map[intervention]
        top_surrogates = split_pipe(str(intervention_row['top_surrogates']))
        pathways = split_pipe(str(intervention_row['pathways']))
        hallmarks = split_pipe(str(intervention_row['hallmarks']))
        pair_ids = [row['pair_id'] for row in intervention_pairs]
        expected_surrogates = '|'.join(row['surrogate'] for row in intervention_pairs)
        templates = [
            (
                'mechanism_prompt',
                f"{intervention}-like drug with {intervention_row['mechanism_hint']}. Which surrogate biomarkers should I track in a short aging trial?",
                f"Recommend biomarkers for a compound similar to {intervention}.",
            ),
            (
                'biology_prompt',
                f"Need surrogate endpoints for a drug targeting {', '.join(pathways[:2])} and {', '.join(hallmarks[:2])}.",
                f"Biology-driven retrieval prompt for {intervention}.",
            ),
            (
                'trial_prompt',
                f"Short clinical trial for {intervention}. Prioritize biomarkers such as {', '.join(top_surrogates[:3])} or similar alternatives.",
                f"Trial-design prompt for {intervention}.",
            ),
        ]
        for query_type, query_text, story in templates:
            queries.append(
                {
                    'query_id': f"query_{slugify(intervention)}_{query_type}",
                    'split': intervention_row['split'],
                    'source': 'templated',
                    'query_type': query_type,
                    'primary_intervention': intervention,
                    'expected_interventions': intervention,
                    'expected_pair_ids': '|'.join(pair_ids),
                    'expected_surrogates': expected_surrogates,
                    'story': story,
                    'query_text': query_text,
                }
            )

    with Path(demo_query_path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            expected_interventions = split_pipe(row['expected_interventions'])
            primary = expected_interventions[0]
            related_pairs: list[str] = []
            related_surrogates: list[str] = []
            for intervention in expected_interventions:
                related_pairs.extend([pair['pair_id'] for pair in pair_groups.get(intervention, [])])
                related_surrogates.extend([pair['surrogate'] for pair in pair_groups.get(intervention, [])])
            queries.append(
                {
                    'query_id': f'demo_query_{index}',
                    'split': split_for_intervention(primary),
                    'source': 'demo',
                    'query_type': 'demo_prompt',
                    'primary_intervention': primary,
                    'expected_interventions': row['expected_interventions'],
                    'expected_pair_ids': '|'.join(unique_in_order(related_pairs)),
                    'expected_surrogates': '|'.join(unique_in_order(related_surrogates)),
                    'story': row['story'],
                    'query_text': row['query_text'],
                }
            )

    return queries


def build_qrels(queries: list[dict[str, object]], pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pair_groups = group_by(pair_rows, 'intervention')
    ranked_groups = {
        intervention: sorted(
            rows,
            key=lambda row: (-float(row['evidence_score']), -float(row['surrogate_confidence']), str(row['surrogate'])),
        )
        for intervention, rows in pair_groups.items()
    }
    qrels: list[dict[str, object]] = []
    for query in queries:
        primary = str(query['primary_intervention'])
        expected_interventions = split_pipe(str(query['expected_interventions']))
        for intervention in expected_interventions:
            for rank, pair in enumerate(ranked_groups.get(intervention, []), start=1):
                if intervention == primary:
                    relevance = 3 if rank <= 3 else 2
                else:
                    relevance = 1 if rank <= 3 else 0
                if relevance <= 0:
                    continue
                qrels.append(
                    {
                        'query_id': query['query_id'],
                        'doc_id': f"pair:{pair['pair_id']}",
                        'relevance': relevance,
                        'target_intervention': intervention,
                    }
                )
    return qrels


def pick_negative_pair(row: dict[str, object], pair_rows: list[dict[str, object]]) -> tuple[dict[str, object], str]:
    same_category = [
        candidate
        for candidate in pair_rows
        if candidate['intervention'] != row['intervention'] and candidate['surrogate_category'] == row['surrogate_category']
    ]
    if same_category:
        same_category.sort(key=lambda candidate: (-float(candidate['evidence_score']), str(candidate['pair_id'])))
        return same_category[0], 'same_category'

    same_hallmark = [
        candidate
        for candidate in pair_rows
        if candidate['intervention'] != row['intervention'] and candidate['hallmark'] == row['hallmark']
    ]
    if same_hallmark:
        same_hallmark.sort(key=lambda candidate: (-float(candidate['evidence_score']), str(candidate['pair_id'])))
        return same_hallmark[0], 'same_hallmark'

    fallback = [candidate for candidate in pair_rows if candidate['intervention'] != row['intervention']]
    fallback.sort(key=lambda candidate: (-float(candidate['evidence_score']), str(candidate['pair_id'])))
    return fallback[0], 'fallback'



def pick_negative_surrogate(
    row: dict[str, object],
    surrogate_rows: list[dict[str, object]],
    relevant_surrogates_by_intervention: dict[str, set[str]],
) -> tuple[dict[str, object], str]:
    relevant = relevant_surrogates_by_intervention[row['intervention']]
    same_category = [
        candidate
        for candidate in surrogate_rows
        if candidate['surrogate'] not in relevant and candidate['surrogate_category'] == row['surrogate_category']
    ]
    if same_category:
        same_category.sort(key=lambda candidate: (-float(candidate['mean_evidence_score']), str(candidate['surrogate_id'])))
        return same_category[0], 'same_category'

    same_hallmark = [
        candidate
        for candidate in surrogate_rows
        if candidate['surrogate'] not in relevant and row['hallmark'] in split_pipe(str(candidate['hallmarks']))
    ]
    if same_hallmark:
        same_hallmark.sort(key=lambda candidate: (-float(candidate['mean_evidence_score']), str(candidate['surrogate_id'])))
        return same_hallmark[0], 'same_hallmark'

    fallback = [candidate for candidate in surrogate_rows if candidate['surrogate'] not in relevant]
    fallback.sort(key=lambda candidate: (-float(candidate['mean_evidence_score']), str(candidate['surrogate_id'])))
    return fallback[0], 'fallback'


def build_triplets(
    pair_rows: list[dict[str, object]],
    surrogate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    surrogate_map = {row['surrogate']: row for row in surrogate_rows}
    relevant_surrogates_by_intervention = {
        intervention: {pair['surrogate'] for pair in pairs}
        for intervention, pairs in group_by(pair_rows, 'intervention').items()
    }
    results: list[dict[str, object]] = []
    for pair in pair_rows:
        anchor = (
            f"Recommend a surrogate biomarker for {pair['intervention']} with mechanism {pair['mechanism_hint']} "
            f"focusing on {pair['pathway']} and {pair['hallmark']}."
        )
        negative_pair, negative_pair_strategy = pick_negative_pair(pair, pair_rows)
        negative_surrogate, negative_surrogate_strategy = pick_negative_surrogate(
            pair,
            surrogate_rows,
            relevant_surrogates_by_intervention,
        )
        positive_surrogate = surrogate_map[pair['surrogate']]
        results.append(
            {
                'triplet_id': f"triplet_pair_{pair['pair_id']}",
                'split': pair['split'],
                'task_type': 'query_to_pair',
                'anchor_text': anchor,
                'positive_id': f"pair:{pair['pair_id']}",
                'positive_text': pair['embedding_text'],
                'negative_id': f"pair:{negative_pair['pair_id']}",
                'negative_text': negative_pair['embedding_text'],
                'negative_strategy': negative_pair_strategy,
                'target_intervention': pair['intervention'],
                'target_surrogate': pair['surrogate'],
            }
        )
        results.append(
            {
                'triplet_id': f"triplet_surrogate_{pair['pair_id']}",
                'split': pair['split'],
                'task_type': 'query_to_surrogate',
                'anchor_text': anchor,
                'positive_id': f"surrogate:{positive_surrogate['surrogate_id']}",
                'positive_text': positive_surrogate['embedding_text'],
                'negative_id': f"surrogate:{negative_surrogate['surrogate_id']}",
                'negative_text': negative_surrogate['embedding_text'],
                'negative_strategy': negative_surrogate_strategy,
                'target_intervention': pair['intervention'],
                'target_surrogate': pair['surrogate'],
            }
        )
    return results


def build_pair_examples(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for pair in pair_rows:
        query_text = (
            f"{pair['intervention']}-like intervention targeting {pair['mechanism_hint']}. "
            f"Would {pair['surrogate']} be a good surrogate endpoint?"
        )
        negative_pair, negative_strategy = pick_negative_pair(pair, pair_rows)
        results.append(
            {
                'example_id': f"example_pos_{pair['pair_id']}",
                'split': pair['split'],
                'query_text': query_text,
                'candidate_id': f"pair:{pair['pair_id']}",
                'candidate_text': pair['embedding_text'],
                'candidate_type': 'pair',
                'label': 1,
                'target_intervention': pair['intervention'],
                'target_surrogate': pair['surrogate'],
                'negative_strategy': '',
            }
        )
        results.append(
            {
                'example_id': f"example_neg_{pair['pair_id']}",
                'split': pair['split'],
                'query_text': query_text,
                'candidate_id': f"pair:{negative_pair['pair_id']}",
                'candidate_text': negative_pair['embedding_text'],
                'candidate_type': 'pair',
                'label': 0,
                'target_intervention': pair['intervention'],
                'target_surrogate': pair['surrogate'],
                'negative_strategy': negative_strategy,
            }
        )
    return results


def build_manifest(record_counts: dict[str, int]) -> list[dict[str, object]]:
    rows = [
        {
            'file_name': 'intervention_texts.csv',
            'record_count': record_counts['intervention_texts.csv'],
            'unit': 'one row per intervention',
            'purpose': 'embedding intervention descriptions',
            'recommended_stage': 'intervention encoder',
        },
        {
            'file_name': 'surrogate_texts.csv',
            'record_count': record_counts['surrogate_texts.csv'],
            'unit': 'one row per surrogate',
            'purpose': 'embedding surrogate descriptions',
            'recommended_stage': 'surrogate encoder',
        },
        {
            'file_name': 'biomarker_semantic_texts.csv',
            'record_count': record_counts['biomarker_semantic_texts.csv'],
            'unit': 'one row per biomarker semantic profile',
            'purpose': 'embedding biomarker semantics with hallmarks and MoA context',
            'recommended_stage': 'biomarker semantic encoder',
        },
        {
            'file_name': 'pair_texts.csv',
            'record_count': record_counts['pair_texts.csv'],
            'unit': 'one row per intervention-surrogate pair',
            'purpose': 'pair retrieval and reranking corpus',
            'recommended_stage': 'pair retrieval',
        },
        {
            'file_name': 'retrieval_corpus.csv',
            'record_count': record_counts['retrieval_corpus.csv'],
            'unit': 'one row per searchable document',
            'purpose': 'FAISS or nearest-neighbor indexing corpus',
            'recommended_stage': 'index build',
        },
        {
            'file_name': 'queries.csv',
            'record_count': record_counts['queries.csv'],
            'unit': 'one row per training/eval query',
            'purpose': 'embedding retrieval queries',
            'recommended_stage': 'training and evaluation',
        },
        {
            'file_name': 'query_qrels.csv',
            'record_count': record_counts['query_qrels.csv'],
            'unit': 'one row per query-document relevance label',
            'purpose': 'retrieval evaluation ground truth',
            'recommended_stage': 'evaluation',
        },
        {
            'file_name': 'contrastive_triplets.csv',
            'record_count': record_counts['contrastive_triplets.csv'],
            'unit': 'one row per contrastive triplet',
            'purpose': 'sentence-transformers or twin-tower fine-tuning',
            'recommended_stage': 'contrastive training',
        },
        {
            'file_name': 'pair_examples.csv',
            'record_count': record_counts['pair_examples.csv'],
            'unit': 'one row per labeled query-candidate example',
            'purpose': 'binary reranker or classifier training',
            'recommended_stage': 'reranking',
        },
    ]
    return rows


def build_stats(
    intervention_rows: list[dict[str, object]],
    surrogate_rows: list[dict[str, object]],
    biomarker_semantic_rows: list[dict[str, object]],
    pair_rows: list[dict[str, object]],
    queries: list[dict[str, object]],
    triplets: list[dict[str, object]],
    pair_examples: list[dict[str, object]],
) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    stats.append({'metric': 'intervention_count', 'value': len(intervention_rows)})
    stats.append({'metric': 'surrogate_count', 'value': len(surrogate_rows)})
    stats.append({'metric': 'biomarker_semantic_count', 'value': len(biomarker_semantic_rows)})
    stats.append({'metric': 'pair_count', 'value': len(pair_rows)})
    stats.append({'metric': 'query_count', 'value': len(queries)})
    stats.append({'metric': 'triplet_count', 'value': len(triplets)})
    stats.append({'metric': 'pair_example_count', 'value': len(pair_examples)})
    for split in ['train', 'val', 'test']:
        stats.append({'metric': f'interventions_{split}', 'value': sum(1 for row in intervention_rows if row['split'] == split)})
        stats.append({'metric': f'pairs_{split}', 'value': sum(1 for row in pair_rows if row['split'] == split)})
        stats.append({'metric': f'queries_{split}', 'value': sum(1 for row in queries if row['split'] == split)})
    return stats


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


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


def prepare_embedding_tables(
    curated_path: str | Path,
    demo_query_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    curated_rows = load_curated_rows(curated_path)
    intervention_rows = build_intervention_texts(curated_rows)
    surrogate_rows = build_surrogate_texts(curated_rows)
    biomarker_semantic_rows = build_biomarker_semantic_texts(curated_rows)
    pair_rows = build_pair_texts(curated_rows)
    retrieval_corpus = build_retrieval_corpus(intervention_rows, surrogate_rows, biomarker_semantic_rows, pair_rows)
    queries = build_queries(pair_rows, intervention_rows, demo_query_path)
    qrels = build_qrels(queries, pair_rows)
    triplets = build_triplets(pair_rows, surrogate_rows)
    pair_examples = build_pair_examples(pair_rows)

    output = Path(output_dir)
    write_csv(output / 'intervention_texts.csv', intervention_rows)
    write_csv(output / 'surrogate_texts.csv', surrogate_rows)
    write_csv(output / 'biomarker_semantic_texts.csv', biomarker_semantic_rows)
    write_csv(output / 'pair_texts.csv', pair_rows)
    write_csv(output / 'retrieval_corpus.csv', retrieval_corpus)
    write_csv(output / 'queries.csv', queries)
    write_csv(output / 'query_qrels.csv', qrels)
    write_csv(output / 'contrastive_triplets.csv', triplets)
    write_csv(output / 'pair_examples.csv', pair_examples)

    record_counts = {
        'intervention_texts.csv': len(intervention_rows),
        'surrogate_texts.csv': len(surrogate_rows),
        'biomarker_semantic_texts.csv': len(biomarker_semantic_rows),
        'pair_texts.csv': len(pair_rows),
        'retrieval_corpus.csv': len(retrieval_corpus),
        'queries.csv': len(queries),
        'query_qrels.csv': len(qrels),
        'contrastive_triplets.csv': len(triplets),
        'pair_examples.csv': len(pair_examples),
    }
    write_csv(output / 'embedding_manifest.csv', build_manifest(record_counts))
    write_csv(output / 'embedding_stats.csv', build_stats(intervention_rows, surrogate_rows, biomarker_semantic_rows, pair_rows, queries, triplets, pair_examples))
    return record_counts


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Prepare embedding-ready training and retrieval tables.')
    parser.add_argument('--curated-path', default=str(root / 'data' / 'processed' / 'curated_biomarkers.csv'))
    parser.add_argument('--demo-query-path', default=str(root / 'data' / 'processed' / 'demo_queries.csv'))
    parser.add_argument('--output-dir', default=str(root / 'data' / 'embedding'))
    args = parser.parse_args(argv)
    counts = prepare_embedding_tables(args.curated_path, args.demo_query_path, args.output_dir)
    for file_name, record_count in counts.items():
        print(f'{file_name}: {record_count}')
    return 0

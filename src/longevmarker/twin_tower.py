from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import random
import re
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

try:
    import selfies as sf
except Exception:  # pragma: no cover
    sf = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\+\-\./]+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


MOLECULAR_ENCODER_PRESETS = {
    'chemberta': {
        'label': 'ChemBERTa',
        'model_name': 'seyonec/ChemBERTa-zinc-base-v1',
        'representation': 'smiles',
        'trust_remote_code': False,
    },
    'molformer': {
        'label': 'MolFormer',
        'model_name': 'ibm-research/MoLFormer-XL-both-10pct',
        'representation': 'smiles',
        'trust_remote_code': True,
    },
    'selformer': {
        'label': 'SELFormer',
        'model_name': 'HUBioDataLab/SELFormer',
        'representation': 'selfies',
        'trust_remote_code': False,
    },
}


INTERVENTION_PROTOTYPES = {
    'Dasatinib': ['Dasatinib + Quercetin'],
    'Quercetin': ['Dasatinib + Quercetin'],
    'Glp-1 Receptor Agonists': ['Semaglutide', 'Liraglutide'],
    'Dual Glp-1/Gip Receptor Agonists': ['Tirzepatide', 'Semaglutide'],
    'Sglt2 Inhibitors': ['Empagliflozin', 'Dapagliflozin'],
    'Sglt-2 Inhibitors': ['Empagliflozin', 'Dapagliflozin'],
    'Sodium-Glucose Cotransporter-2 Inhibitors': ['Empagliflozin', 'Dapagliflozin'],
    'Mtor Inhibitors': ['Rapamycin'],
    'Rapamycin Analogs': ['Rapamycin'],
    'Nad+ Precursors': ['Nicotinamide Riboside', 'Nmn'],
    'Nad+ Therapeutics': ['Nicotinamide Riboside', 'Nmn'],
    'Senolytic Strategies': ['Dasatinib + Quercetin', 'Fisetin'],
    'Senolytic Therapies': ['Dasatinib + Quercetin', 'Fisetin'],
    'Senolytics': ['Dasatinib + Quercetin', 'Fisetin'],
}


INTERVENTION_ALIASES = {
    'Nmn': 'NMN',
}


SUPPLEMENTAL_STRUCTURES = {
    'Curcumin': ['COC1=CC(=CC(=C1O)OC)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC'],
    'N-Acetylcysteine': ['CC(=O)N[C@@H](CS)C(=O)O'],
    'Urolithin A': ['O=C1OC2=CC(O)=CC(O)=C2C=C1'],
    'Tirzepatide': ['CC(C)CC(NC(=O)CNC(=O)C(CC(=O)O)NC(=O)C(CCCNC(=N)N)NC(=O)CNC(=O)C(CC1=CC=CC=C1)NC(=O)C(CO)NC(=O)C(CC(=O)O)NC(=O)CNC(=O)C(CCCNC(=N)N)NC(=O)C(C)NC(=O)C(CC2=CNC3=CC=CC=C23)NC(=O)C(C(C)C)NC(=O)C(C(C)C)NC(=O)CNC(=O)C(C(C)O)NC(=O)CNC(=O)CNC(=O)C(C(C)CC)NC(=O)C(C)NC(=O)C(C)N)C(=O)NCC(=O)N'],
    'Coenzyme Q10': ['CC1=C(C(=O)C(=C(C1=O)OC)OC)C(/C=C(/C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)=O'],
    'Mitoq': ['C[N+](C)(C)CCCCC(=O)C1=CC(=O)C(=C(C1=O)OC)OC'],
    'Corticosterone': ['CC12CCC(=O)C=C1CCC3C2CCC4(C3CC(C4)O)C(=O)CO'],
    'Betulinic Acid': ['CC1(C2CCC3(C(C2(CCC1O)C)C)CCC4=CC(=O)CCC34C)C(=O)O'],
    'Dapagliflozin': ['CC1COC(O1)COC2=CC=C(C=C2)CC3=CC(=CC=C3)Cl'],
    'Lanatoside C': ['CC1(C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)OC3C(C(C(C(O3)CO)O)O)O)O)O)O)CCC4C1(CC(C5(C4CCC6(C5CC(C(C6)O)OC7C(C(C(C(O7)CO)O)O)O)O)C)O)C'],
    'Bleomycin': ['CC1=C(C(=O)N)N=C(N1)NCC2OC(O)C(O)C(O)C2O'],
    'Insulin': ['insulin'],
    'Immune Checkpoint Inhibitors': ['immune checkpoint inhibitor'],
    'Microglial Depletion-Repopulation Paradigms': ['microglial depletion repopulation'],
    'Dpp-4 Inhibitors': ['dpp4 inhibitor'],
    'Ras Inhibitors': ['ras inhibitor'],
}


@dataclass
class MolecularEncoderConfig:
    preset: str
    model_name: str
    representation: str
    trust_remote_code: bool
    max_length: int | None = None
    device: str = 'cpu'
    unfreeze_last_n_layers: int = 1


@dataclass
class TrainConfig:
    curated_csv: str
    compound_registry_csv: str
    compound_components_csv: str
    seed_interventions_csv: str
    output_dir: str
    molecular_encoder: MolecularEncoderConfig
    query_text_max_features: int = 2048
    candidate_text_max_features: int = 4096
    hidden_dim: int = 256
    embedding_dim: int = 128
    dropout: float = 0.1
    triplet_margin: float = 0.2
    bce_weight: float = 1.0
    triplet_weight: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 18
    early_stopping_patience: int = 4
    top_k_metrics: int = 10
    predict_top_k: int = 3
    seed: int = 42
    num_folds: int = 5


@dataclass
class LoadedModelBundle:
    model: 'ChemicalBiomarkerTwinTower'
    query_vectorizer: 'SimpleTfidfVectorizer'
    candidate_vectorizer: 'SimpleTfidfVectorizer'
    query_rows: list[dict[str, str]]
    candidate_rows: list[dict[str, str]]
    candidate_text_features: np.ndarray
    class_to_biomarker: dict[str, str]
    biomarker_to_class: dict[str, int]
    config: dict[str, object]


@dataclass
class RuntimeDataset:
    query_rows: list[dict[str, str]]
    candidate_rows: list[dict[str, str]]
    relation_rows: list[dict[str, object]]
    triplet_rows: list[dict[str, object]]
    fold_assignments: list[dict[str, object]]
    structure_resolution_rows: list[dict[str, object]]


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


def write_json(path: str | Path, payload: dict[str, object] | list[object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_ngrams(tokens: list[str]) -> list[str]:
    return tokens + [f'{tokens[i]}__{tokens[i + 1]}' for i in range(len(tokens) - 1)]


def slugify(text: str) -> str:
    slug = NON_ALNUM_PATTERN.sub('_', text.lower()).strip('_')
    return slug or 'item'


def safe_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value in (None, ''):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unique_join(values: Iterable[str], sep: str = '|') -> str:
    seen: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return sep.join(seen)


class SimpleTfidfVectorizer:
    def __init__(self, max_features: int = 4096) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.dim = 0

    def fit(self, texts: Iterable[str]) -> None:
        doc_freq: Counter[str] = Counter()
        texts = list(texts)
        for text in texts:
            doc_freq.update(set(build_ngrams(tokenize(text))))
        ordered = sorted(doc_freq.items(), key=lambda item: (-item[1], item[0]))[: self.max_features]
        self.vocab = {term: index for index, (term, _) in enumerate(ordered)}
        n_docs = max(len(texts), 1)
        self.idf = np.ones(len(self.vocab), dtype=np.float32)
        for term, index in self.vocab.items():
            self.idf[index] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        self.dim = len(self.vocab)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError('Vectorizer must be fitted before encoding')
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row_index, text in enumerate(texts):
            counts: Counter[int] = Counter()
            for term in build_ngrams(tokenize(text)):
                index = self.vocab.get(term)
                if index is not None:
                    counts[index] += 1
            if not counts:
                continue
            for index, count in counts.items():
                matrix[row_index, index] = (1.0 + math.log(count)) * float(self.idf[index])
        return normalize_rows(matrix)

    def save(self, path: str | Path) -> None:
        payload = {
            'max_features': self.max_features,
            'vocab': self.vocab,
            'idf': self.idf.tolist() if self.idf is not None else [],
        }
        with Path(path).open('wb') as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> 'SimpleTfidfVectorizer':
        with Path(path).open('rb') as handle:
            payload = pickle.load(handle)
        vectorizer = cls(max_features=int(payload['max_features']))
        vectorizer.vocab = dict(payload['vocab'])
        vectorizer.idf = np.asarray(payload['idf'], dtype=np.float32)
        vectorizer.dim = len(vectorizer.vocab)
        return vectorizer


def build_registry_index(compound_registry_rows: list[dict[str, str]], component_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    registry_by_name = {row['intervention']: row for row in compound_registry_rows}
    component_by_name = {row['component_name']: row for row in component_rows}
    return registry_by_name, component_by_name


def resolve_intervention_sequences(
    intervention: str,
    registry_by_name: dict[str, dict[str, str]],
    component_by_name: dict[str, dict[str, str]],
    seen: set[str] | None = None,
) -> tuple[list[str], str, str]:
    seen = seen or set()
    normalized_name = INTERVENTION_ALIASES.get(intervention, intervention)
    if normalized_name in seen:
        return [intervention], 'fallback_cycle', intervention
    seen.add(normalized_name)

    registry_row = registry_by_name.get(normalized_name)
    if registry_row:
        if registry_row.get('component_count', '1') != '1' and registry_row.get('component_canonical_smiles'):
            sequences = [item for item in registry_row['component_canonical_smiles'].split('|') if item]
            return sequences or [registry_row.get('canonical_smiles', normalized_name)], 'registry_components', normalized_name
        if registry_row.get('canonical_smiles'):
            return [registry_row['canonical_smiles']], 'registry_exact', normalized_name

    component_row = component_by_name.get(normalized_name)
    if component_row and component_row.get('canonical_smiles'):
        return [component_row['canonical_smiles']], 'component_exact', normalized_name

    if normalized_name in SUPPLEMENTAL_STRUCTURES:
        return SUPPLEMENTAL_STRUCTURES[normalized_name], 'supplemental_structure', normalized_name

    prototypes = INTERVENTION_PROTOTYPES.get(normalized_name)
    if prototypes:
        sequences: list[str] = []
        sources: list[str] = []
        for prototype in prototypes:
            resolved, _, source = resolve_intervention_sequences(prototype, registry_by_name, component_by_name, seen=set(seen))
            if resolved:
                sequences.extend(resolved)
                sources.append(source)
        if sequences:
            return sequences, 'prototype_average', '|'.join(sources)

    return [normalized_name], 'text_fallback', normalized_name


def build_fold_assignments(interventions: list[str], num_folds: int, seed: int) -> list[dict[str, object]]:
    shuffled = list(sorted(interventions))
    random.Random(seed).shuffle(shuffled)
    folds = [shuffled[index::num_folds] for index in range(num_folds)]
    rows: list[dict[str, object]] = []
    for fold_index, members in enumerate(folds):
        for intervention in members:
            rows.append({'intervention': intervention, 'fold': fold_index})
    return rows


def build_runtime_dataset(
    curated_csv: str | Path,
    compound_registry_csv: str | Path,
    compound_components_csv: str | Path,
    seed_interventions_csv: str | Path,
    num_folds: int,
    seed: int,
) -> RuntimeDataset:
    curated_rows = load_csv(curated_csv)
    compound_registry_rows = load_csv(compound_registry_csv)
    component_rows = load_csv(compound_components_csv)
    seed_rows = load_csv(seed_interventions_csv)
    seed_map = {row['intervention']: row for row in seed_rows}
    registry_by_name, component_by_name = build_registry_index(compound_registry_rows, component_rows)

    surrogate_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    intervention_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in curated_rows:
        surrogate_groups[row['surrogate']].append(row)
        intervention_groups[row['intervention']].append(row)

    candidate_rows: list[dict[str, str]] = []
    candidate_id_map: dict[str, str] = {}
    for surrogate, rows in sorted(surrogate_groups.items()):
        surrogate_id = slugify(surrogate)
        candidate_id = f'surrogate:{surrogate_id}'
        candidate_id_map[surrogate] = candidate_id
        mean_evidence = np.mean([safe_float(row.get('evidence_score')) for row in rows])
        mean_confidence = np.mean([safe_float(row.get('surrogate_confidence')) for row in rows])
        mean_lit = np.mean([safe_float(row.get('literature_support')) for row in rows])
        top_snippets = unique_join([row.get('sample_snippet', '')[:160] for row in rows[:2]], sep=' | ')
        embedding_text = (
            f"Biomarker semantic profile: {surrogate}. "
            f"Category: {unique_join([row['surrogate_category'] for row in rows], sep=', ')}. "
            f"Pathways: {unique_join([row['pathway'] for row in rows], sep=', ')}. "
            f"Aging hallmarks: {unique_join([row['hallmark'] for row in rows], sep=', ')}. "
            f"Linked interventions: {unique_join([row['intervention'] for row in rows], sep=', ')}. "
            f"Mechanism contexts: {unique_join([row['mechanism_hint'] for row in rows], sep=', ')}. "
            f"Evidence score: {mean_evidence:.3f}. Surrogate confidence: {mean_confidence:.3f}. Literature support: {mean_lit:.3f}. "
            f"Example context: {top_snippets}."
        )
        candidate_rows.append(
            {
                'candidate_id': candidate_id,
                'surrogate_id': surrogate_id,
                'surrogate': surrogate,
                'surrogate_category': unique_join([row['surrogate_category'] for row in rows], sep='|'),
                'pathways': unique_join([row['pathway'] for row in rows], sep='|'),
                'hallmarks': unique_join([row['hallmark'] for row in rows], sep='|'),
                'linked_interventions': unique_join([row['intervention'] for row in rows], sep='|'),
                'mechanism_hints': unique_join([row['mechanism_hint'] for row in rows], sep='|'),
                'source_pair_count': str(len(rows)),
                'mean_evidence_score': f'{mean_evidence:.6f}',
                'mean_surrogate_confidence': f'{mean_confidence:.6f}',
                'mean_literature_support': f'{mean_lit:.6f}',
                'embedding_text': embedding_text,
            }
        )

    query_rows: list[dict[str, str]] = []
    structure_resolution_rows: list[dict[str, object]] = []
    for intervention, rows in sorted(intervention_groups.items()):
        sequences, source_type, source_detail = resolve_intervention_sequences(intervention, registry_by_name, component_by_name)
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                safe_float(row.get('evidence_score')) + safe_float(row.get('surrogate_confidence')) + safe_float(row.get('literature_support')),
                row['surrogate'],
            ),
            reverse=True,
        )
        top_surrogates = '|'.join(row['surrogate'] for row in sorted_rows[:5])
        mechanism_hints = unique_join([seed_map.get(intervention, {}).get('mechanism_hint', '')] + [row['mechanism_hint'] for row in rows], sep='|')
        query_text = (
            f"Intervention profile: {intervention}. "
            f"Mechanism hints: {mechanism_hints.replace('|', ', ')}. "
            f"Observed biomarker categories: {unique_join([row['surrogate_category'] for row in rows], sep=', ')}. "
            f"Observed pathways: {unique_join([row['pathway'] for row in rows], sep=', ')}. "
            f"Observed hallmarks: {unique_join([row['hallmark'] for row in rows], sep=', ')}. "
            f"Top linked biomarkers: {top_surrogates.replace('|', ', ')}."
        )
        query_id = f"intervention:{slugify(intervention)}"
        query_rows.append(
            {
                'query_id': query_id,
                'intervention': intervention,
                'mechanism_hint': mechanism_hints,
                'query_text': query_text,
                'structure_inputs': json.dumps(sequences),
                'structure_source': source_type,
                'structure_source_detail': source_detail,
                'structure_count': str(len(sequences)),
                'top_surrogates': top_surrogates,
            }
        )
        structure_resolution_rows.append(
            {
                'intervention': intervention,
                'structure_source': source_type,
                'structure_source_detail': source_detail,
                'structure_count': len(sequences),
                'structure_inputs_preview': ' | '.join(sequences[:2]),
            }
        )

    raw_scores: list[float] = []
    relation_seed_rows: list[dict[str, object]] = []
    for row in curated_rows:
        combined_score = (
            0.45 * safe_float(row.get('evidence_score'))
            + 0.35 * safe_float(row.get('surrogate_confidence'))
            + 0.20 * safe_float(row.get('literature_support'))
        )
        raw_scores.append(combined_score)
        relation_seed_rows.append(
            {
                'query_id': f"intervention:{slugify(row['intervention'])}",
                'candidate_id': candidate_id_map[row['surrogate']],
                'intervention': row['intervention'],
                'surrogate': row['surrogate'],
                'score': combined_score,
            }
        )
    low_threshold = float(np.quantile(raw_scores, 0.33)) if raw_scores else 0.33
    high_threshold = float(np.quantile(raw_scores, 0.66)) if raw_scores else 0.66

    relation_index: dict[tuple[str, str], dict[str, object]] = {}
    for row in relation_seed_rows:
        relevance = 3 if row['score'] >= high_threshold else 2 if row['score'] >= low_threshold else 1
        key = (str(row['query_id']), str(row['candidate_id']))
        existing = relation_index.get(key)
        payload = {
            'query_id': row['query_id'],
            'candidate_id': row['candidate_id'],
            'intervention': row['intervention'],
            'surrogate': row['surrogate'],
            'relevance': relevance,
            'label_weight': round(1.0 + float(row['score']), 6),
        }
        if existing is None or int(payload['relevance']) > int(existing['relevance']):
            relation_index[key] = payload
    relation_rows = list(relation_index.values())

    candidate_meta = {row['candidate_id']: row for row in candidate_rows}
    positives_by_query: dict[str, list[str]] = defaultdict(list)
    for row in relation_rows:
        positives_by_query[str(row['query_id'])].append(str(row['candidate_id']))
    all_candidate_ids = [row['candidate_id'] for row in candidate_rows]
    triplet_rows: list[dict[str, object]] = []
    for query_row in query_rows:
        query_id = query_row['query_id']
        positives = positives_by_query.get(query_id, [])
        positive_set = set(positives)
        for positive_id in positives:
            positive_meta = candidate_meta[positive_id]
            positive_categories = set(filter(None, positive_meta.get('surrogate_category', '').split('|')))
            positive_hallmarks = set(filter(None, positive_meta.get('hallmarks', '').split('|')))
            hard_negative = None
            for candidate_id in all_candidate_ids:
                if candidate_id in positive_set:
                    continue
                candidate = candidate_meta[candidate_id]
                categories = set(filter(None, candidate.get('surrogate_category', '').split('|')))
                hallmarks = set(filter(None, candidate.get('hallmarks', '').split('|')))
                if positive_categories & categories or positive_hallmarks & hallmarks:
                    hard_negative = candidate_id
                    break
            if hard_negative is None:
                for candidate_id in all_candidate_ids:
                    if candidate_id not in positive_set:
                        hard_negative = candidate_id
                        break
            if hard_negative is None:
                continue
            triplet_rows.append(
                {
                    'query_id': query_id,
                    'intervention': query_row['intervention'],
                    'positive_id': positive_id,
                    'negative_id': hard_negative,
                    'task_type': 'chemical_to_biomarker',
                }
            )

    fold_assignments = build_fold_assignments([row['intervention'] for row in query_rows], num_folds=num_folds, seed=seed)
    return RuntimeDataset(
        query_rows=query_rows,
        candidate_rows=candidate_rows,
        relation_rows=relation_rows,
        triplet_rows=triplet_rows,
        fold_assignments=fold_assignments,
        structure_resolution_rows=structure_resolution_rows,
    )


class MolecularEncoder(nn.Module):
    def __init__(self, config: MolecularEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.output_dim = int(getattr(self.model.config, 'hidden_size', 768))
        self.effective_max_length = self._resolve_max_length(config.max_length)
        self._freeze_except_last_layers(config.unfreeze_last_n_layers)
        self.to(config.device)

    def _resolve_max_length(self, requested: int | None) -> int:
        model_max = getattr(self.tokenizer, 'model_max_length', None)
        if not isinstance(model_max, int) or model_max <= 0 or model_max > 100000:
            model_max = 512
        if requested is None:
            return model_max
        return min(requested, model_max)

    def _iter_transformer_blocks(self) -> list[nn.Module]:
        candidates = [
            getattr(getattr(self.model, 'encoder', None), 'layer', None),
            getattr(getattr(getattr(self.model, 'roberta', None), 'encoder', None), 'layer', None),
            getattr(getattr(getattr(self.model, 'bert', None), 'encoder', None), 'layer', None),
            getattr(getattr(self.model, 'transformer', None), 'layer', None),
            getattr(self.model, 'layers', None),
            getattr(self.model, 'blocks', None),
        ]
        for candidate in candidates:
            if candidate is not None:
                return list(candidate)
        return []

    def _freeze_except_last_layers(self, unfreeze_last_n_layers: int) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        blocks = self._iter_transformer_blocks()
        if blocks and unfreeze_last_n_layers > 0:
            for block in blocks[-unfreeze_last_n_layers:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
        for attr_name in ('pooler', 'ln_f', 'final_layer_norm', 'layernorm'):
            module = getattr(self.model, attr_name, None)
            if module is not None and hasattr(module, 'parameters'):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _prepare_sequence(self, sequence: str) -> str:
        sequence = sequence.strip()
        if self.config.representation != 'selfies':
            return sequence
        if sf is None:
            return sequence
        if sequence.startswith('[') and sequence.endswith(']'):
            return sequence
        try:
            return sf.encoder(sequence)
        except Exception:
            return sequence

    def encode_sequence_groups(self, sequence_groups: list[list[str]]) -> torch.Tensor:
        flat_sequences: list[str] = []
        owners: list[int] = []
        for owner, sequences in enumerate(sequence_groups):
            group = sequences or ['']
            for sequence in group:
                flat_sequences.append(self._prepare_sequence(sequence))
                owners.append(owner)
        if not flat_sequences:
            return torch.zeros((0, self.output_dim), device=self.config.device)
        inputs = self.tokenizer(
            flat_sequences,
            padding=True,
            truncation=True,
            max_length=self.effective_max_length,
            return_tensors='pt',
        )
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        pooled = self._mean_pool(hidden_states, inputs['attention_mask'])
        pooled = F.normalize(pooled, dim=-1)
        grouped = torch.zeros((len(sequence_groups), pooled.shape[-1]), dtype=pooled.dtype, device=pooled.device)
        counts = torch.zeros((len(sequence_groups), 1), dtype=pooled.dtype, device=pooled.device)
        for index, owner in enumerate(owners):
            grouped[owner] += pooled[index]
            counts[owner] += 1.0
        grouped = grouped / counts.clamp(min=1.0)
        return F.normalize(grouped, dim=-1)


class ProjectionTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


class ChemicalBiomarkerTwinTower(nn.Module):
    def __init__(
        self,
        molecular_encoder: MolecularEncoder,
        query_text_dim: int,
        candidate_text_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.molecular_encoder = molecular_encoder
        self.query_tower = ProjectionTower(molecular_encoder.output_dim + query_text_dim, hidden_dim, embedding_dim, dropout)
        self.candidate_tower = ProjectionTower(candidate_text_dim, hidden_dim, embedding_dim, dropout)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode_queries(self, sequence_groups: list[list[str]], query_text_features: torch.Tensor) -> torch.Tensor:
        molecular_features = self.molecular_encoder.encode_sequence_groups(sequence_groups)
        combined = torch.cat([molecular_features, query_text_features], dim=-1)
        return self.query_tower(combined)

    def encode_candidates(self, candidate_text_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_tower(candidate_text_features)

    def score_matrix(self, sequence_groups: list[list[str]], query_text_features: torch.Tensor, candidate_text_features: torch.Tensor) -> torch.Tensor:
        query_embeddings = self.encode_queries(sequence_groups, query_text_features)
        candidate_embeddings = self.encode_candidates(candidate_text_features)
        temperature = torch.exp(self.log_temperature).clamp_min(1e-4)
        return query_embeddings @ candidate_embeddings.T / temperature

    def score_triplets(
        self,
        sequence_groups: list[list[str]],
        query_text_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = self.encode_queries(sequence_groups, query_text_features)
        positive_embeddings = self.encode_candidates(positive_features)
        negative_embeddings = self.encode_candidates(negative_features)
        temperature = torch.exp(self.log_temperature).clamp_min(1e-4)
        positive_scores = (query_embeddings * positive_embeddings).sum(dim=-1) / temperature
        negative_scores = (query_embeddings * negative_embeddings).sum(dim=-1) / temperature
        return positive_scores, negative_scores


def build_qrels_map(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        qrels[str(row['query_id'])][str(row['candidate_id'])] = int(row['relevance'])
    return qrels


def build_weight_map(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        weights[str(row['query_id'])][str(row['candidate_id'])] = float(row['label_weight'])
    return weights


def assign_splits(query_rows: list[dict[str, str]], fold_assignments: list[dict[str, object]], fold_index: int, num_folds: int) -> list[dict[str, str]]:
    fold_map = {row['intervention']: int(row['fold']) for row in fold_assignments}
    test_fold = fold_index
    val_fold = (fold_index + 1) % num_folds
    assigned: list[dict[str, str]] = []
    for row in query_rows:
        fold = fold_map[row['intervention']]
        split = 'train'
        if fold == test_fold:
            split = 'test'
        elif fold == val_fold:
            split = 'val'
        updated = dict(row)
        updated['split'] = split
        assigned.append(updated)
    return assigned


def build_query_target_matrices(
    query_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    qrels_map: dict[str, dict[str, int]],
    weight_map: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_index = {row['candidate_id']: idx for idx, row in enumerate(candidate_rows)}
    targets = np.zeros((len(query_rows), len(candidate_rows)), dtype=np.float32)
    weights = np.ones((len(query_rows), len(candidate_rows)), dtype=np.float32)
    for query_idx, row in enumerate(query_rows):
        for candidate_id, relevance in qrels_map.get(row['query_id'], {}).items():
            if candidate_id not in candidate_index:
                continue
            column = candidate_index[candidate_id]
            targets[query_idx, column] = 1.0
            weights[query_idx, column] = weight_map.get(row['query_id'], {}).get(candidate_id, float(relevance))
    return targets, weights


def prepare_triplet_rows(
    triplet_rows: list[dict[str, object]],
    split_queries: list[dict[str, str]],
    candidate_index: dict[str, int],
) -> list[dict[str, object]]:
    allowed_query_ids = {row['query_id'] for row in split_queries}
    query_index = {row['query_id']: idx for idx, row in enumerate(split_queries)}
    prepared: list[dict[str, object]] = []
    for row in triplet_rows:
        if row['query_id'] not in allowed_query_ids:
            continue
        if row['positive_id'] not in candidate_index or row['negative_id'] not in candidate_index:
            continue
        prepared.append(
            {
                'query_id': row['query_id'],
                'query_index': query_index[str(row['query_id'])],
                'positive_index': candidate_index[str(row['positive_id'])],
                'negative_index': candidate_index[str(row['negative_id'])],
                'intervention': row['intervention'],
            }
        )
    return prepared


def ranking_metrics(
    query_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    similarity: np.ndarray,
    qrels_map: dict[str, dict[str, int]],
    top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rankings: list[dict[str, object]] = []
    metric_buckets: dict[str, list[float]] = defaultdict(list)
    candidate_ids = [row['candidate_id'] for row in candidate_rows]
    for query_index, query_row in enumerate(query_rows):
        query_id = query_row['query_id']
        qrels = qrels_map.get(query_id, {})
        scores = similarity[query_index]
        ranked_indices = np.argsort(-scores)[:top_k]
        reciprocal_rank = 0.0
        hits5 = 0.0
        dcg = 0.0
        ideal_relevances = sorted(qrels.values(), reverse=True)
        idcg = sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevances[:top_k])) or 1.0
        for rank, candidate_index in enumerate(ranked_indices, start=1):
            candidate_id = candidate_ids[candidate_index]
            relevance = qrels.get(candidate_id, 0)
            if reciprocal_rank == 0.0 and relevance > 0:
                reciprocal_rank = 1.0 / rank
            if rank <= 5 and relevance > 0:
                hits5 = 1.0
            dcg += (2 ** relevance - 1) / math.log2(rank + 1)
            rankings.append(
                {
                    'query_id': query_id,
                    'intervention': query_row['intervention'],
                    'split': query_row.get('split', ''),
                    'rank': rank,
                    'candidate_id': candidate_id,
                    'surrogate': candidate_rows[candidate_index]['surrogate'],
                    'score': round(float(scores[candidate_index]), 6),
                    'relevance': relevance,
                }
            )
        metric_buckets['mrr@10'].append(reciprocal_rank)
        metric_buckets['recall@5'].append(hits5)
        metric_buckets['ndcg@10'].append(dcg / idcg)
    metrics = [{'metric': metric, 'value': round(float(np.mean(values)) if values else 0.0, 6)} for metric, values in metric_buckets.items()]
    return rankings, metrics


def choose_metric(metrics: list[dict[str, object]], metric_name: str = 'mrr@10') -> float:
    for row in metrics:
        if row['metric'] == metric_name:
            return float(row['value'])
    return 0.0


def parse_structure_inputs(query_rows: list[dict[str, str]]) -> list[list[str]]:
    parsed: list[list[str]] = []
    for row in query_rows:
        value = row.get('structure_inputs', '[]')
        try:
            payload = json.loads(value)
            parsed.append([str(item) for item in payload])
        except json.JSONDecodeError:
            parsed.append([value])
    return parsed


def to_tensor(matrix: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def evaluate_model(
    model: ChemicalBiomarkerTwinTower,
    query_rows: list[dict[str, str]],
    query_text_features: np.ndarray,
    candidate_rows: list[dict[str, str]],
    candidate_text_features: np.ndarray,
    qrels_map: dict[str, dict[str, int]],
    top_k: int,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not query_rows:
        return [], [
            {'metric': 'mrr@10', 'value': 0.0},
            {'metric': 'recall@5', 'value': 0.0},
            {'metric': 'ndcg@10', 'value': 0.0},
        ]
    sequence_groups = parse_structure_inputs(query_rows)
    model.eval()
    with torch.no_grad():
        scores = model.score_matrix(
            sequence_groups,
            to_tensor(query_text_features, device),
            to_tensor(candidate_text_features, device),
        ).detach().cpu().numpy()
    return ranking_metrics(query_rows, candidate_rows, scores, qrels_map, top_k)


def save_config(path: str | Path, config: TrainConfig) -> None:
    payload = {
        'curated_csv': config.curated_csv,
        'compound_registry_csv': config.compound_registry_csv,
        'compound_components_csv': config.compound_components_csv,
        'seed_interventions_csv': config.seed_interventions_csv,
        'output_dir': config.output_dir,
        'query_text_max_features': config.query_text_max_features,
        'candidate_text_max_features': config.candidate_text_max_features,
        'hidden_dim': config.hidden_dim,
        'embedding_dim': config.embedding_dim,
        'dropout': config.dropout,
        'triplet_margin': config.triplet_margin,
        'bce_weight': config.bce_weight,
        'triplet_weight': config.triplet_weight,
        'lr': config.lr,
        'weight_decay': config.weight_decay,
        'epochs': config.epochs,
        'early_stopping_patience': config.early_stopping_patience,
        'top_k_metrics': config.top_k_metrics,
        'predict_top_k': config.predict_top_k,
        'seed': config.seed,
        'num_folds': config.num_folds,
        'molecular_encoder': {
            'preset': config.molecular_encoder.preset,
            'model_name': config.molecular_encoder.model_name,
            'representation': config.molecular_encoder.representation,
            'trust_remote_code': config.molecular_encoder.trust_remote_code,
            'max_length': config.molecular_encoder.max_length,
            'device': config.molecular_encoder.device,
            'unfreeze_last_n_layers': config.molecular_encoder.unfreeze_last_n_layers,
        },
    }
    write_json(path, payload)


def load_training_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_class_mapping(candidate_rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, int]]:
    class_to_biomarker = {str(index): row['surrogate'] for index, row in enumerate(candidate_rows)}
    biomarker_to_class = {row['surrogate']: index for index, row in enumerate(candidate_rows)}
    return class_to_biomarker, biomarker_to_class


def fit_vectorizers(query_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]], query_text_max_features: int, candidate_text_max_features: int) -> tuple[SimpleTfidfVectorizer, SimpleTfidfVectorizer, np.ndarray, np.ndarray]:
    query_vectorizer = SimpleTfidfVectorizer(max_features=query_text_max_features)
    candidate_vectorizer = SimpleTfidfVectorizer(max_features=candidate_text_max_features)
    query_texts = [row['query_text'] for row in query_rows]
    candidate_texts = [row['embedding_text'] for row in candidate_rows]
    query_vectorizer.fit(query_texts)
    candidate_vectorizer.fit(candidate_texts)
    return (
        query_vectorizer,
        candidate_vectorizer,
        query_vectorizer.encode(query_texts),
        candidate_vectorizer.encode(candidate_texts),
    )


def train_single_split(
    query_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    relation_rows: list[dict[str, object]],
    triplet_rows: list[dict[str, object]],
    config: TrainConfig,
    output_dir: Path,
) -> dict[str, object]:
    qrels_map = build_qrels_map(relation_rows)
    weight_map = build_weight_map(relation_rows)
    query_vectorizer, candidate_vectorizer, query_text_features, candidate_text_features = fit_vectorizers(
        query_rows,
        candidate_rows,
        config.query_text_max_features,
        config.candidate_text_max_features,
    )
    candidate_index = {row['candidate_id']: idx for idx, row in enumerate(candidate_rows)}
    class_to_biomarker, biomarker_to_class = build_class_mapping(candidate_rows)

    train_queries = [row for row in query_rows if row['split'] == 'train']
    val_queries = [row for row in query_rows if row['split'] == 'val']
    test_queries = [row for row in query_rows if row['split'] == 'test']

    train_targets, train_weights = build_query_target_matrices(train_queries, candidate_rows, qrels_map, weight_map)
    train_query_positions = [query_rows.index(row) for row in train_queries]
    val_query_positions = [query_rows.index(row) for row in val_queries]
    test_query_positions = [query_rows.index(row) for row in test_queries]

    model = ChemicalBiomarkerTwinTower(
        molecular_encoder=MolecularEncoder(config.molecular_encoder),
        query_text_dim=query_text_features.shape[1],
        candidate_text_dim=candidate_text_features.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(config.molecular_encoder.device)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    candidate_tensor = to_tensor(candidate_text_features, config.molecular_encoder.device)
    train_query_text_tensor = to_tensor(query_text_features[train_query_positions], config.molecular_encoder.device)
    train_targets_tensor = to_tensor(train_targets, config.molecular_encoder.device)
    train_weights_tensor = to_tensor(train_weights, config.molecular_encoder.device)
    train_sequences = parse_structure_inputs(train_queries)

    positive_count = max(float(train_targets.sum()), 1.0)
    negative_count = max(float(train_targets.size - train_targets.sum()), 1.0)
    pos_weight_value = negative_count / positive_count
    pos_weight = torch.full((candidate_text_features.shape[0],), pos_weight_value, dtype=torch.float32, device=config.molecular_encoder.device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_mrr = -1.0
    best_epoch = 0
    patience = 0
    history_rows: list[dict[str, object]] = []

    train_triplets = prepare_triplet_rows(triplet_rows, train_queries, candidate_index)

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model.score_matrix(train_sequences, train_query_text_tensor, candidate_tensor)
        bce_per_element = F.binary_cross_entropy_with_logits(logits, train_targets_tensor, reduction='none', pos_weight=pos_weight)
        bce_loss = (bce_per_element * train_weights_tensor).mean()

        if train_triplets:
            anchor_sequences = [train_sequences[int(row['query_index'])] for row in train_triplets]
            anchor_text = to_tensor(
                np.stack([query_text_features[train_query_positions[int(row['query_index'])]] for row in train_triplets], axis=0),
                config.molecular_encoder.device,
            )
            positive_tensor = to_tensor(
                np.stack([candidate_text_features[int(row['positive_index'])] for row in train_triplets], axis=0),
                config.molecular_encoder.device,
            )
            negative_tensor = to_tensor(
                np.stack([candidate_text_features[int(row['negative_index'])] for row in train_triplets], axis=0),
                config.molecular_encoder.device,
            )
            positive_scores, negative_scores = model.score_triplets(anchor_sequences, anchor_text, positive_tensor, negative_tensor)
            triplet_loss = F.relu(config.triplet_margin - positive_scores + negative_scores).mean()
        else:
            triplet_loss = torch.tensor(0.0, device=config.molecular_encoder.device)

        total_loss = config.bce_weight * bce_loss + config.triplet_weight * triplet_loss
        total_loss.backward()
        optimizer.step()

        _, val_metrics = evaluate_model(
            model,
            val_queries,
            query_text_features[val_query_positions] if val_query_positions else np.zeros((0, query_text_features.shape[1]), dtype=np.float32),
            candidate_rows,
            candidate_text_features,
            qrels_map,
            config.top_k_metrics,
            config.molecular_encoder.device,
        )
        val_mrr = choose_metric(val_metrics, 'mrr@10')
        history_rows.append({
            'epoch': epoch,
            'total_loss': round(float(total_loss.detach().cpu()), 6),
            'bce_loss': round(float(bce_loss.detach().cpu()), 6),
            'triplet_loss': round(float(triplet_loss.detach().cpu()), 6),
            'val_mrr@10': round(val_mrr, 6),
        })
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_rows: list[dict[str, object]] = []
    split_query_positions = {
        'train': train_query_positions,
        'val': val_query_positions,
        'test': test_query_positions,
    }
    for split_name, split_queries in [('train', train_queries), ('val', val_queries), ('test', test_queries)]:
        positions = split_query_positions[split_name]
        rankings, metrics = evaluate_model(
            model,
            split_queries,
            query_text_features[positions] if positions else np.zeros((0, query_text_features.shape[1]), dtype=np.float32),
            candidate_rows,
            candidate_text_features,
            qrels_map,
            config.top_k_metrics,
            config.molecular_encoder.device,
        )
        write_csv(output_dir / f'rankings_{split_name}.csv', rankings)
        metrics_rows.extend([{'split': split_name, **row} for row in metrics])

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'query_text_dim': query_text_features.shape[1],
        'candidate_text_dim': candidate_text_features.shape[1],
        'hidden_dim': config.hidden_dim,
        'embedding_dim': config.embedding_dim,
        'dropout': config.dropout,
    }
    torch.save(checkpoint, output_dir / 'model.pt')
    torch.save(model.query_tower.state_dict(), output_dir / 'query_tower_weights.pt')
    torch.save(model.candidate_tower.state_dict(), output_dir / 'candidate_tower_weights.pt')
    save_config(output_dir / 'config.json', config)
    query_vectorizer.save(output_dir / 'query_text_vectorizer.pkl')
    candidate_vectorizer.save(output_dir / 'candidate_text_vectorizer.pkl')
    np.save(output_dir / 'query_text_features.npy', query_text_features)
    np.save(output_dir / 'candidate_text_features.npy', candidate_text_features)
    write_csv(output_dir / 'query_rows.csv', query_rows)
    write_csv(output_dir / 'biomarker_candidates.csv', candidate_rows)
    write_csv(output_dir / 'relation_rows.csv', relation_rows)
    write_csv(output_dir / 'triplet_rows.csv', triplet_rows)
    write_csv(output_dir / 'training_history.csv', history_rows)
    write_csv(output_dir / 'metrics.csv', metrics_rows)
    write_json(output_dir / 'class_to_biomarker.json', class_to_biomarker)
    write_json(output_dir / 'biomarker_to_class.json', biomarker_to_class)

    return {
        'best_val_mrr@10': round(best_val_mrr, 6),
        'best_epoch': best_epoch,
        'epochs_trained': len(history_rows),
        'metrics': metrics_rows,
    }


def cross_validate(config: TrainConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_runtime_dataset(
        curated_csv=config.curated_csv,
        compound_registry_csv=config.compound_registry_csv,
        compound_components_csv=config.compound_components_csv,
        seed_interventions_csv=config.seed_interventions_csv,
        num_folds=config.num_folds,
        seed=config.seed,
    )
    write_csv(output_dir / 'runtime_queries.csv', dataset.query_rows)
    write_csv(output_dir / 'runtime_biomarker_candidates.csv', dataset.candidate_rows)
    write_csv(output_dir / 'runtime_relations.csv', dataset.relation_rows)
    write_csv(output_dir / 'runtime_triplets.csv', dataset.triplet_rows)
    write_csv(output_dir / 'fold_assignments.csv', dataset.fold_assignments)
    write_csv(output_dir / 'structure_resolution.csv', dataset.structure_resolution_rows)

    fold_rows: list[dict[str, object]] = []
    best_epochs: list[int] = []
    for fold_index in range(config.num_folds):
        fold_dir = output_dir / f'fold_{fold_index}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_queries = assign_splits(dataset.query_rows, dataset.fold_assignments, fold_index, config.num_folds)
        split_relation_rows = [row for row in dataset.relation_rows if any(query['query_id'] == row['query_id'] for query in split_queries)]
        summary = train_single_split(split_queries, dataset.candidate_rows, split_relation_rows, dataset.triplet_rows, config, fold_dir)
        best_epochs.append(int(summary['best_epoch']) or 1)
        metric_lookup = {(row['split'], row['metric']): float(row['value']) for row in summary['metrics']}
        fold_rows.append(
            {
                'fold': fold_index,
                'best_epoch': summary['best_epoch'],
                'train_mrr@10': metric_lookup.get(('train', 'mrr@10'), 0.0),
                'train_recall@5': metric_lookup.get(('train', 'recall@5'), 0.0),
                'train_ndcg@10': metric_lookup.get(('train', 'ndcg@10'), 0.0),
                'val_mrr@10': metric_lookup.get(('val', 'mrr@10'), 0.0),
                'val_recall@5': metric_lookup.get(('val', 'recall@5'), 0.0),
                'val_ndcg@10': metric_lookup.get(('val', 'ndcg@10'), 0.0),
                'test_mrr@10': metric_lookup.get(('test', 'mrr@10'), 0.0),
                'test_recall@5': metric_lookup.get(('test', 'recall@5'), 0.0),
                'test_ndcg@10': metric_lookup.get(('test', 'ndcg@10'), 0.0),
            }
        )
    write_csv(output_dir / 'cross_validation_folds.csv', fold_rows)

    summary_rows: list[dict[str, object]] = []
    for metric_name in ['train_mrr@10', 'train_recall@5', 'train_ndcg@10', 'val_mrr@10', 'val_recall@5', 'val_ndcg@10', 'test_mrr@10', 'test_recall@5', 'test_ndcg@10']:
        values = [float(row[metric_name]) for row in fold_rows]
        summary_rows.append({'metric': metric_name, 'mean': round(float(np.mean(values)), 6), 'std': round(float(np.std(values)), 6)})
    write_csv(output_dir / 'cross_validation_summary.csv', summary_rows)
    return {
        'fold_rows': fold_rows,
        'summary_rows': summary_rows,
        'best_epoch_mean': max(1, int(round(float(np.mean(best_epochs))))) if best_epochs else 1,
        'runtime_dataset': dataset,
    }


def train_final_model(config: TrainConfig, dataset: RuntimeDataset, final_epochs: int, output_dir: str | Path) -> dict[str, object]:
    final_output_dir = Path(output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    all_queries = [dict(row, split='train_all') for row in dataset.query_rows]
    query_vectorizer, candidate_vectorizer, query_text_features, candidate_text_features = fit_vectorizers(
        all_queries,
        dataset.candidate_rows,
        config.query_text_max_features,
        config.candidate_text_max_features,
    )
    qrels_map = build_qrels_map(dataset.relation_rows)
    weight_map = build_weight_map(dataset.relation_rows)
    targets, weights = build_query_target_matrices(all_queries, dataset.candidate_rows, qrels_map, weight_map)
    class_to_biomarker, biomarker_to_class = build_class_mapping(dataset.candidate_rows)
    candidate_index = {row['candidate_id']: idx for idx, row in enumerate(dataset.candidate_rows)}
    train_triplets = prepare_triplet_rows(dataset.triplet_rows, all_queries, candidate_index)

    model = ChemicalBiomarkerTwinTower(
        molecular_encoder=MolecularEncoder(config.molecular_encoder),
        query_text_dim=query_text_features.shape[1],
        candidate_text_dim=candidate_text_features.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(config.molecular_encoder.device)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    candidate_tensor = to_tensor(candidate_text_features, config.molecular_encoder.device)
    query_text_tensor = to_tensor(query_text_features, config.molecular_encoder.device)
    targets_tensor = to_tensor(targets, config.molecular_encoder.device)
    weights_tensor = to_tensor(weights, config.molecular_encoder.device)
    sequences = parse_structure_inputs(all_queries)
    positive_count = max(float(targets.sum()), 1.0)
    negative_count = max(float(targets.size - targets.sum()), 1.0)
    pos_weight_value = negative_count / positive_count
    pos_weight = torch.full((candidate_text_features.shape[0],), pos_weight_value, dtype=torch.float32, device=config.molecular_encoder.device)

    history_rows: list[dict[str, object]] = []
    for epoch in range(1, final_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model.score_matrix(sequences, query_text_tensor, candidate_tensor)
        bce_per_element = F.binary_cross_entropy_with_logits(logits, targets_tensor, reduction='none', pos_weight=pos_weight)
        bce_loss = (bce_per_element * weights_tensor).mean()
        if train_triplets:
            anchor_sequences = [sequences[int(row['query_index'])] for row in train_triplets]
            anchor_text = to_tensor(np.stack([query_text_features[int(row['query_index'])] for row in train_triplets], axis=0), config.molecular_encoder.device)
            positive_tensor = to_tensor(np.stack([candidate_text_features[int(row['positive_index'])] for row in train_triplets], axis=0), config.molecular_encoder.device)
            negative_tensor = to_tensor(np.stack([candidate_text_features[int(row['negative_index'])] for row in train_triplets], axis=0), config.molecular_encoder.device)
            positive_scores, negative_scores = model.score_triplets(anchor_sequences, anchor_text, positive_tensor, negative_tensor)
            triplet_loss = F.relu(config.triplet_margin - positive_scores + negative_scores).mean()
        else:
            triplet_loss = torch.tensor(0.0, device=config.molecular_encoder.device)
        total_loss = config.bce_weight * bce_loss + config.triplet_weight * triplet_loss
        total_loss.backward()
        optimizer.step()
        history_rows.append({'epoch': epoch, 'total_loss': round(float(total_loss.detach().cpu()), 6), 'bce_loss': round(float(bce_loss.detach().cpu()), 6), 'triplet_loss': round(float(triplet_loss.detach().cpu()), 6)})

    rankings, metrics = evaluate_model(
        model,
        all_queries,
        query_text_features,
        dataset.candidate_rows,
        candidate_text_features,
        qrels_map,
        config.top_k_metrics,
        config.molecular_encoder.device,
    )

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'query_text_dim': query_text_features.shape[1],
        'candidate_text_dim': candidate_text_features.shape[1],
        'hidden_dim': config.hidden_dim,
        'embedding_dim': config.embedding_dim,
        'dropout': config.dropout,
    }
    torch.save(checkpoint, final_output_dir / 'model.pt')
    torch.save(model.query_tower.state_dict(), final_output_dir / 'query_tower_weights.pt')
    torch.save(model.candidate_tower.state_dict(), final_output_dir / 'candidate_tower_weights.pt')
    save_config(final_output_dir / 'config.json', config)
    query_vectorizer.save(final_output_dir / 'query_text_vectorizer.pkl')
    candidate_vectorizer.save(final_output_dir / 'candidate_text_vectorizer.pkl')
    np.save(final_output_dir / 'query_text_features.npy', query_text_features)
    np.save(final_output_dir / 'candidate_text_features.npy', candidate_text_features)
    write_csv(final_output_dir / 'query_rows.csv', all_queries)
    write_csv(final_output_dir / 'biomarker_candidates.csv', dataset.candidate_rows)
    write_csv(final_output_dir / 'relation_rows.csv', dataset.relation_rows)
    write_csv(final_output_dir / 'triplet_rows.csv', dataset.triplet_rows)
    write_csv(final_output_dir / 'training_history.csv', history_rows)
    write_csv(final_output_dir / 'metrics.csv', [{'split': 'train_all', **row} for row in metrics])
    write_csv(final_output_dir / 'rankings_train_all.csv', rankings)
    write_csv(final_output_dir / 'fold_assignments.csv', dataset.fold_assignments)
    write_csv(final_output_dir / 'structure_resolution.csv', dataset.structure_resolution_rows)
    write_json(final_output_dir / 'class_to_biomarker.json', class_to_biomarker)
    write_json(final_output_dir / 'biomarker_to_class.json', biomarker_to_class)
    return {
        'metrics': metrics,
        'history_rows': history_rows,
    }


def train_twin_tower(config: TrainConfig) -> dict[str, object]:
    cv_summary = cross_validate(config)
    final_output_dir = Path(config.output_dir) / 'final_model'
    final_summary = train_final_model(config, cv_summary['runtime_dataset'], cv_summary['best_epoch_mean'], final_output_dir)
    metric_lookup = {(row['metric']): float(row['value']) for row in final_summary['metrics']}
    return {
        'output_dir': str(Path(config.output_dir)),
        'final_model_dir': str(final_output_dir),
        'best_epoch_mean': cv_summary['best_epoch_mean'],
        'cv_test_mrr@10_mean': next(row['mean'] for row in cv_summary['summary_rows'] if row['metric'] == 'test_mrr@10'),
        'cv_test_recall@5_mean': next(row['mean'] for row in cv_summary['summary_rows'] if row['metric'] == 'test_recall@5'),
        'cv_test_ndcg@10_mean': next(row['mean'] for row in cv_summary['summary_rows'] if row['metric'] == 'test_ndcg@10'),
        'final_train_all_mrr@10': metric_lookup.get('mrr@10', 0.0),
        'final_train_all_recall@5': metric_lookup.get('recall@5', 0.0),
        'final_train_all_ndcg@10': metric_lookup.get('ndcg@10', 0.0),
    }


def load_bundle(model_dir: str | Path, device: str = 'cpu') -> LoadedModelBundle:
    model_dir = Path(model_dir)
    config = load_training_config(model_dir / 'config.json')
    checkpoint = torch.load(model_dir / 'model.pt', map_location=device)
    molecular_encoder = MolecularEncoder(
        MolecularEncoderConfig(
            preset=str(config['molecular_encoder']['preset']),
            model_name=str(config['molecular_encoder']['model_name']),
            representation=str(config['molecular_encoder']['representation']),
            trust_remote_code=bool(config['molecular_encoder']['trust_remote_code']),
            max_length=config['molecular_encoder'].get('max_length'),
            device=device,
            unfreeze_last_n_layers=int(config['molecular_encoder'].get('unfreeze_last_n_layers', 1)),
        )
    )
    model = ChemicalBiomarkerTwinTower(
        molecular_encoder=molecular_encoder,
        query_text_dim=int(checkpoint['query_text_dim']),
        candidate_text_dim=int(checkpoint['candidate_text_dim']),
        hidden_dim=int(checkpoint['hidden_dim']),
        embedding_dim=int(checkpoint['embedding_dim']),
        dropout=float(checkpoint['dropout']),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return LoadedModelBundle(
        model=model,
        query_vectorizer=SimpleTfidfVectorizer.load(model_dir / 'query_text_vectorizer.pkl'),
        candidate_vectorizer=SimpleTfidfVectorizer.load(model_dir / 'candidate_text_vectorizer.pkl'),
        query_rows=load_csv(model_dir / 'query_rows.csv'),
        candidate_rows=load_csv(model_dir / 'biomarker_candidates.csv'),
        candidate_text_features=np.load(model_dir / 'candidate_text_features.npy'),
        class_to_biomarker=json.loads((model_dir / 'class_to_biomarker.json').read_text(encoding='utf-8')),
        biomarker_to_class=json.loads((model_dir / 'biomarker_to_class.json').read_text(encoding='utf-8')),
        config=config,
    )


def predict_biomarkers_for_query_feature(
    bundle: LoadedModelBundle,
    sequence_groups: list[list[str]],
    query_texts: list[str],
    top_k: int,
    device: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    query_text_features = bundle.query_vectorizer.encode(query_texts)
    query_tensor = to_tensor(query_text_features, device)
    candidate_tensor = to_tensor(bundle.candidate_text_features, device)
    with torch.no_grad():
        logits = bundle.model.score_matrix(sequence_groups, query_tensor, candidate_tensor).detach().cpu().numpy()[0]
    ranked_indices = np.argsort(-logits)[:top_k]
    top_rows: list[dict[str, object]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        row = bundle.candidate_rows[index]
        top_rows.append(
            {
                'rank': rank,
                'class_index': index,
                'candidate_id': row['candidate_id'],
                'surrogate': row['surrogate'],
                'logit': round(float(logits[index]), 6),
                'surrogate_category': row.get('surrogate_category', ''),
                'pathways': row.get('pathways', ''),
                'hallmarks': row.get('hallmarks', ''),
            }
        )
    return logits, top_rows


def resolve_prediction_input(bundle: LoadedModelBundle, intervention: str, structure: str, query_text: str) -> tuple[list[list[str]], list[str]]:
    if intervention:
        matched = [row for row in bundle.query_rows if row['intervention'] == intervention]
        if not matched:
            raise ValueError(f'Intervention not found in saved model query table: {intervention}')
        row = matched[0]
        return [json.loads(row['structure_inputs'])], [row['query_text']]
    if structure:
        fallback_text = query_text or 'Custom intervention query without curated metadata.'
        return [[structure]], [fallback_text]
    raise ValueError('Provide either intervention or structure')


def train_main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Train the chemical-to-biomarker Twin Tower recommender with cross-validation.')
    parser.add_argument('--curated-csv', default=str(root / 'data' / 'processed' / 'curated_biomarkers.csv'))
    parser.add_argument('--compound-registry-csv', default=str(root / 'data' / 'processed' / 'compound_registry.csv'))
    parser.add_argument('--compound-components-csv', default=str(root / 'data' / 'processed' / 'compound_components.csv'))
    parser.add_argument('--seed-interventions-csv', default=str(root / 'data' / 'seed_interventions.csv'))
    parser.add_argument('--output-dir', default=str(root / 'outputs' / 'twin_tower_mvp'))
    parser.add_argument('--molecule-encoder', choices=sorted(MOLECULAR_ENCODER_PRESETS.keys()), default='molformer')
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--query-text-max-features', type=int, default=2048)
    parser.add_argument('--candidate-text-max-features', type=int, default=4096)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--triplet-margin', type=float, default=0.2)
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--triplet-weight', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--early-stopping-patience', type=int, default=4)
    parser.add_argument('--top-k-metrics', type=int, default=10)
    parser.add_argument('--predict-top-k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--unfreeze-last-n-layers', type=int, default=1)
    args = parser.parse_args(argv)

    preset = MOLECULAR_ENCODER_PRESETS[args.molecule_encoder]
    encoder_config = MolecularEncoderConfig(
        preset=args.molecule_encoder,
        model_name=preset['model_name'],
        representation=preset['representation'],
        trust_remote_code=bool(preset['trust_remote_code']),
        max_length=args.max_length,
        device=args.device,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
    )
    config = TrainConfig(
        curated_csv=args.curated_csv,
        compound_registry_csv=args.compound_registry_csv,
        compound_components_csv=args.compound_components_csv,
        seed_interventions_csv=args.seed_interventions_csv,
        output_dir=args.output_dir,
        molecular_encoder=encoder_config,
        query_text_max_features=args.query_text_max_features,
        candidate_text_max_features=args.candidate_text_max_features,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        triplet_margin=args.triplet_margin,
        bce_weight=args.bce_weight,
        triplet_weight=args.triplet_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        top_k_metrics=args.top_k_metrics,
        predict_top_k=args.predict_top_k,
        seed=args.seed,
        num_folds=args.num_folds,
    )
    summary = train_twin_tower(config)
    for key, value in summary.items():
        print(f'{key}: {value}')
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Evaluate a trained chemical-to-biomarker Twin Tower model.')
    parser.add_argument('--model-dir', default=str(root / 'outputs' / 'twin_tower_mvp' / 'final_model'))
    parser.add_argument('--top-k-metrics', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)

    bundle = load_bundle(args.model_dir, device=args.device)
    query_rows = load_csv(Path(args.model_dir) / 'query_rows.csv')
    relation_rows = load_csv(Path(args.model_dir) / 'relation_rows.csv')
    qrels_map = build_qrels_map(relation_rows)
    query_text_features = np.load(Path(args.model_dir) / 'query_text_features.npy')
    rankings, metrics = evaluate_model(
        bundle.model,
        query_rows,
        query_text_features,
        bundle.candidate_rows,
        bundle.candidate_text_features,
        qrels_map,
        args.top_k_metrics,
        args.device,
    )
    write_csv(Path(args.model_dir) / 'eval_rankings.csv', rankings)
    write_csv(Path(args.model_dir) / 'eval_metrics.csv', [{'split': 'train_all', **row} for row in metrics])
    for row in metrics:
        print(f"train_all {row['metric']}: {row['value']}")
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Predict ranked biomarker logits from a chemical intervention.')
    parser.add_argument('--model-dir', default=str(root / 'outputs' / 'twin_tower_mvp' / 'final_model'))
    parser.add_argument('--intervention', default='')
    parser.add_argument('--structure', default='')
    parser.add_argument('--query-text', default='')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args(argv)

    bundle = load_bundle(args.model_dir, device=args.device)
    sequence_groups, query_texts = resolve_prediction_input(bundle, args.intervention, args.structure, args.query_text)
    logits, top_rows = predict_biomarkers_for_query_feature(bundle, sequence_groups, query_texts, args.top_k, args.device)
    if args.output_csv:
        write_csv(args.output_csv, top_rows)
    print(f'candidate_count: {len(logits)}')
    print('class_to_biomarker:', json.dumps(bundle.class_to_biomarker, ensure_ascii=False))
    for row in top_rows:
        print(f"{row['rank']}. class={row['class_index']} biomarker={row['surrogate']} logit={row['logit']}")
    return 0

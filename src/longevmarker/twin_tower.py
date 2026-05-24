from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import re
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\+\-]+")


MOLECULAR_ENCODER_PRESETS = {
    'chemberta': {
        'label': 'ChemBERTa',
        'model_name': 'seyonec/ChemBERTa-zinc-base-v1',
        'input_column': 'canonical_smiles',
        'trust_remote_code': False,
        'representation': 'smiles',
    },
    'molformer': {
        'label': 'MolFormer',
        'model_name': 'ibm-research/MoLFormer-XL-both-10pct',
        'input_column': 'canonical_smiles',
        'trust_remote_code': True,
        'representation': 'smiles',
    },
    'selformer': {
        'label': 'SELFormer',
        'model_name': 'HUBioDataLab/SELFormer',
        'input_column': 'selfies_sequence',
        'trust_remote_code': False,
        'representation': 'selfies',
    },
}


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


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_ngrams(tokens: list[str]) -> list[str]:
    return tokens + [f'{tokens[i]}__{tokens[i + 1]}' for i in range(len(tokens) - 1)]


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


@dataclass
class MolecularEncoderConfig:
    preset: str
    model_name: str
    input_column: str
    trust_remote_code: bool
    representation: str
    max_length: int | None = None
    device: str = 'cpu'


class MolecularEncoder:
    def __init__(self, config: MolecularEncoderConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()
        self.model.to(config.device)
        self.output_dim = int(getattr(self.model.config, 'hidden_size', 768))
        self.effective_max_length = self._resolve_max_length(config.max_length)

    def _resolve_max_length(self, requested: int | None) -> int:
        model_max = getattr(self.tokenizer, 'model_max_length', None)
        if not isinstance(model_max, int) or model_max <= 0 or model_max > 100000:
            model_max = 512
        if requested is None:
            return model_max
        return min(requested, model_max)

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _encode_sequences(self, sequences: list[str], batch_size: int = 4) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.effective_max_length,
                return_tensors='pt',
            )
            inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                pooled = self._mean_pool(hidden_states, inputs['attention_mask'])
                pooled = F.normalize(pooled, dim=-1)
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, self.output_dim), dtype=np.float32)

    def _row_sequences(self, row: dict[str, str]) -> list[str]:
        input_column = self.config.input_column
        if input_column == 'canonical_smiles' and row.get('component_count', '1') != '1' and row.get('component_canonical_smiles'):
            return [item for item in row['component_canonical_smiles'].split('|') if item]
        if input_column == 'connectivity_smiles' and row.get('component_count', '1') != '1' and row.get('component_connectivity_smiles'):
            return [item for item in row['component_connectivity_smiles'].split('|') if item]
        value = row.get(input_column, '').strip()
        return [value] if value else []

    def encode_rows(self, rows: list[dict[str, str]], batch_size: int = 4) -> np.ndarray:
        sequence_cache: dict[str, np.ndarray] = {}
        all_sequences: list[str] = []
        for row in rows:
            for sequence in self._row_sequences(row):
                if sequence not in sequence_cache:
                    all_sequences.append(sequence)
                    sequence_cache[sequence] = np.zeros((self.output_dim,), dtype=np.float32)
        if all_sequences:
            encoded = self._encode_sequences(all_sequences, batch_size=batch_size)
            for sequence, vector in zip(all_sequences, encoded, strict=True):
                sequence_cache[sequence] = vector
        row_embeddings: list[np.ndarray] = []
        for row in rows:
            components = self._row_sequences(row)
            if not components:
                row_embeddings.append(np.zeros((self.output_dim,), dtype=np.float32))
                continue
            component_vectors = np.stack([sequence_cache[sequence] for sequence in components], axis=0)
            row_embeddings.append(component_vectors.mean(axis=0))
        return normalize_rows(np.stack(row_embeddings, axis=0).astype(np.float32))

    def encode_structure(self, structure: str) -> np.ndarray:
        return self._encode_sequences([structure], batch_size=1)[0]


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
    def __init__(self, query_input_dim: int, candidate_input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_tower = ProjectionTower(query_input_dim, hidden_dim, embedding_dim, dropout)
        self.candidate_tower = ProjectionTower(candidate_input_dim, hidden_dim, embedding_dim, dropout)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode_queries(self, query_features: torch.Tensor) -> torch.Tensor:
        return self.query_tower(query_features)

    def encode_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_tower(candidate_features)

    def score_matrix(self, query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        query_embeddings = self.encode_queries(query_features)
        candidate_embeddings = self.encode_candidates(candidate_features)
        temperature = torch.exp(self.log_temperature).clamp_min(1e-4)
        return query_embeddings @ candidate_embeddings.T / temperature

    def score_triplets(
        self,
        query_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = self.encode_queries(query_features)
        positive_embeddings = self.encode_candidates(positive_features)
        negative_embeddings = self.encode_candidates(negative_features)
        temperature = torch.exp(self.log_temperature).clamp_min(1e-4)
        positive_scores = (query_embeddings * positive_embeddings).sum(dim=-1) / temperature
        negative_scores = (query_embeddings * negative_embeddings).sum(dim=-1) / temperature
        return positive_scores, negative_scores


@dataclass
class TrainConfig:
    molecular_queries_csv: str
    biomarker_candidates_csv: str
    qrels_csv: str
    triplets_csv: str
    output_dir: str
    molecular_encoder: MolecularEncoderConfig
    text_max_features: int = 4096
    hidden_dim: int = 256
    embedding_dim: int = 128
    dropout: float = 0.1
    triplet_margin: float = 0.2
    bce_weight: float = 1.0
    triplet_weight: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    batch_size: int = 4
    top_k: int = 10
    seed: int = 42


@dataclass
class LoadedModelBundle:
    model: ChemicalBiomarkerTwinTower
    vectorizer: SimpleTfidfVectorizer
    query_rows: list[dict[str, str]]
    candidate_rows: list[dict[str, str]]
    candidate_base_features: np.ndarray
    config: dict[str, object]


def split_rows(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row['split'] == split_name]


def build_candidate_rows(path: str | Path) -> list[dict[str, str]]:
    rows = load_csv(path)
    candidates: list[dict[str, str]] = []
    for row in rows:
        normalized = dict(row)
        normalized['candidate_id'] = f"surrogate:{row['surrogate_id']}"
        candidates.append(normalized)
    return candidates


def build_qrels_map(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        qrels[row['query_id']][row['doc_id']] = int(row['relevance'])
    return qrels


def build_query_target_matrices(
    query_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    qrels_map: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_index = {row['candidate_id']: idx for idx, row in enumerate(candidate_rows)}
    targets = np.zeros((len(query_rows), len(candidate_rows)), dtype=np.float32)
    weights = np.ones((len(query_rows), len(candidate_rows)), dtype=np.float32)
    for query_idx, row in enumerate(query_rows):
        for doc_id, relevance in qrels_map.get(row['query_id'], {}).items():
            if doc_id not in candidate_index:
                continue
            column = candidate_index[doc_id]
            targets[query_idx, column] = 1.0
            weights[query_idx, column] = float(relevance)
    return targets, weights


def prepare_triplet_rows(
    triplet_rows: list[dict[str, str]],
    query_index_by_intervention: dict[str, int],
    candidate_index: dict[str, int],
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for row in triplet_rows:
        intervention = row['intervention']
        positive_id = row['positive_id']
        negative_id = row['negative_id']
        if intervention not in query_index_by_intervention:
            continue
        if positive_id not in candidate_index or negative_id not in candidate_index:
            continue
        prepared.append(
            {
                'split': row['split'],
                'query_index': query_index_by_intervention[intervention],
                'positive_index': candidate_index[positive_id],
                'negative_index': candidate_index[negative_id],
                'intervention': intervention,
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
                    'split': query_row['split'],
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
    metrics = [
        {'metric': metric, 'value': round(float(np.mean(values)) if values else 0.0, 6)}
        for metric, values in metric_buckets.items()
    ]
    return rankings, metrics


def to_tensor(matrix: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def choose_metric(metrics: list[dict[str, object]], metric_name: str = 'mrr@10') -> float:
    for row in metrics:
        if row['metric'] == metric_name:
            return float(row['value'])
    return 0.0


def evaluate_model(
    model: ChemicalBiomarkerTwinTower,
    query_rows: list[dict[str, str]],
    query_base_features: np.ndarray,
    candidate_rows: list[dict[str, str]],
    candidate_base_features: np.ndarray,
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
    query_indices = [int(row['_query_matrix_index']) for row in query_rows]
    model.eval()
    with torch.no_grad():
        scores = model.score_matrix(
            to_tensor(query_base_features[query_indices], device),
            to_tensor(candidate_base_features, device),
        ).detach().cpu().numpy()
    return ranking_metrics(query_rows, candidate_rows, scores, qrels_map, top_k)


def save_config(path: str | Path, config: TrainConfig) -> None:
    payload = {
        'molecular_queries_csv': config.molecular_queries_csv,
        'biomarker_candidates_csv': config.biomarker_candidates_csv,
        'qrels_csv': config.qrels_csv,
        'triplets_csv': config.triplets_csv,
        'output_dir': config.output_dir,
        'text_max_features': config.text_max_features,
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
        'batch_size': config.batch_size,
        'top_k': config.top_k,
        'seed': config.seed,
        'molecular_encoder': {
            'preset': config.molecular_encoder.preset,
            'model_name': config.molecular_encoder.model_name,
            'input_column': config.molecular_encoder.input_column,
            'trust_remote_code': config.molecular_encoder.trust_remote_code,
            'representation': config.molecular_encoder.representation,
            'max_length': config.molecular_encoder.max_length,
            'device': config.molecular_encoder.device,
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def load_training_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def train_twin_tower(config: TrainConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_rows = load_csv(config.molecular_queries_csv)
    for index, row in enumerate(query_rows):
        row['_query_matrix_index'] = str(index)
    candidate_rows = build_candidate_rows(config.biomarker_candidates_csv)
    qrels_map = build_qrels_map(load_csv(config.qrels_csv))
    triplet_rows = load_csv(config.triplets_csv)

    vectorizer = SimpleTfidfVectorizer(max_features=config.text_max_features)
    vectorizer.fit(row['embedding_text'] for row in candidate_rows)
    candidate_base_features = vectorizer.encode([row['embedding_text'] for row in candidate_rows])

    molecular_encoder = MolecularEncoder(config.molecular_encoder)
    query_base_features = molecular_encoder.encode_rows(query_rows, batch_size=config.batch_size)

    candidate_index = {row['candidate_id']: idx for idx, row in enumerate(candidate_rows)}
    query_index_by_intervention = {row['intervention']: int(row['_query_matrix_index']) for row in query_rows}
    prepared_triplets = prepare_triplet_rows(triplet_rows, query_index_by_intervention, candidate_index)

    train_queries = split_rows(query_rows, 'train')
    val_queries = split_rows(query_rows, 'val')
    test_queries = split_rows(query_rows, 'test')

    train_targets, train_weights = build_query_target_matrices(train_queries, candidate_rows, qrels_map)

    model = ChemicalBiomarkerTwinTower(
        query_input_dim=query_base_features.shape[1],
        candidate_input_dim=candidate_base_features.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(config.molecular_encoder.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_query_indices = [int(row['_query_matrix_index']) for row in train_queries]
    train_query_tensor = to_tensor(query_base_features[train_query_indices], config.molecular_encoder.device)
    candidate_tensor = to_tensor(candidate_base_features, config.molecular_encoder.device)
    train_targets_tensor = to_tensor(train_targets, config.molecular_encoder.device)
    train_weights_tensor = to_tensor(train_weights, config.molecular_encoder.device)

    positive_count = max(float(train_targets.sum()), 1.0)
    negative_count = max(float(train_targets.size - train_targets.sum()), 1.0)
    pos_weight_value = negative_count / positive_count
    pos_weight = torch.full((candidate_base_features.shape[0],), pos_weight_value, dtype=torch.float32, device=config.molecular_encoder.device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_mrr = -1.0
    patience = 0
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model.score_matrix(train_query_tensor, candidate_tensor)
        bce_per_element = F.binary_cross_entropy_with_logits(
            logits,
            train_targets_tensor,
            reduction='none',
            pos_weight=pos_weight,
        )
        bce_loss = (bce_per_element * train_weights_tensor).mean()

        train_triplets = [row for row in prepared_triplets if row['split'] == 'train']
        if train_triplets:
            anchor_tensor = to_tensor(
                np.stack([query_base_features[int(row['query_index'])] for row in train_triplets], axis=0),
                config.molecular_encoder.device,
            )
            positive_tensor = to_tensor(
                np.stack([candidate_base_features[int(row['positive_index'])] for row in train_triplets], axis=0),
                config.molecular_encoder.device,
            )
            negative_tensor = to_tensor(
                np.stack([candidate_base_features[int(row['negative_index'])] for row in train_triplets], axis=0),
                config.molecular_encoder.device,
            )
            positive_scores, negative_scores = model.score_triplets(anchor_tensor, positive_tensor, negative_tensor)
            triplet_loss = F.relu(config.triplet_margin - positive_scores + negative_scores).mean()
        else:
            triplet_loss = torch.tensor(0.0, device=config.molecular_encoder.device)

        total_loss = config.bce_weight * bce_loss + config.triplet_weight * triplet_loss
        total_loss.backward()
        optimizer.step()

        _, val_metrics = evaluate_model(
            model=model,
            query_rows=val_queries,
            query_base_features=query_base_features,
            candidate_rows=candidate_rows,
            candidate_base_features=candidate_base_features,
            qrels_map=qrels_map,
            top_k=config.top_k,
            device=config.molecular_encoder.device,
        )
        val_mrr = choose_metric(val_metrics, 'mrr@10')
        history_rows.append(
            {
                'epoch': epoch,
                'total_loss': round(float(total_loss.detach().cpu()), 6),
                'bce_loss': round(float(bce_loss.detach().cpu()), 6),
                'triplet_loss': round(float(triplet_loss.detach().cpu()), 6),
                'val_mrr@10': round(val_mrr, 6),
            }
        )
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    all_metrics_rows: list[dict[str, object]] = []
    for split_name, split_queries in [('train', train_queries), ('val', val_queries), ('test', test_queries)]:
        rankings, metrics = evaluate_model(
            model=model,
            query_rows=split_queries,
            query_base_features=query_base_features,
            candidate_rows=candidate_rows,
            candidate_base_features=candidate_base_features,
            qrels_map=qrels_map,
            top_k=config.top_k,
            device=config.molecular_encoder.device,
        )
        write_csv(output_dir / f'rankings_{split_name}.csv', rankings)
        all_metrics_rows.extend([{'split': split_name, **row} for row in metrics])

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'query_input_dim': query_base_features.shape[1],
        'candidate_input_dim': candidate_base_features.shape[1],
        'hidden_dim': config.hidden_dim,
        'embedding_dim': config.embedding_dim,
        'dropout': config.dropout,
    }
    torch.save(checkpoint, output_dir / 'model.pt')
    save_config(output_dir / 'config.json', config)
    vectorizer.save(output_dir / 'text_vectorizer.pkl')
    np.save(output_dir / 'query_base_features.npy', query_base_features)
    np.save(output_dir / 'candidate_base_features.npy', candidate_base_features)
    write_csv(output_dir / 'query_rows.csv', [{k: v for k, v in row.items() if not k.startswith('_')} for row in query_rows])
    write_csv(output_dir / 'biomarker_candidates.csv', candidate_rows)
    write_csv(output_dir / 'training_history.csv', history_rows)
    write_csv(output_dir / 'metrics.csv', all_metrics_rows)

    return {
        'output_dir': str(output_dir),
        'best_val_mrr@10': round(best_val_mrr, 6),
        'epochs_trained': len(history_rows),
        'encoder': MOLECULAR_ENCODER_PRESETS[config.molecular_encoder.preset]['label'],
    }


def load_bundle(model_dir: str | Path, device: str = 'cpu') -> LoadedModelBundle:
    model_dir = Path(model_dir)
    config = load_training_config(model_dir / 'config.json')
    checkpoint = torch.load(model_dir / 'model.pt', map_location=device)
    model = ChemicalBiomarkerTwinTower(
        query_input_dim=int(checkpoint['query_input_dim']),
        candidate_input_dim=int(checkpoint['candidate_input_dim']),
        hidden_dim=int(checkpoint['hidden_dim']),
        embedding_dim=int(checkpoint['embedding_dim']),
        dropout=float(checkpoint['dropout']),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return LoadedModelBundle(
        model=model,
        vectorizer=SimpleTfidfVectorizer.load(model_dir / 'text_vectorizer.pkl'),
        query_rows=load_csv(model_dir / 'query_rows.csv'),
        candidate_rows=load_csv(model_dir / 'biomarker_candidates.csv'),
        candidate_base_features=np.load(model_dir / 'candidate_base_features.npy'),
        config=config,
    )


def predict_biomarkers_for_query_feature(
    bundle: LoadedModelBundle,
    query_feature: np.ndarray,
    top_k: int,
    device: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    query_tensor = to_tensor(query_feature.reshape(1, -1), device)
    candidate_tensor = to_tensor(bundle.candidate_base_features, device)
    with torch.no_grad():
        logits = bundle.model.score_matrix(query_tensor, candidate_tensor).detach().cpu().numpy()[0]
    ranked_indices = np.argsort(-logits)[:top_k]
    top_rows: list[dict[str, object]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        row = bundle.candidate_rows[index]
        top_rows.append(
            {
                'rank': rank,
                'candidate_id': row['candidate_id'],
                'surrogate': row['surrogate'],
                'logit': round(float(logits[index]), 6),
                'surrogate_category': row.get('surrogate_category', ''),
                'pathways': row.get('pathways', ''),
                'hallmarks': row.get('hallmarks', ''),
            }
        )
    return logits, top_rows


def load_encoder_config(payload: dict[str, object], device: str) -> MolecularEncoderConfig:
    return MolecularEncoderConfig(
        preset=str(payload['preset']),
        model_name=str(payload['model_name']),
        input_column=str(payload['input_column']),
        trust_remote_code=bool(payload['trust_remote_code']),
        representation=str(payload['representation']),
        max_length=payload.get('max_length'),
        device=device,
    )


def resolve_query_feature(
    bundle: LoadedModelBundle,
    device: str,
    intervention: str = '',
    structure: str = '',
) -> np.ndarray:
    if intervention:
        query_rows = [row for row in bundle.query_rows if row['intervention'] == intervention]
        if not query_rows:
            raise ValueError(f'Intervention not found in saved query table: {intervention}')
        query_base = np.load(Path(bundle.config['output_dir']) / 'query_base_features.npy')
        query_index = next(i for i, row in enumerate(bundle.query_rows) if row['intervention'] == intervention)
        return query_base[query_index]
    if structure:
        encoder = MolecularEncoder(load_encoder_config(bundle.config['molecular_encoder'], device=device))
        return encoder.encode_structure(structure)
    raise ValueError('Provide either intervention or structure')


def train_main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Train the chemical-to-biomarker Twin Tower recommender.')
    parser.add_argument('--molecular-queries-csv', default=str(root / 'data' / 'embedding' / 'molecular_interventions.csv'))
    parser.add_argument('--biomarker-candidates-csv', default=str(root / 'data' / 'embedding' / 'biomarker_semantic_texts.csv'))
    parser.add_argument('--qrels-csv', default=str(root / 'data' / 'embedding' / 'molecular_biomarker_semantic_qrels.csv'))
    parser.add_argument('--triplets-csv', default=str(root / 'data' / 'embedding' / 'molecule_to_biomarker_semantic_triplets.csv'))
    parser.add_argument('--output-dir', default=str(root / 'outputs' / 'twin_tower_mvp'))
    parser.add_argument('--molecule-encoder', choices=sorted(MOLECULAR_ENCODER_PRESETS.keys()), default='molformer')
    parser.add_argument('--input-column', default='')
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--text-max-features', type=int, default=4096)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--triplet-margin', type=float, default=0.2)
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--triplet-weight', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    preset = MOLECULAR_ENCODER_PRESETS[args.molecule_encoder]
    encoder_config = MolecularEncoderConfig(
        preset=args.molecule_encoder,
        model_name=preset['model_name'],
        input_column=args.input_column or preset['input_column'],
        trust_remote_code=bool(preset['trust_remote_code']),
        representation=preset['representation'],
        max_length=args.max_length,
        device=args.device,
    )
    config = TrainConfig(
        molecular_queries_csv=args.molecular_queries_csv,
        biomarker_candidates_csv=args.biomarker_candidates_csv,
        qrels_csv=args.qrels_csv,
        triplets_csv=args.triplets_csv,
        output_dir=args.output_dir,
        molecular_encoder=encoder_config,
        text_max_features=args.text_max_features,
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
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    summary = train_twin_tower(config)
    for key, value in summary.items():
        print(f'{key}: {value}')
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Evaluate a trained chemical-to-biomarker Twin Tower model.')
    parser.add_argument('--model-dir', default=str(root / 'outputs' / 'twin_tower_mvp'))
    parser.add_argument('--qrels-csv', default=str(root / 'data' / 'embedding' / 'molecular_biomarker_semantic_qrels.csv'))
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)

    bundle = load_bundle(args.model_dir, device=args.device)
    query_rows = load_csv(Path(args.model_dir) / 'query_rows.csv')
    for index, row in enumerate(query_rows):
        row['_query_matrix_index'] = str(index)
    qrels_map = build_qrels_map(load_csv(args.qrels_csv))
    query_base_features = np.load(Path(args.model_dir) / 'query_base_features.npy')
    candidate_base_features = bundle.candidate_base_features

    all_metrics_rows: list[dict[str, object]] = []
    for split_name in ['train', 'val', 'test']:
        split_query_rows = split_rows(query_rows, split_name)
        rankings, metrics = evaluate_model(
            model=bundle.model,
            query_rows=split_query_rows,
            query_base_features=query_base_features,
            candidate_rows=bundle.candidate_rows,
            candidate_base_features=candidate_base_features,
            qrels_map=qrels_map,
            top_k=args.top_k,
            device=args.device,
        )
        write_csv(Path(args.model_dir) / f'eval_rankings_{split_name}.csv', rankings)
        all_metrics_rows.extend([{'split': split_name, **row} for row in metrics])
    write_csv(Path(args.model_dir) / 'eval_metrics.csv', all_metrics_rows)
    for row in all_metrics_rows:
        print(f"{row['split']} {row['metric']}: {row['value']}")
    return 0


def predict_main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Predict ranked biomarker logits from a chemical intervention.')
    parser.add_argument('--model-dir', default=str(root / 'outputs' / 'twin_tower_mvp'))
    parser.add_argument('--intervention', default='')
    parser.add_argument('--structure', default='')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args(argv)

    bundle = load_bundle(args.model_dir, device=args.device)
    query_feature = resolve_query_feature(
        bundle=bundle,
        device=args.device,
        intervention=args.intervention,
        structure=args.structure,
    )
    logits, top_rows = predict_biomarkers_for_query_feature(bundle, query_feature, args.top_k, args.device)
    if args.output_csv:
        write_csv(args.output_csv, top_rows)
    print(f'candidate_count: {len(logits)}')
    for row in top_rows:
        print(f"{row['rank']}. {row['surrogate']} | logit={row['logit']}")
    return 0

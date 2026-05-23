from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def load_rows(path: str | Path) -> list[dict[str, str]]:
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


def mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def infer_max_length(tokenizer, model, explicit_max_length: int | None = None) -> int | None:
    if explicit_max_length is not None:
        return explicit_max_length
    tokenizer_limit = getattr(tokenizer, 'model_max_length', None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        return tokenizer_limit
    config_limit = getattr(getattr(model, 'config', None), 'max_position_embeddings', None)
    if isinstance(config_limit, int) and config_limit > 4:
        # Keep a small margin for special tokens on BERT/RoBERTa-style models.
        return config_limit - 2
    return None


def load_hf_encoder(model_name: str, trust_remote_code: bool = False):
    try:
        import torch
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('transformers and torch are required. Install them before running embedding scripts.') from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return tokenizer, model, device


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 8,
    max_length: int | None = None,
) -> np.ndarray:
    import torch
    outputs = []
    resolved_max_length = infer_max_length(tokenizer, model, explicit_max_length=max_length)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokenize_kwargs = {
                'padding': True,
                'truncation': resolved_max_length is not None,
                'return_tensors': 'pt',
            }
            if resolved_max_length is not None:
                tokenize_kwargs['max_length'] = resolved_max_length
            encoded = tokenizer(batch, **tokenize_kwargs)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            model_out = model(**encoded)
            last_hidden = model_out.last_hidden_state if hasattr(model_out, 'last_hidden_state') else model_out.hidden_states[-1]
            pooled = mean_pool(last_hidden, encoded['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def encode_component_aware(
    rows: list[dict[str, str]],
    input_column: str,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int | None,
) -> np.ndarray:
    import torch
    all_embeddings = []
    for row in rows:
        raw = row.get(input_column, '') or ''
        if '|' in raw:
            parts = [part.strip() for part in raw.split('|') if part.strip()]
        else:
            parts = [raw.strip()] if raw.strip() else []
        if not parts:
            raise ValueError(f'Missing molecular input for row: {row}')
        part_embeddings = encode_texts(parts, tokenizer, model, device, batch_size=batch_size, max_length=max_length)
        mean_embedding = part_embeddings.mean(axis=0, keepdims=True)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding, axis=1, keepdims=True)
        all_embeddings.append(mean_embedding[0])
    return np.vstack(all_embeddings)


def build_metadata(rows: list[dict[str, str]], id_column: str, input_column: str, model_name: str) -> list[dict[str, object]]:
    metadata = []
    for row in rows:
        metadata.append(
            {
                'row_id': row[id_column],
                'split': row.get('split', ''),
                'intervention': row.get('intervention', ''),
                'input_column': input_column,
                'model_name': model_name,
                'molecule_type': row.get('molecule_type', ''),
                'component_count': row.get('component_count', ''),
            }
        )
    return metadata


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Embed molecular intervention inputs with a Hugging Face encoder.')
    parser.add_argument('--input-csv', default=str(root / 'data' / 'embedding' / 'molecular_interventions.csv'))
    parser.add_argument('--id-column', default='query_id')
    parser.add_argument('--input-column', default='canonical_smiles')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--component-aware', action='store_true')
    args = parser.parse_args(argv)

    rows = load_rows(args.input_csv)
    tokenizer, model, device = load_hf_encoder(args.model_name, trust_remote_code=args.trust_remote_code)
    if args.component_aware:
        embeddings = encode_component_aware(
            rows,
            args.input_column,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        texts = [row[args.input_column] for row in rows]
        embeddings = encode_texts(texts, tokenizer, model, device, batch_size=args.batch_size, max_length=args.max_length)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix('.embeddings.npy'), embeddings)
    write_csv(prefix.with_suffix('.metadata.csv'), build_metadata(rows, args.id_column, args.input_column, args.model_name))
    print(f'rows_embedded: {len(rows)}')
    print(f'embedding_dim: {embeddings.shape[1] if embeddings.size else 0}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

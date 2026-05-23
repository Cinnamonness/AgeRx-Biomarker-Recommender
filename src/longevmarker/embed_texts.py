from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .embed_molecules import encode_texts, load_hf_encoder, write_csv


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def build_metadata(rows: list[dict[str, str]], id_column: str, text_column: str, model_name: str) -> list[dict[str, object]]:
    metadata = []
    for row in rows:
        metadata.append(
            {
                'row_id': row[id_column],
                'split': row.get('split', row.get('source', '')),
                'text_column': text_column,
                'model_name': model_name,
                'title': row.get('title', row.get('surrogate', row.get('intervention', ''))),
            }
        )
    return metadata


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Embed text rows with a Hugging Face encoder.')
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--id-column', required=True)
    parser.add_argument('--text-column', default='embedding_text')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args(argv)

    rows = load_rows(args.input_csv)
    tokenizer, model, device = load_hf_encoder(args.model_name, trust_remote_code=args.trust_remote_code)
    texts = [row[args.text_column] for row in rows]
    embeddings = encode_texts(texts, tokenizer, model, device, batch_size=args.batch_size)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix('.embeddings.npy'), embeddings)
    write_csv(prefix.with_suffix('.metadata.csv'), build_metadata(rows, args.id_column, args.text_column, args.model_name))
    print(f'rows_embedded: {len(rows)}')
    print(f'embedding_dim: {embeddings.shape[1] if embeddings.size else 0}')
    return 0

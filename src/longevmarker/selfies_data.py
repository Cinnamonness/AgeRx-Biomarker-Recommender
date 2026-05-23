from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def smiles_to_selfies(smiles: str) -> str:
    try:
        import selfies
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('selfies is not installed. Run: python3 -m pip install --user selfies') from exc
    return selfies.encoder(smiles)


def populate_selfies(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    updated: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        smiles = row.get('selfies_source_smiles') or row.get('canonical_smiles') or ''
        try:
            record['selfies_sequence'] = smiles_to_selfies(smiles) if smiles else ''
            record['selfies_conversion_status'] = 'ok' if smiles else 'missing_smiles'
        except Exception:
            record['selfies_sequence'] = ''
            record['selfies_conversion_status'] = 'conversion_failed'
        updated.append(record)
    return updated


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Populate SELFIES strings from SMILES in molecular_interventions.csv.')
    parser.add_argument('--input-path', default=str(root / 'data' / 'embedding' / 'molecular_interventions.csv'))
    parser.add_argument('--output-path', default='')
    args = parser.parse_args(argv)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path
    rows = load_csv(input_path)
    updated = populate_selfies(rows)
    write_csv(output_path, updated)
    print(f'molecular_interventions_with_selfies: {len(updated)}')
    return 0

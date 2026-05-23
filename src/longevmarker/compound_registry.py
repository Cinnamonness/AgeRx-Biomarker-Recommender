from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
import json

REGISTRY_SPECS = [
    {'intervention': 'Metformin', 'components': ['Metformin']},
    {'intervention': 'Rapamycin', 'components': ['Rapamycin']},
    {'intervention': 'Acarbose', 'components': ['Acarbose']},
    {'intervention': 'Semaglutide', 'components': ['Semaglutide']},
    {'intervention': 'Liraglutide', 'components': ['Liraglutide']},
    {'intervention': 'Empagliflozin', 'components': ['Empagliflozin']},
    {'intervention': 'Dasatinib + Quercetin', 'components': ['Dasatinib', 'Quercetin']},
    {'intervention': 'Fisetin', 'components': ['Fisetin']},
    {'intervention': 'Nicotinamide Riboside', 'components': ['Nicotinamide Riboside']},
    {'intervention': 'NMN', 'components': ['Nicotinamide mononucleotide']},
    {'intervention': 'Spermidine', 'components': ['Spermidine']},
    {'intervention': 'Resveratrol', 'components': ['Resveratrol']},
]

PUBCHEM_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,ConnectivitySMILES,IUPACName,MolecularFormula/JSON'


def slugify(text: str) -> str:
    import re
    normalized = re.sub(r'[^a-z0-9]+', '_', text.lower())
    return normalized.strip('_')


def fetch_pubchem_properties(name: str) -> dict[str, str]:
    url = PUBCHEM_URL.format(name=quote(name))
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    prop = data['PropertyTable']['Properties'][0]
    return {
        'lookup_name': name,
        'pubchem_cid': str(prop['CID']),
        'canonical_smiles': prop.get('SMILES', prop['ConnectivitySMILES']),
        'connectivity_smiles': prop['ConnectivitySMILES'],
        'iupac_name': prop.get('IUPACName', ''),
        'molecular_formula': prop.get('MolecularFormula', ''),
        'pubchem_url': f"https://pubchem.ncbi.nlm.nih.gov/compound/{prop['CID']}",
    }


def build_registry() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    intervention_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for spec in REGISTRY_SPECS:
        intervention = spec['intervention']
        fetched_components = [fetch_pubchem_properties(component_name) for component_name in spec['components']]
        for index, component in enumerate(fetched_components, start=1):
            component_rows.append(
                {
                    'intervention_id': slugify(intervention),
                    'intervention': intervention,
                    'component_index': index,
                    'component_name': component['lookup_name'],
                    'pubchem_cid': component['pubchem_cid'],
                    'canonical_smiles': component['canonical_smiles'],
                    'connectivity_smiles': component['connectivity_smiles'],
                    'molecular_formula': component['molecular_formula'],
                    'iupac_name': component['iupac_name'],
                    'pubchem_url': component['pubchem_url'],
                }
            )
        intervention_rows.append(
            {
                'intervention_id': slugify(intervention),
                'intervention': intervention,
                'molecule_type': 'combination' if len(fetched_components) > 1 else 'single_molecule',
                'component_count': len(fetched_components),
                'component_names': '|'.join(component['lookup_name'] for component in fetched_components),
                'pubchem_cids': '|'.join(component['pubchem_cid'] for component in fetched_components),
                'canonical_smiles': '.'.join(component['canonical_smiles'] for component in fetched_components),
                'connectivity_smiles': '.'.join(component['connectivity_smiles'] for component in fetched_components),
                'component_canonical_smiles': '|'.join(component['canonical_smiles'] for component in fetched_components),
                'component_connectivity_smiles': '|'.join(component['connectivity_smiles'] for component in fetched_components),
                'component_formulas': '|'.join(component['molecular_formula'] for component in fetched_components),
                'pubchem_urls': '|'.join(component['pubchem_url'] for component in fetched_components),
                'source': 'pubchem_pug_rest',
            }
        )
    return intervention_rows, component_rows


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


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Fetch compound registry for drug-only interventions from PubChem.')
    parser.add_argument('--registry-output', default=str(root / 'data' / 'processed' / 'compound_registry.csv'))
    parser.add_argument('--components-output', default=str(root / 'data' / 'processed' / 'compound_components.csv'))
    args = parser.parse_args(argv)

    registry_rows, component_rows = build_registry()
    write_csv(args.registry_output, registry_rows)
    write_csv(args.components_output, component_rows)
    print(f'compound_registry.csv: {len(registry_rows)}')
    print(f'compound_components.csv: {len(component_rows)}')
    return 0

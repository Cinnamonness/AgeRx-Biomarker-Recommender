from pathlib import Path
import unittest

from longevmarker.embedding_data import (
    build_biomarker_semantic_texts,
    build_pair_texts,
    build_surrogate_texts,
    load_curated_rows,
)
from longevmarker.molecular_embedding_data import (
    build_encoder_candidates,
    build_model_comparison_runs,
    build_molecular_interventions,
    build_molecule_to_biomarker_semantic_triplets,
    build_molecule_to_pair_triplets,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / 'data' / 'processed' / 'curated_biomarkers.csv'
REGISTRY = ROOT / 'data' / 'processed' / 'compound_registry.csv'


class MolecularEmbeddingDataTests(unittest.TestCase):
    def test_molecular_interventions_align_with_registry(self) -> None:
        curated_rows = load_curated_rows(CURATED)
        with REGISTRY.open('r', encoding='utf-8', newline='') as handle:
            import csv
            registry_rows = list(csv.DictReader(handle))
        molecular_rows = build_molecular_interventions(curated_rows, registry_rows)
        self.assertEqual(len(molecular_rows), 12)
        self.assertTrue(all(row['canonical_smiles'] for row in molecular_rows))

    def test_molecule_to_pair_triplets_cover_all_pairs(self) -> None:
        curated_rows = load_curated_rows(CURATED)
        pair_rows = build_pair_texts(curated_rows)
        surrogate_rows = build_surrogate_texts(curated_rows)
        biomarker_semantic_rows = build_biomarker_semantic_texts(curated_rows)
        with REGISTRY.open('r', encoding='utf-8', newline='') as handle:
            import csv
            registry_rows = list(csv.DictReader(handle))
        molecular_rows = build_molecular_interventions(curated_rows, registry_rows)
        triplets = build_molecule_to_pair_triplets(molecular_rows, pair_rows)
        biomarker_triplets = build_molecule_to_biomarker_semantic_triplets(molecular_rows, pair_rows, biomarker_semantic_rows)
        self.assertEqual(len(triplets), len(pair_rows))
        self.assertEqual(len(biomarker_triplets), len(pair_rows))
        self.assertTrue(all(row['positive_id'] != row['negative_id'] for row in triplets))
        self.assertTrue(surrogate_rows)

    def test_encoder_benchmark_tables_have_expected_sizes(self) -> None:
        candidates = build_encoder_candidates()
        runs = build_model_comparison_runs()
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(runs), 15)


if __name__ == '__main__':
    unittest.main()

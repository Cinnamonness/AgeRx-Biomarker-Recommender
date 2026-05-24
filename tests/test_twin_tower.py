from pathlib import Path
import unittest

from longevmarker.twin_tower import (
    build_runtime_dataset,
    build_qrels_map,
    build_query_target_matrices,
    build_weight_map,
    assign_splits,
    prepare_triplet_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / 'data' / 'processed' / 'curated_biomarkers.csv'
COMPOUND_REGISTRY = ROOT / 'data' / 'processed' / 'compound_registry.csv'
COMPOUND_COMPONENTS = ROOT / 'data' / 'processed' / 'compound_components.csv'
SEED = ROOT / 'data' / 'seed_interventions.csv'


class TwinTowerChemicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_runtime_dataset(
            curated_csv=CURATED,
            compound_registry_csv=COMPOUND_REGISTRY,
            compound_components_csv=COMPOUND_COMPONENTS,
            seed_interventions_csv=SEED,
            num_folds=5,
            seed=42,
        )

    def test_runtime_dataset_uses_full_curated_intervention_space(self) -> None:
        interventions = {row['intervention'] for row in self.dataset.query_rows}
        self.assertEqual(len(interventions), 42)
        self.assertGreaterEqual(len(self.dataset.candidate_rows), 41)

    def test_candidate_ids_are_surrogate_prefixed(self) -> None:
        self.assertTrue(self.dataset.candidate_rows)
        self.assertTrue(all(row['candidate_id'].startswith('surrogate:') for row in self.dataset.candidate_rows))

    def test_qrels_expand_into_dense_target_matrix(self) -> None:
        qrels_map = build_qrels_map(self.dataset.relation_rows)
        weight_map = build_weight_map(self.dataset.relation_rows)
        split_queries = assign_splits(self.dataset.query_rows, self.dataset.fold_assignments, 0, 5)
        targets, weights = build_query_target_matrices(split_queries, self.dataset.candidate_rows, qrels_map, weight_map)
        self.assertEqual(targets.shape[0], len(split_queries))
        self.assertEqual(targets.shape[1], len(self.dataset.candidate_rows))
        self.assertEqual(weights.shape, targets.shape)
        self.assertGreater(float(targets.sum()), 0.0)

    def test_triplets_reference_existing_queries_and_candidates(self) -> None:
        split_queries = assign_splits(self.dataset.query_rows, self.dataset.fold_assignments, 0, 5)
        candidate_index = {row['candidate_id']: idx for idx, row in enumerate(self.dataset.candidate_rows)}
        prepared = prepare_triplet_rows(self.dataset.triplet_rows, split_queries, candidate_index)
        self.assertTrue(prepared)
        self.assertTrue(all(row['positive_index'] != row['negative_index'] for row in prepared))

    def test_fold_assignments_cover_all_queries_once(self) -> None:
        counts = {}
        for row in self.dataset.fold_assignments:
            counts[row['intervention']] = counts.get(row['intervention'], 0) + 1
        self.assertTrue(counts)
        self.assertTrue(all(value == 1 for value in counts.values()))


if __name__ == '__main__':
    unittest.main()

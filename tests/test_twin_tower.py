from pathlib import Path
import unittest

from longevmarker.twin_tower import (
    build_candidate_rows,
    build_qrels_map,
    build_query_target_matrices,
    load_csv,
    prepare_triplet_rows,
)

ROOT = Path(__file__).resolve().parents[1]
MOLECULAR_QUERIES = ROOT / 'data' / 'embedding' / 'molecular_interventions.csv'
BIOMARKER_CANDIDATES = ROOT / 'data' / 'embedding' / 'biomarker_semantic_texts.csv'
QRELS = ROOT / 'data' / 'embedding' / 'molecular_biomarker_semantic_qrels.csv'
TRIPLETS = ROOT / 'data' / 'embedding' / 'molecule_to_biomarker_semantic_triplets.csv'


class TwinTowerChemicalTests(unittest.TestCase):
    def test_candidate_ids_are_surrogate_prefixed(self) -> None:
        rows = build_candidate_rows(BIOMARKER_CANDIDATES)
        self.assertTrue(rows)
        self.assertTrue(all(row['candidate_id'].startswith('surrogate:') for row in rows))

    def test_qrels_expand_into_dense_target_matrix(self) -> None:
        query_rows = load_csv(MOLECULAR_QUERIES)
        candidate_rows = build_candidate_rows(BIOMARKER_CANDIDATES)
        qrels_map = build_qrels_map(load_csv(QRELS))
        targets, weights = build_query_target_matrices(query_rows, candidate_rows, qrels_map)
        self.assertEqual(targets.shape[0], len(query_rows))
        self.assertEqual(targets.shape[1], len(candidate_rows))
        self.assertEqual(weights.shape, targets.shape)
        self.assertGreater(float(targets.sum()), 0.0)

    def test_triplets_reference_existing_queries_and_candidates(self) -> None:
        query_rows = load_csv(MOLECULAR_QUERIES)
        candidate_rows = build_candidate_rows(BIOMARKER_CANDIDATES)
        candidate_index = {row['candidate_id']: idx for idx, row in enumerate(candidate_rows)}
        query_index_by_intervention = {row['intervention']: idx for idx, row in enumerate(query_rows)}
        prepared = prepare_triplet_rows(load_csv(TRIPLETS), query_index_by_intervention, candidate_index)
        self.assertTrue(prepared)
        self.assertTrue(all(row['positive_index'] != row['negative_index'] for row in prepared))
        self.assertTrue(all(row['intervention'] in query_index_by_intervention for row in prepared))


if __name__ == '__main__':
    unittest.main()

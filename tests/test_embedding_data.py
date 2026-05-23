from pathlib import Path
import unittest

from longevmarker.embedding_data import (
    build_biomarker_semantic_texts,
    build_pair_texts,
    build_qrels,
    build_queries,
    build_surrogate_texts,
    build_triplets,
    build_intervention_texts,
    load_curated_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / 'data' / 'processed' / 'curated_biomarkers.csv'
DEMO = ROOT / 'data' / 'processed' / 'demo_queries.csv'


class EmbeddingDataTests(unittest.TestCase):
    def test_embedding_tables_have_expected_basic_sizes(self) -> None:
        rows = load_curated_rows(CURATED)
        intervention_rows = build_intervention_texts(rows)
        surrogate_rows = build_surrogate_texts(rows)
        biomarker_semantic_rows = build_biomarker_semantic_texts(rows)
        pair_rows = build_pair_texts(rows)

        self.assertEqual(len(rows), 70)
        self.assertEqual(len(pair_rows), 70)
        self.assertEqual(len(intervention_rows), 12)
        self.assertGreaterEqual(len(surrogate_rows), 15)
        self.assertEqual(len(biomarker_semantic_rows), len(surrogate_rows))

    def test_biomarker_semantic_rows_include_moa_and_hallmark_context(self) -> None:
        rows = load_curated_rows(CURATED)
        biomarker_semantic_rows = build_biomarker_semantic_texts(rows)
        self.assertTrue(all('mechanism_hints' in row for row in biomarker_semantic_rows))
        self.assertTrue(all('mechanism_themes' in row for row in biomarker_semantic_rows))
        self.assertTrue(any(row['mechanism_themes'] for row in biomarker_semantic_rows))
        self.assertTrue(all('Aging hallmarks:' in row['embedding_text'] for row in biomarker_semantic_rows))

    def test_queries_and_qrels_reference_existing_pairs(self) -> None:
        rows = load_curated_rows(CURATED)
        intervention_rows = build_intervention_texts(rows)
        pair_rows = build_pair_texts(rows)
        queries = build_queries(pair_rows, intervention_rows, DEMO)
        qrels = build_qrels(queries, pair_rows)
        pair_doc_ids = {f"pair:{row['pair_id']}" for row in pair_rows}

        self.assertGreater(len(queries), 20)
        self.assertTrue(qrels)
        self.assertTrue(all(row['doc_id'] in pair_doc_ids for row in qrels))

    def test_triplets_use_distinct_positive_and_negative_targets(self) -> None:
        rows = load_curated_rows(CURATED)
        pair_rows = build_pair_texts(rows)
        surrogate_rows = build_surrogate_texts(rows)
        triplets = build_triplets(pair_rows, surrogate_rows)

        self.assertEqual(len(triplets), len(pair_rows) * 2)
        self.assertTrue(all(row['positive_id'] != row['negative_id'] for row in triplets))


if __name__ == '__main__':
    unittest.main()

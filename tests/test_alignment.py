import unittest

import numpy as np

from longevmarker.alignment import fit_ridge_projection, project_embeddings
from longevmarker.retrieval_benchmark import filter_queries_by_split


class AlignmentTests(unittest.TestCase):
    def test_ridge_projection_matches_target_space_shape(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        weights = fit_ridge_projection(x, y, regularization=0.01)
        projected = project_embeddings(x, weights)
        self.assertEqual(projected.shape, (2, 3))
        self.assertTrue(np.allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5))

    def test_filter_queries_by_split_keeps_requested_rows(self):
        embeddings = np.arange(12, dtype=np.float32).reshape(4, 3)
        meta = [
            {'row_id': 'a', 'split': 'train'},
            {'row_id': 'b', 'split': 'val'},
            {'row_id': 'c', 'split': 'train'},
            {'row_id': 'd', 'split': 'test'},
        ]
        filtered_embeddings, filtered_meta = filter_queries_by_split(embeddings, meta, 'train')
        self.assertEqual(filtered_embeddings.shape, (2, 3))
        self.assertEqual([row['row_id'] for row in filtered_meta], ['a', 'c'])


if __name__ == '__main__':
    unittest.main()

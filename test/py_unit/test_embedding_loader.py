import unittest

import numpy as np

from experiments.embedding_loader import EmbeddingLayer


class TestEmbeddingLayerUpdates(unittest.TestCase):
    def test_update_caps_aggregated_deltas(self):
        layer = EmbeddingLayer(num_embeddings=3, embedding_size=2, encoding_type="uniform")
        layer.mu.fill(0.0)
        layer.var.fill(1.0)

        layer.update(
            idx=np.array([1, 1], dtype=np.int32),
            mu_delta=np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
            var_delta=np.array([[2.0, -2.0], [2.0, -2.0]], dtype=np.float32),
        )

        np.testing.assert_allclose(layer.mu[1], np.array([0.5, 0.5], dtype=np.float32))
        np.testing.assert_allclose(layer.var[1], np.array([1.0, 0.5], dtype=np.float32))

    def test_apply_accumulated_updates_uses_same_cap(self):
        layer = EmbeddingLayer(num_embeddings=3, embedding_size=2, encoding_type="uniform")
        layer.mu.fill(0.0)
        layer.var.fill(1.0)

        layer.update(
            idx=np.array([1], dtype=np.int32),
            mu_delta=np.array([[3.0, 3.0]], dtype=np.float32),
            var_delta=np.array([[3.0, -3.0]], dtype=np.float32),
            accumulate=True,
        )
        layer.update(
            idx=np.array([1], dtype=np.int32),
            mu_delta=np.array([[3.0, 3.0]], dtype=np.float32),
            var_delta=np.array([[3.0, -3.0]], dtype=np.float32),
            accumulate=True,
        )

        layer.apply_accumulated_updates()

        np.testing.assert_allclose(layer.mu[1], np.array([0.5, 0.5], dtype=np.float32))
        np.testing.assert_allclose(layer.var[1], np.array([1.0, 0.5], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

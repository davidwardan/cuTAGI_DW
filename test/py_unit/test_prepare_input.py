import sys
import types
import unittest

import numpy as np

if "networkx" not in sys.modules:
    sys.modules["networkx"] = types.ModuleType("networkx")

from experiments.utils import LSTMStateContainer, prepare_input


class DummyEmbeddings:
    def __init__(self, mu: np.ndarray, var: np.ndarray):
        self._mu = np.asarray(mu, dtype=np.float32)
        self._var = np.asarray(var, dtype=np.float32)

    def __call__(self, indices: np.ndarray):
        idx = np.asarray(indices, dtype=np.int64)
        return self._mu[idx].copy(), self._var[idx].copy()


class TestPrepareInputEmbeddings(unittest.TestCase):
    def test_embeddings_added_for_non_sequential_model(self):
        x = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([2, -1], dtype=np.int32)
        embeddings = DummyEmbeddings(
            mu=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            var=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        )

        out_x, out_var_x = prepare_input(
            x=x,
            var_x=None,
            look_back_mu=None,
            look_back_var=None,
            indices=indices,
            embeddings=embeddings,
            sequential_model=False,
        )

        out_x = out_x.reshape(2, 6)
        out_var_x = out_var_x.reshape(2, 6)

        expected_x = np.array(
            [
                [10.0, 11.0, 12.0, 13.0, 5.0, 6.0],
                [20.0, 21.0, 22.0, 23.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        expected_var_x = np.array(
            [
                [1e-6, 1e-6, 1e-6, 1e-6, 0.5, 0.6],
                [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(out_x, expected_x)
        np.testing.assert_allclose(out_var_x, expected_var_x)

    def test_embeddings_added_for_sequential_model(self):
        x = np.array(
            [
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
                [4.0, 40.0, 5.0, 50.0, 6.0, 60.0],
            ],
            dtype=np.float32,
        )
        var_x = np.full_like(x, 0.25, dtype=np.float32)
        indices = np.array([1, -1], dtype=np.int32)
        embeddings = DummyEmbeddings(
            mu=np.array([[0.1, 0.2], [1.1, 1.2]], dtype=np.float32),
            var=np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
        )

        out_x, out_var_x = prepare_input(
            x=x,
            var_x=var_x,
            look_back_mu=None,
            look_back_var=None,
            indices=indices,
            embeddings=embeddings,
            input_seq_len=3,
            sequential_model=True,
        )

        self.assertEqual(out_x.shape, (2, 3, 4))
        out_var_x = out_var_x.reshape(2, 3, 4)

        expected_x = np.array(
            [
                [
                    [1.0, 10.0, 1.1, 1.2],
                    [2.0, 20.0, 1.1, 1.2],
                    [3.0, 30.0, 1.1, 1.2],
                ],
                [
                    [4.0, 40.0, 0.0, 0.0],
                    [5.0, 50.0, 0.0, 0.0],
                    [6.0, 60.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        expected_var_x = np.array(
            [
                [
                    [0.25, 0.25, 0.03, 0.04],
                    [0.25, 0.25, 0.03, 0.04],
                    [0.25, 0.25, 0.03, 0.04],
                ],
                [
                    [0.25, 0.25, 1e-6, 1e-6],
                    [0.25, 0.25, 1e-6, 1e-6],
                    [0.25, 0.25, 1e-6, 1e-6],
                ],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(out_x, expected_x)
        np.testing.assert_allclose(out_var_x, expected_var_x)


class DummyNetNoLstmStates:
    def __init__(self):
        self.set_calls = 0

    def get_lstm_states(self, time_step: int = -1):
        return {}

    def set_lstm_states(self, states: dict) -> None:
        self.set_calls += 1


class TestLSTMStateContainerNoStates(unittest.TestCase):
    def test_skip_set_states_when_model_has_no_lstm_states(self):
        container = LSTMStateContainer(num_series=5, layer_state_shapes={0: 4, 1: 4})
        net = DummyNetNoLstmStates()
        indices = np.array([0, 1, -1], dtype=np.int32)

        container.set_states_on_net(indices, net)
        container.update_states_from_net(indices, net)

        self.assertEqual(net.set_calls, 0)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
import yaml

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.modules.setdefault("networkx", MagicMock())

from experiments.config import Config
from experiments.data_loader import TimeSeriesDataBuilder
from experiments.utils import (
    prepare_input,
    extract_target_history,
)


class TestExperimentDataLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.x_path = os.path.join(self.tmpdir.name, "x.csv")
        self.dt_path = os.path.join(self.tmpdir.name, "dt.csv")

        pd.DataFrame({"value": [0, 1, 2, 3, 4]}).to_csv(self.x_path, index=False)
        pd.DataFrame(
            {
                "datetime": [
                    "2024-01-01T00:00:00",
                    "2024-01-01T01:00:00",
                    "2024-01-01T02:00:00",
                    "2024-01-01T03:00:00",
                    "2024-01-01T04:00:00",
                ]
            }
        ).to_csv(self.dt_path, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build(self, covariate_window_mode: str) -> TimeSeriesDataBuilder:
        return TimeSeriesDataBuilder(
            x_file=self.x_path,
            date_time_file=self.dt_path,
            input_seq_len=2,
            output_seq_len=1,
            stride=1,
            time_covariates=["hour_of_day"],
            covariate_window_mode=covariate_window_mode,
            order_mode="by_window",
            scale_method=None,
        )

    def test_last_step_covariates_keep_only_final_step(self):
        data = self._build("last_step")
        x, y = data.dataset["value"]

        self.assertEqual(x.shape, (3, 3))
        np.testing.assert_allclose(x[0], np.array([0, 1, 1], dtype=np.float32))
        np.testing.assert_allclose(y[:, 0], np.array([2, 3, 4], dtype=np.float32))

    def test_all_steps_covariates_are_interleaved_per_step(self):
        data = self._build("all_steps")
        x, y = data.dataset["value"]

        self.assertEqual(x.shape, (3, 4))
        np.testing.assert_allclose(x[0], np.array([0, 0, 1, 1], dtype=np.float32))
        np.testing.assert_allclose(y[:, 0], np.array([2, 3, 4], dtype=np.float32))

    def test_prepare_input_updates_only_target_slots_for_all_steps(self):
        data = self._build("all_steps")
        x, _ = data.dataset["value"]

        x_batch = x[:2].copy()
        x_original = x_batch.copy()
        var_x = np.zeros_like(x_batch)
        look_back_mu = np.array([[10, 11]], dtype=np.float32)
        look_back_var = np.array([[0.1, 0.2]], dtype=np.float32)
        indices = np.array([0, -1], dtype=np.int32)

        x_updated, var_updated = prepare_input(
            x=x_batch,
            var_x=var_x,
            look_back_mu=look_back_mu,
            look_back_var=look_back_var,
            indices=indices,
        )

        np.testing.assert_allclose(
            x_updated.reshape(2, 4)[0],
            np.array([10, 0, 11, 1], dtype=np.float32),
        )
        np.testing.assert_allclose(
            var_updated.reshape(2, 4)[0],
            np.array([0.1, 1e-6, 0.2, 1e-6], dtype=np.float32),
        )
        np.testing.assert_allclose(x_updated.reshape(2, 4)[1], x_batch[1])
        np.testing.assert_allclose(
            extract_target_history(x_original, input_seq_len=2),
            np.array([[0, 1], [1, 2]], dtype=np.float32),
        )

    def test_config_input_size_uses_look_back_covariates_and_embeddings(self):
        config = Config.model_validate(
            {
                "data": {
                    "loader": {
                        "look_back_len": 4,
                        "time_covariates": ["hour_of_day", "day_of_week"],
                        "input_seq_len": 4,
                        "covariate_window_mode": "all_steps",
                    }
                }
            }
        )

        self.assertEqual(config.input_size, 6)

    def test_sequential_model_input_size_is_step_feature_count(self):
        config = Config.model_validate(
            {
                "data": {
                    "loader": {
                        "look_back_len": 1,
                        "num_features": 3,
                        "input_seq_len": 4,
                        "covariate_window_mode": "all_steps",
                    }
                },
                "model": {"sequential_model": True},
                "embeddings": {"standard": {"embedding_size": 5}},
            }
        )

        self.assertEqual(config.input_size, 7)
        self.assertEqual(config.window_len, 4)

    def test_sequential_model_rejects_look_back_len_greater_than_one(self):
        config = Config.model_validate(
            {
                "data": {
                    "loader": {
                        "look_back_len": 2,
                        "input_seq_len": 4,
                        "covariate_window_mode": "all_steps",
                    }
                },
                "model": {"sequential_model": True},
            }
        )

        with self.assertRaises(ValueError):
            _ = config.input_size

    def test_non_sequential_uses_look_back_len_for_window_and_input_size(self):
        config = Config.model_validate(
            {
                "data": {
                    "loader": {
                        "look_back_len": 6,
                        "num_features": 3,
                        "input_seq_len": 2,
                        "covariate_window_mode": "last_step",
                    }
                },
                "model": {"sequential_model": False},
            }
        )

        self.assertEqual(config.window_len, 6)
        self.assertEqual(config.input_size, 7)

    def test_prepare_input_repeats_embeddings_for_sequential_model(self):
        class DummyEmbeddings:
            def __call__(self, indices):
                mu = np.stack(
                    [indices.astype(np.float32), indices.astype(np.float32) + 10],
                    axis=1,
                )
                var = np.ones_like(mu, dtype=np.float32) * 0.5
                return mu, var

        x = np.array([[0, 0, 1, 1]], dtype=np.float32)
        var_x = np.zeros_like(x)

        x_updated, var_updated = prepare_input(
            x=x,
            var_x=var_x,
            look_back_mu=None,
            look_back_var=None,
            indices=np.array([2], dtype=np.int32),
            embeddings=DummyEmbeddings(),
            input_seq_len=2,
            sequential_model=True,
        )

        np.testing.assert_allclose(
            x_updated.reshape(1, 8)[0],
            np.array([0, 0, 2, 12, 1, 1, 2, 12], dtype=np.float32),
        )
        np.testing.assert_allclose(
            var_updated.reshape(1, 8)[0],
            np.array([1e-6, 1e-6, 0.5, 0.5, 1e-6, 1e-6, 0.5, 0.5], dtype=np.float32),
        )

    def test_series_ids_match_ts_to_use_values(self):
        x_path = os.path.join(self.tmpdir.name, "x_multi.csv")
        dt_path = os.path.join(self.tmpdir.name, "dt_multi.csv")

        pd.DataFrame(
            {
                "s0": [0, 1, 2, 3, 4],
                "s1": [10, 11, 12, 13, 14],
                "s2": [20, 21, 22, 23, 24],
                "s3": [30, 31, 32, 33, 34],
            }
        ).to_csv(x_path, index=False)
        pd.DataFrame({"datetime": [f"2024-01-01T0{i}:00:00" for i in range(5)]}).to_csv(
            dt_path, index=False
        )

        data = TimeSeriesDataBuilder(
            x_file=x_path,
            date_time_file=dt_path,
            input_seq_len=2,
            output_seq_len=1,
            stride=1,
            time_covariates=[],
            covariate_window_mode="last_step",
            order_mode="by_window",
            scale_method=None,
            ts_to_use=[1, 3],
        )

        self.assertSetEqual(set(np.unique(data.dataset["series_id"]).tolist()), {1, 3})

    def test_config_allows_missing_top_level_sections(self):
        config = Config.model_validate(
            {
                "data": {
                    "loader": {
                        "input_seq_len": 3,
                    }
                }
            }
        )

        self.assertEqual(config.model.hidden_sizes, [40, 40])
        self.assertEqual(config.training.num_epochs, 100)
        self.assertTrue(config.use_AGVI)
        self.assertEqual(config.total_embedding_size, 0)

    def test_to_yaml_can_exclude_defaults(self):
        config = Config.model_validate(
            {
                "seed": 123,
                "data": {"loader": {"input_seq_len": 1}},
            }
        )
        out_path = os.path.join(self.tmpdir.name, "cfg.yaml")
        config.to_yaml(out_path)

        with open(out_path, "r") as f:
            dumped = yaml.safe_load(f)

        self.assertEqual(dumped["seed"], 123)
        self.assertEqual(dumped["data"]["loader"]["input_seq_len"], 1)
        self.assertNotIn("training", dumped)
        self.assertNotIn("model", dumped)


if __name__ == "__main__":
    unittest.main()

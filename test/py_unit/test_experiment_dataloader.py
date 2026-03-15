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
    prepare_data,
    split_full_dataset,
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

    def test_test_split_context_falls_back_to_train_when_val_is_short(self):
        train_x = os.path.join(self.tmpdir.name, "train_x.csv")
        train_dt = os.path.join(self.tmpdir.name, "train_dt.csv")
        val_x = os.path.join(self.tmpdir.name, "val_x.csv")
        val_dt = os.path.join(self.tmpdir.name, "val_dt.csv")
        test_x = os.path.join(self.tmpdir.name, "test_x.csv")
        test_dt = os.path.join(self.tmpdir.name, "test_dt.csv")

        pd.DataFrame({"value": [1, 2, 3, 4, 5]}).to_csv(train_x, index=False)
        pd.DataFrame(
            {"datetime": [f"2024-01-01T0{i}:00:00" for i in range(5)]}
        ).to_csv(train_dt, index=False)
        pd.DataFrame({"value": [6]}).to_csv(val_x, index=False)
        pd.DataFrame({"datetime": ["2024-01-01T05:00:00"]}).to_csv(val_dt, index=False)
        pd.DataFrame({"value": [7, 8]}).to_csv(test_x, index=False)
        pd.DataFrame({"datetime": ["2024-01-01T06:00:00", "2024-01-01T07:00:00"]}).to_csv(
            test_dt, index=False
        )

        _, _, test_data = prepare_data(
            x_file=[train_x, val_x, test_x],
            date_file=[train_dt, val_dt, test_dt],
            input_seq_len=4,
            carry_split_context=True,
            time_covariates=[],
            covariate_window_mode="last_step",
            scale_method=None,
            order_mode="by_window",
            ts_to_use=None,
        )

        x_test, y_test = test_data.dataset["value"]
        self.assertGreater(len(x_test), 0)
        np.testing.assert_allclose(x_test[0], np.array([3, 4, 5, 6], dtype=np.float32))
        np.testing.assert_allclose(y_test[:, 0], np.array([7, 8], dtype=np.float32))

    def test_split_full_dataset_trims_trailing_nans_and_adds_context(self):
        full_x = os.path.join(self.tmpdir.name, "full_x.csv")
        full_dt = os.path.join(self.tmpdir.name, "full_dt.csv")

        pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, np.nan, np.nan]}).to_csv(
            full_x, index=False
        )
        pd.DataFrame(
            {"datetime": [f"2024-01-01T0{i}:00:00" for i in range(9)]}
        ).to_csv(full_dt, index=False)

        split = split_full_dataset(
            x_full_file=full_x,
            date_full_file=full_dt,
            input_seq_len=2,
            carry_split_context=True,
            train_ratio=0.5,
            val_ratio=0.25,
            ts_to_use=None,
        )

        train = split["train_x"][:, 0]
        val = split["val_x"][:, 0]
        test = split["test_x"][:, 0]
        truth_val = split["truth_val_x"][:, 0]
        truth_test = split["truth_test_x"][:, 0]

        np.testing.assert_allclose(train[~np.isnan(train)], np.array([1, 2, 3], dtype=np.float32))
        np.testing.assert_allclose(
            val[~np.isnan(val)], np.array([2, 3, 4], dtype=np.float32)
        )
        np.testing.assert_allclose(
            test[~np.isnan(test)], np.array([3, 4, 5, 6, 7], dtype=np.float32)
        )
        np.testing.assert_allclose(
            truth_val[~np.isnan(truth_val)], np.array([4], dtype=np.float32)
        )
        np.testing.assert_allclose(
            truth_test[~np.isnan(truth_test)], np.array([5, 6, 7], dtype=np.float32)
        )

    def test_split_full_dataset_keeps_val_test_fixed_when_train_use_ratio_changes(self):
        full_x = os.path.join(self.tmpdir.name, "full_x_ratio.csv")
        full_dt = os.path.join(self.tmpdir.name, "full_dt_ratio.csv")

        pd.DataFrame({"value": np.arange(1, 13, dtype=np.float32)}).to_csv(
            full_x, index=False
        )
        pd.DataFrame(
            {"datetime": [f"2024-01-01T{i:02d}:00:00" for i in range(12)]}
        ).to_csv(full_dt, index=False)

        split_full = split_full_dataset(
            x_full_file=full_x,
            date_full_file=full_dt,
            input_seq_len=3,
            carry_split_context=True,
            train_ratio=0.6,
            val_ratio=0.2,
            train_use_ratio=1.0,
            ts_to_use=None,
        )
        split_half = split_full_dataset(
            x_full_file=full_x,
            date_full_file=full_dt,
            input_seq_len=3,
            carry_split_context=True,
            train_ratio=0.6,
            val_ratio=0.2,
            train_use_ratio=0.5,
            ts_to_use=None,
        )

        train_full = split_full["train_x"][:, 0]
        train_half = split_half["train_x"][:, 0]
        val_full = split_full["val_x"][:, 0]
        val_half = split_half["val_x"][:, 0]
        test_full = split_full["test_x"][:, 0]
        test_half = split_half["test_x"][:, 0]
        truth_val_full = split_full["truth_val_x"][:, 0]
        truth_val_half = split_half["truth_val_x"][:, 0]
        truth_test_full = split_full["truth_test_x"][:, 0]
        truth_test_half = split_half["truth_test_x"][:, 0]

        self.assertLess(np.sum(~np.isnan(train_half)), np.sum(~np.isnan(train_full)))
        np.testing.assert_allclose(
            train_half[~np.isnan(train_half)],
            np.array([5, 6, 7], dtype=np.float32),
        )
        np.testing.assert_allclose(
            truth_val_full[~np.isnan(truth_val_full)],
            truth_val_half[~np.isnan(truth_val_half)],
        )
        np.testing.assert_allclose(
            truth_test_full[~np.isnan(truth_test_full)],
            truth_test_half[~np.isnan(truth_test_half)],
        )
        np.testing.assert_allclose(
            val_full[~np.isnan(val_full)],
            val_half[~np.isnan(val_half)],
        )
        np.testing.assert_allclose(
            test_full[~np.isnan(test_full)],
            test_half[~np.isnan(test_half)],
        )

    def test_split_full_dataset_handles_all_nan_series(self):
        full_x = os.path.join(self.tmpdir.name, "full_x_all_nan.csv")
        full_dt = os.path.join(self.tmpdir.name, "full_dt_all_nan.csv")

        pd.DataFrame(
            {
                "all_nan": [np.nan, np.nan, np.nan, np.nan],
                "valid": [1.0, 2.0, 3.0, 4.0],
            }
        ).to_csv(full_x, index=False)
        pd.DataFrame(
            {
                "d0": [f"2024-01-01T0{i}:00:00" for i in range(4)],
                "d1": [f"2024-01-01T0{i}:00:00" for i in range(4)],
            }
        ).to_csv(full_dt, index=False)

        split = split_full_dataset(
            x_full_file=full_x,
            date_full_file=full_dt,
            input_seq_len=2,
            carry_split_context=True,
            train_ratio=0.5,
            val_ratio=0.25,
            ts_to_use=None,
        )

        nan_series_train = split["train_x"][:, 0]
        valid_series_train = split["train_x"][:, 1]
        self.assertTrue(np.all(np.isnan(nan_series_train)))
        self.assertGreater(np.sum(~np.isnan(valid_series_train)), 0)


if __name__ == "__main__":
    unittest.main()

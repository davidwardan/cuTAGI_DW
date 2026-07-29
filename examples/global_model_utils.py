"""Small utilities for the global stateful LSTM example.

This module intentionally supports one data layout and one batching strategy:
a CSV whose columns are time series, batched with ``by_series_batch``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from pytagi import manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential


Dataset = dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]]


@dataclass
class DataSplit:
    """One chronological split and its rolling-window dataset."""

    name: str
    values: np.ndarray
    dataset: Dataset
    means: np.ndarray
    stds: np.ndarray


@dataclass
class SeriesBatch:
    """A single time step for a fixed group of series."""

    x: np.ndarray
    y: np.ndarray
    series_ids: np.ndarray
    time_ids: np.ndarray
    starts_new_group: bool


class EarlyStopping:
    """Keep the model with the lowest validation RMSE."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = np.inf
        self.best_state = None
        self.epochs_without_improvement = 0

    def update(self, score: float, model: Sequential) -> bool:
        """Record an improvement and return True when training should stop."""
        if np.isfinite(score) and score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best(self, model: Sequential) -> None:
        if self.best_state is None:
            raise RuntimeError(
                "Early stopping did not observe a finite validation score."
            )
        model.load_state_dict(self.best_state)


def load_values(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a headered CSV with one time series per column."""
    path = Path(path)
    column_names = np.genfromtxt(path, delimiter=",", max_rows=1, dtype=str)
    values = np.genfromtxt(
        path, delimiter=",", skip_header=1, dtype=np.float32, ndmin=2
    )
    column_names = np.atleast_1d(column_names)
    if values.shape[1] != len(column_names):
        raise ValueError(
            f"{path} has {values.shape[1]} data columns but "
            f"{len(column_names)} column names."
        )
    return values, column_names


def split_values(
    values: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split every series, ignoring trailing NaN padding."""
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train_ratio and validation_ratio must be positive.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    split_columns: list[list[np.ndarray]] = [[], [], []]
    for series in values.T:
        series = _trim_trailing_nans(series)
        train_end = int(len(series) * train_ratio)
        validation_end = train_end + int(len(series) * validation_ratio)

        if (
            train_end == 0
            or validation_end == train_end
            or validation_end == len(series)
        ):
            raise ValueError(
                "Every series must contain at least one value in each split. "
                "Use more data or adjust the split ratios."
            )

        split_columns[0].append(series[:train_end])
        split_columns[1].append(series[train_end:validation_end])
        split_columns[2].append(series[validation_end:])

    return tuple(_pad_columns(columns) for columns in split_columns)


def prepare_data(
    values_file: str | Path,
    train_ratio: float,
    validation_ratio: float,
    lookback: int,
) -> tuple[DataSplit, DataSplit, DataSplit, np.ndarray]:
    """Load, split, standardize, and window the time series."""
    values, column_names = load_values(values_file)
    train_values, validation_values, test_values = split_values(
        values, train_ratio, validation_ratio
    )

    means = np.nanmean(train_values, axis=0).astype(np.float32)
    stds = np.nanstd(train_values, axis=0).astype(np.float32)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0).astype(np.float32)

    train = _make_split("train", train_values, means, stds, lookback, history=None)
    validation = _make_split(
        "validation",
        validation_values,
        means,
        stds,
        lookback,
        history=train_values,
    )
    test_history = _concat_columns(train_values, validation_values)
    test = _make_split("test", test_values, means, stds, lookback, history=test_history)
    return train, validation, test, column_names


def by_series_batch(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> Generator[SeriesBatch, None, None]:
    """Yield fixed groups of series, advancing one window at a time.

    Short series and an incomplete final group are padded with NaNs. Their
    ``series_ids`` and ``time_ids`` are set to ``-1``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    x, y = dataset["value"]
    series_ids = dataset["series_id"]
    time_ids = dataset["time_id"]
    if len(x) == 0:
        return

    unique_ids = np.unique(series_ids)
    if shuffle:
        np.random.default_rng(seed).shuffle(unique_ids)

    series_rows = {
        series_id: np.flatnonzero(series_ids == series_id) for series_id in unique_ids
    }

    for group_start in range(0, len(unique_ids), batch_size):
        group_ids = unique_ids[group_start : group_start + batch_size]
        group_rows = [series_rows[series_id] for series_id in group_ids]
        group_length = max(len(rows) for rows in group_rows)

        for step in range(group_length):
            x_batch = np.full((batch_size, x.shape[1]), np.nan, dtype=np.float32)
            y_batch = np.full((batch_size, y.shape[1]), np.nan, dtype=np.float32)
            series_batch = np.full(batch_size, -1, dtype=np.int32)
            time_batch = np.full(batch_size, -1, dtype=np.int32)

            for slot, rows in enumerate(group_rows):
                if step >= len(rows):
                    continue
                row = rows[step]
                x_batch[slot] = x[row]
                y_batch[slot] = y[row]
                series_batch[slot] = series_ids[row]
                time_batch[slot] = time_ids[row]

            yield SeriesBatch(
                x=x_batch,
                y=y_batch,
                series_ids=series_batch,
                time_ids=time_batch,
                starts_new_group=step == 0,
            )


def build_model(
    input_size: int,
    hidden_sizes: tuple[int, ...],
    seed: int,
    device: str,
    cpu_threads: int | None = None,
) -> tuple[Sequential, OutputUpdater]:
    """Build a stateful LSTM followed by a one-value output layer."""
    if not hidden_sizes:
        raise ValueError("hidden_sizes must contain at least one layer size.")

    manual_seed(seed)
    layers = []
    layer_input_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(
            LSTM(layer_input_size, hidden_size, last_timestep=True, seq_len=1)
        )
        layer_input_size = hidden_size
    layers.append(Linear(layer_input_size, 1))

    model = Sequential(*layers)
    if device == "cuda":
        model.to_device("cuda")
    elif device == "cpu":
        if cpu_threads is not None:
            model.set_threads(cpu_threads)
    else:
        raise ValueError("device must be 'cpu' or 'cuda'.")

    return model, OutputUpdater(model.device)


def prepare_inputs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN padding in model inputs and flatten the batch."""
    means = np.nan_to_num(x, nan=0.0).astype(np.float32, copy=False).reshape(-1)
    variances = np.zeros_like(means, dtype=np.float32)
    return means, variances


def update_model(
    model: Sequential,
    output_updater: OutputUpdater,
    targets: np.ndarray,
    observation_variance: float,
) -> None:
    """Apply one TAGI parameter update; NaN targets are ignored by the updater."""
    target_vector = targets.astype(np.float32, copy=False).reshape(-1)
    target_variance = np.full(
        target_vector.shape, observation_variance, dtype=np.float32
    )
    output_updater.update(
        output_states=model.output_z_buffer,
        mu_obs=target_vector,
        var_obs=target_variance,
        delta_states=model.input_delta_z_buffer,
    )
    model.backward()
    model.step()


def predictions_to_original_scale(
    split: DataSplit,
    batches: list[tuple[SeriesBatch, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Place predictions in arrays aligned with the original split."""
    means = np.full_like(split.values, np.nan, dtype=np.float32)
    stds = np.full_like(split.values, np.nan, dtype=np.float32)

    for batch, predicted_means, predicted_variances in batches:
        active = batch.series_ids >= 0
        series_ids = batch.series_ids[active]
        time_ids = batch.time_ids[active]
        scale = split.stds[series_ids]

        means[time_ids, series_ids] = (
            predicted_means[active] * scale + split.means[series_ids]
        )
        stds[time_ids, series_ids] = (
            np.sqrt(np.maximum(predicted_variances[active], 0.0)) * scale
        )

    return means, stds


def _make_split(
    name: str,
    values: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    lookback: int,
    history: np.ndarray | None,
) -> DataSplit:
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    x_rows, y_rows, series_rows, time_rows = [], [], [], []
    for series_id in range(values.shape[1]):
        current = _trim_trailing_nans(values[:, series_id])
        previous = (
            np.empty(0, dtype=np.float32)
            if history is None
            else _trim_trailing_nans(history[:, series_id])
        )

        current_scaled = (current - means[series_id]) / stds[series_id]
        previous_scaled = (previous - means[series_id]) / stds[series_id]
        previous_scaled = previous_scaled[-lookback:]
        combined = np.concatenate((previous_scaled, current_scaled))
        history_length = len(previous_scaled)

        for time_id in range(len(current_scaled)):
            target_position = history_length + time_id
            if target_position < lookback:
                continue
            x_rows.append(combined[target_position - lookback : target_position])
            y_rows.append([current_scaled[time_id]])
            series_rows.append(series_id)
            time_rows.append(time_id)

    dataset: Dataset = {
        "value": (
            np.asarray(x_rows, dtype=np.float32).reshape(-1, lookback),
            np.asarray(y_rows, dtype=np.float32).reshape(-1, 1),
        ),
        "series_id": np.asarray(series_rows, dtype=np.int32),
        "time_id": np.asarray(time_rows, dtype=np.int32),
    }
    return DataSplit(name, values, dataset, means, stds)


def _trim_trailing_nans(series: np.ndarray) -> np.ndarray:
    valid_positions = np.flatnonzero(~np.isnan(series))
    if len(valid_positions) == 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(series[: valid_positions[-1] + 1], dtype=np.float32)


def _pad_columns(columns: list[np.ndarray]) -> np.ndarray:
    max_length = max(len(column) for column in columns)
    output = np.full((max_length, len(columns)), np.nan, dtype=np.float32)
    for column_id, column in enumerate(columns):
        output[: len(column), column_id] = column
    return output


def _concat_columns(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    columns = []
    for series_id in range(first.shape[1]):
        columns.append(
            np.concatenate(
                (
                    _trim_trailing_nans(first[:, series_id]),
                    _trim_trailing_nans(second[:, series_id]),
                )
            )
        )
    return _pad_columns(columns)

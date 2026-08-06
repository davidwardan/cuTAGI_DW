"""Small utilities for the global stateful LSTM example.

This module intentionally supports one data layout and one batching strategy:
paired value/datetime CSVs whose columns are time series, batched with
``by_series_batch``.
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
PREDICTION_MODES = ("one_step_ahead", "multi_step_ahead")


@dataclass
class DataSplit:
    """One chronological split and its rolling-window dataset."""

    name: str
    values: np.ndarray
    datetimes: np.ndarray
    dataset: Dataset
    means: np.ndarray
    stds: np.ndarray


@dataclass
class SeriesBatch:
    """A single time step for a fixed group of series."""

    x: np.ndarray
    time_covariates: np.ndarray
    y: np.ndarray
    series_ids: np.ndarray
    time_ids: np.ndarray
    starts_new_group: bool


class ForecastLookback:
    """Rolling mean and variance used during validation and testing."""

    def __init__(self, initial_means: np.ndarray) -> None:
        self.means = initial_means.astype(np.float32, copy=True)
        self.variances = np.zeros_like(self.means, dtype=np.float32)

    def update(
        self,
        prior_means: np.ndarray,
        prior_variances: np.ndarray,
        targets: np.ndarray,
        active: np.ndarray,
        mode: str,
        observation_variance: float,
    ) -> None:
        """Append either the posterior or prior prediction to the window."""
        if mode not in PREDICTION_MODES:
            raise ValueError(
                f"mode must be one of {PREDICTION_MODES}; received {mode!r}."
            )
        if observation_variance <= 0:
            raise ValueError("observation_variance must be positive.")

        prior_means = np.asarray(prior_means, dtype=np.float32).reshape(-1)
        prior_variances = np.maximum(
            np.asarray(prior_variances, dtype=np.float32).reshape(-1), 0.0
        )
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        active = np.asarray(active, dtype=bool).reshape(-1)

        if mode == "one_step_ahead":
            posterior_means = prior_means.copy()
            posterior_variances = prior_variances.copy()
            observed = active & np.isfinite(targets)
            kalman_gain = prior_variances[observed] / (
                prior_variances[observed] + observation_variance
            )
            posterior_means[observed] += kalman_gain * (
                targets[observed] - prior_means[observed]
            )
            posterior_variances[observed] *= 1.0 - kalman_gain
            lookback_means = posterior_means
            lookback_variances = posterior_variances
        else:
            lookback_means = prior_means
            lookback_variances = prior_variances

        self.means = np.roll(self.means, -1, axis=1)
        self.variances = np.roll(self.variances, -1, axis=1)
        self.means[:, -1] = np.where(active, lookback_means, np.nan)
        self.variances[:, -1] = np.where(active, lookback_variances, 0.0)


class EarlyStopping:
    """Keep the model with the best validation score."""

    def __init__(
        self,
        patience: int,
        min_delta: float,
        mode: str = "min",
        warmup_epochs: int = 0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.epochs_seen = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        self.best_state = None
        self.best_sigma_v = None
        self.epochs_without_improvement = 0

    def update(self, score: float, model: Sequential, sigma_v: float) -> bool:
        """Record an improvement and return True when training should stop."""
        self.epochs_seen += 1
        if self.epochs_seen <= self.warmup_epochs:
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )
        if np.isfinite(score) and improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_sigma_v = float(sigma_v)
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best(self, model: Sequential) -> float:
        if self.best_state is None or self.best_sigma_v is None:
            raise RuntimeError(
                "Early stopping did not observe a finite validation score."
            )
        model.load_state_dict(self.best_state)
        return self.best_sigma_v


def gaussian_log_likelihood_terms(
    targets: np.ndarray,
    predicted_means: np.ndarray,
    predicted_variances: np.ndarray,
) -> np.ndarray:
    """Return elementwise Gaussian log-likelihoods, with invalid entries as NaN."""
    targets = np.asarray(targets, dtype=np.float64)
    predicted_means = np.asarray(predicted_means, dtype=np.float64)
    predicted_variances = np.asarray(predicted_variances, dtype=np.float64)
    if not (targets.shape == predicted_means.shape == predicted_variances.shape):
        raise ValueError("Targets, means, and variances must have identical shapes.")

    valid = (
        np.isfinite(targets)
        & np.isfinite(predicted_means)
        & np.isfinite(predicted_variances)
        & (predicted_variances > 0)
    )
    terms = np.full(targets.shape, np.nan, dtype=np.float64)
    errors = targets[valid] - predicted_means[valid]
    terms[valid] = -0.5 * (
        np.log(2.0 * np.pi * predicted_variances[valid])
        + errors**2 / predicted_variances[valid]
    )
    return terms


def macro_average_series_log_likelihood(
    log_likelihoods: np.ndarray,
    series_ids: np.ndarray,
) -> float:
    """Sum log-likelihood within each series, then average across series."""
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64).reshape(-1)
    series_ids = np.asarray(series_ids, dtype=np.int64).reshape(-1)
    if log_likelihoods.shape != series_ids.shape:
        raise ValueError("Log-likelihoods and series IDs must have identical shapes.")

    valid = np.isfinite(log_likelihoods) & (series_ids >= 0)
    if not np.any(valid):
        raise RuntimeError("There are no usable series log-likelihood values.")

    _, inverse_ids = np.unique(series_ids[valid], return_inverse=True)
    series_sums = np.bincount(inverse_ids, weights=log_likelihoods[valid])
    return float(np.mean(series_sums))


def sigma_v_schedule(
    num_epochs: int,
    start: float,
    end: float,
    decay_factor: float,
) -> np.ndarray:
    """Build the exponential observation-noise schedule used in experiments."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if start <= 0 or end <= 0:
        raise ValueError("sigma_v start and end values must be positive.")
    if not 0 < decay_factor <= 1:
        raise ValueError("decay_factor must be in the interval (0, 1].")

    epochs = np.arange(num_epochs, dtype=np.float32)
    return end + (start - end) * (decay_factor**epochs)


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


def load_datetimes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a headered datetime CSV with one time series per column."""
    path = Path(path)
    column_names = np.genfromtxt(path, delimiter=",", max_rows=1, dtype=str)
    raw_datetimes = np.genfromtxt(
        path, delimiter=",", skip_header=1, dtype=str, ndmin=2
    )
    column_names = np.atleast_1d(column_names)
    if raw_datetimes.shape[1] != len(column_names):
        raise ValueError(
            f"{path} has {raw_datetimes.shape[1]} data columns but "
            f"{len(column_names)} column names."
        )

    try:
        datetimes = raw_datetimes.astype("datetime64[ns]")
    except ValueError as error:
        raise ValueError(f"{path} contains an invalid datetime value.") from error
    return datetimes, column_names


def load_time_series_files(
    values_path: str | Path,
    datetime_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate paired value and datetime CSVs."""
    values, value_columns = load_values(values_path)
    datetimes, datetime_columns = load_datetimes(datetime_path)

    if not np.array_equal(value_columns, datetime_columns):
        raise ValueError(
            "Values and datetime CSVs must have identical column names in the "
            "same order."
        )
    if values.shape != datetimes.shape:
        raise ValueError(
            "Values and datetime CSVs must have identical shapes; received "
            f"{values.shape} and {datetimes.shape}."
        )

    missing_dates = np.isfinite(values) & np.isnat(datetimes)
    if np.any(missing_dates):
        row, column = np.argwhere(missing_dates)[0]
        raise ValueError(
            "Every finite value must have a datetime. Missing datetime for "
            f"column {value_columns[column]!r} at data row {row + 1}."
        )
    return values, datetimes, value_columns


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


def split_values_and_datetimes(
    values: np.ndarray,
    datetimes: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Chronologically split paired values and dates using identical cuts."""
    if values.shape != datetimes.shape:
        raise ValueError("values and datetimes must have identical shapes.")
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("train_ratio and validation_ratio must be positive.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    value_columns: list[list[np.ndarray]] = [[], [], []]
    datetime_columns: list[list[np.ndarray]] = [[], [], []]
    for series, series_datetimes in zip(values.T, datetimes.T):
        series, series_datetimes = _trim_pair(series, series_datetimes)
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

        cuts = (0, train_end, validation_end, len(series))
        for split_id, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            value_columns[split_id].append(series[start:end])
            datetime_columns[split_id].append(series_datetimes[start:end])

    return tuple(
        (
            _pad_columns(value_split),
            _pad_datetime_columns(datetime_split),
        )
        for value_split, datetime_split in zip(value_columns, datetime_columns)
    )


def prepare_data(
    values_file: str | Path,
    datetime_file: str | Path,
    train_ratio: float,
    validation_ratio: float,
    lookback: int,
    time_covariates: tuple[str, ...] = (),
) -> tuple[DataSplit, DataSplit, DataSplit, np.ndarray]:
    """Load, split, standardize, and window paired values and datetimes."""
    values, datetimes, column_names = load_time_series_files(values_file, datetime_file)
    train, validation, test = split_values_and_datetimes(
        values, datetimes, train_ratio, validation_ratio
    )
    train_values, train_datetimes = train
    validation_values, validation_datetimes = validation
    test_values, test_datetimes = test

    means = np.nanmean(train_values, axis=0).astype(np.float32)
    stds = np.nanstd(train_values, axis=0).astype(np.float32)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0).astype(np.float32)

    covariate_means, covariate_stds = _covariate_scaling(
        train_values, train_datetimes, time_covariates
    )

    train = _make_split(
        "train",
        train_values,
        train_datetimes,
        means,
        stds,
        covariate_means,
        covariate_stds,
        time_covariates,
        lookback,
        history=None,
        history_datetimes=None,
    )
    validation = _make_split(
        "validation",
        validation_values,
        validation_datetimes,
        means,
        stds,
        covariate_means,
        covariate_stds,
        time_covariates,
        lookback,
        history=train_values,
        history_datetimes=train_datetimes,
    )
    test_history = _concat_columns(train_values, validation_values)
    test_datetime_history = _concat_datetime_columns(
        train_datetimes, validation_datetimes
    )
    test = _make_split(
        "test",
        test_values,
        test_datetimes,
        means,
        stds,
        covariate_means,
        covariate_stds,
        time_covariates,
        lookback,
        history=test_history,
        history_datetimes=test_datetime_history,
    )
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
    time_covariates = dataset["time_covariates"]
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
            covariate_batch = np.full(
                (batch_size, time_covariates.shape[1]), np.nan, dtype=np.float32
            )
            y_batch = np.full((batch_size, y.shape[1]), np.nan, dtype=np.float32)
            series_batch = np.full(batch_size, -1, dtype=np.int32)
            time_batch = np.full(batch_size, -1, dtype=np.int32)

            for slot, rows in enumerate(group_rows):
                if step >= len(rows):
                    continue
                row = rows[step]
                x_batch[slot] = x[row]
                covariate_batch[slot] = time_covariates[row]
                y_batch[slot] = y[row]
                series_batch[slot] = series_ids[row]
                time_batch[slot] = time_ids[row]

            yield SeriesBatch(
                x=x_batch,
                time_covariates=covariate_batch,
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


def prepare_inputs(
    x: np.ndarray,
    variances: np.ndarray | None = None,
    time_covariates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Append known calendar inputs, replace padding, and flatten each batch."""
    if time_covariates is not None:
        if time_covariates.ndim != 2 or time_covariates.shape[0] != x.shape[0]:
            raise ValueError(
                "Time covariates must be a 2D array with the same batch size as x."
            )
        input_means = np.concatenate((x, time_covariates), axis=1)
    else:
        input_means = x

    means = (
        np.nan_to_num(input_means, nan=0.0).astype(np.float32, copy=False).reshape(-1)
    )
    if variances is None:
        input_variances = np.zeros_like(means, dtype=np.float32)
    else:
        if variances.shape != x.shape:
            raise ValueError("Input means and variances must have the same shape.")
        if time_covariates is not None:
            variances = np.concatenate(
                (
                    variances,
                    np.zeros_like(time_covariates, dtype=np.float32),
                ),
                axis=1,
            )
        input_variances = np.nan_to_num(variances, nan=0.0, posinf=2.0, neginf=0.0)
        input_variances = np.clip(input_variances, 0.0, 2.0)
        input_variances = input_variances.astype(np.float32, copy=False).reshape(-1)
    return means, input_variances


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
    datetimes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    covariate_means: np.ndarray,
    covariate_stds: np.ndarray,
    time_covariates: tuple[str, ...],
    lookback: int,
    history: np.ndarray | None,
    history_datetimes: np.ndarray | None,
) -> DataSplit:
    if lookback <= 0:
        raise ValueError("lookback must be positive.")

    x_rows, covariate_rows, y_rows, series_rows, time_rows = [], [], [], [], []
    for series_id in range(values.shape[1]):
        current, current_datetimes = _trim_pair(
            values[:, series_id], datetimes[:, series_id]
        )
        if history is None or history_datetimes is None:
            previous = np.empty(0, dtype=np.float32)
            previous_datetimes = np.empty(0, dtype="datetime64[ns]")
        else:
            previous, previous_datetimes = _trim_pair(
                history[:, series_id], history_datetimes[:, series_id]
            )

        current_scaled = (current - means[series_id]) / stds[series_id]
        previous_scaled = (previous - means[series_id]) / stds[series_id]
        previous_scaled = previous_scaled[-lookback:]
        previous_datetimes = previous_datetimes[-lookback:]
        combined = np.concatenate((previous_scaled, current_scaled))
        combined_datetimes = np.concatenate((previous_datetimes, current_datetimes))
        history_length = len(previous_scaled)

        for time_id in range(len(current_scaled)):
            target_position = history_length + time_id
            if target_position < lookback:
                continue
            x_rows.append(combined[target_position - lookback : target_position])
            raw_covariates = _datetime_covariates(
                combined_datetimes[target_position - 1 : target_position],
                time_covariates,
            )[0]
            covariate_rows.append(
                (raw_covariates - covariate_means[series_id])
                / covariate_stds[series_id]
            )
            y_rows.append([current_scaled[time_id]])
            series_rows.append(series_id)
            time_rows.append(time_id)

    if time_covariates:
        covariate_array = np.asarray(covariate_rows, dtype=np.float32).reshape(
            -1, len(time_covariates)
        )
    else:
        covariate_array = np.empty((len(x_rows), 0), dtype=np.float32)

    dataset: Dataset = {
        "value": (
            np.asarray(x_rows, dtype=np.float32).reshape(-1, lookback),
            np.asarray(y_rows, dtype=np.float32).reshape(-1, 1),
        ),
        "time_covariates": covariate_array,
        "series_id": np.asarray(series_rows, dtype=np.int32),
        "time_id": np.asarray(time_rows, dtype=np.int32),
    }
    return DataSplit(name, values, datetimes, dataset, means, stds)


def _covariate_scaling(
    values: np.ndarray,
    datetimes: np.ndarray,
    time_covariates: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-series calendar scaling from the training split."""
    num_series = values.shape[1]
    if not time_covariates:
        empty = np.empty((num_series, 0), dtype=np.float32)
        return empty, empty

    means = np.empty((num_series, len(time_covariates)), dtype=np.float32)
    stds = np.empty_like(means)
    for series_id in range(num_series):
        _, series_datetimes = _trim_pair(values[:, series_id], datetimes[:, series_id])
        covariates = _datetime_covariates(series_datetimes, time_covariates)
        means[series_id] = np.nanmean(covariates, axis=0)
        stds[series_id] = np.nanstd(covariates, axis=0)

    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0).astype(np.float32)
    return means, stds


def _datetime_covariates(
    datetimes: np.ndarray,
    names: tuple[str, ...],
) -> np.ndarray:
    """Build one numeric calendar column for every requested covariate."""
    datetimes = np.asarray(datetimes, dtype="datetime64[ns]").reshape(-1)
    if not names:
        return np.empty((len(datetimes), 0), dtype=np.float32)

    valid = ~np.isnat(datetimes)
    days = datetimes.astype("datetime64[D]")
    day_numbers = days.astype(np.int64)
    columns = []
    for name in names:
        covariate = np.full(len(datetimes), np.nan, dtype=np.float32)
        if name == "hour_of_day":
            hours = datetimes.astype("datetime64[h]").astype(np.int64)
            covariate[valid] = hours[valid] % 24
        elif name == "day_of_week":
            covariate[valid] = (day_numbers[valid] + 3) % 7
        elif name == "week_of_year":
            # ISO weeks belong to the year containing their Thursday.
            weekday = (day_numbers + 3) % 7
            thursdays = days + (3 - weekday).astype("timedelta64[D]")
            iso_year_starts = thursdays.astype("datetime64[Y]")
            weeks = (thursdays - iso_year_starts).astype("timedelta64[D]").astype(
                int
            ) // 7 + 1
            covariate[valid] = weeks[valid]
        elif name == "month_of_year":
            months = datetimes.astype("datetime64[M]").astype(np.int64)
            covariate[valid] = months[valid] % 12 + 1
        elif name == "quarter_of_year":
            months = datetimes.astype("datetime64[M]").astype(np.int64)
            covariate[valid] = (months[valid] % 12) // 3 + 1
        else:
            raise ValueError(
                f"Unknown time covariate {name!r}. Supported values are "
                "'hour_of_day', 'day_of_week', 'week_of_year', "
                "'month_of_year', and 'quarter_of_year'."
            )
        columns.append(covariate)
    return np.stack(columns, axis=1)


def _trim_trailing_nans(series: np.ndarray) -> np.ndarray:
    valid_positions = np.flatnonzero(~np.isnan(series))
    if len(valid_positions) == 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(series[: valid_positions[-1] + 1], dtype=np.float32)


def _trim_pair(
    series: np.ndarray,
    datetimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim value padding and apply the identical cut to its datetimes."""
    valid_positions = np.flatnonzero(~np.isnan(series))
    if len(valid_positions) == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
        )
    end = valid_positions[-1] + 1
    return (
        np.asarray(series[:end], dtype=np.float32),
        np.asarray(datetimes[:end], dtype="datetime64[ns]"),
    )


def _pad_columns(columns: list[np.ndarray]) -> np.ndarray:
    max_length = max(len(column) for column in columns)
    output = np.full((max_length, len(columns)), np.nan, dtype=np.float32)
    for column_id, column in enumerate(columns):
        output[: len(column), column_id] = column
    return output


def _pad_datetime_columns(columns: list[np.ndarray]) -> np.ndarray:
    max_length = max(len(column) for column in columns)
    output = np.full(
        (max_length, len(columns)), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
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


def _concat_datetime_columns(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    columns = []
    for series_id in range(first.shape[1]):
        first_column = _trim_trailing_nats(first[:, series_id])
        second_column = _trim_trailing_nats(second[:, series_id])
        columns.append(np.concatenate((first_column, second_column)))
    return _pad_datetime_columns(columns)


def _trim_trailing_nats(datetimes: np.ndarray) -> np.ndarray:
    valid_positions = np.flatnonzero(~np.isnat(datetimes))
    if len(valid_positions) == 0:
        return np.empty(0, dtype="datetime64[ns]")
    return np.asarray(datetimes[: valid_positions[-1] + 1], dtype="datetime64[ns]")

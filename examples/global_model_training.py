"""Train one global stateful LSTM with ``by_series_batch``.

Edit the constants in the first section, then run:

    python examples/global_model_training.py
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytagi.metric as metric
from pytagi.nn import Sequential

from examples.global_model_utils import (
    DataSplit,
    EarlyStopping,
    ForecastLookback,
    build_model,
    by_series_batch,
    predictions_to_original_scale,
    prepare_data,
    prepare_inputs,
    sigma_v_schedule,
    update_model,
    validate_nonnegative_variances,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

VALUES_FILE = "data/hq_benchmark/weekly_values.csv"
DATETIMES_FILE = "data/hq_benchmark/weekly_datetimes.csv"
TIME_COVARIATES = ("week_of_year",)
OUTPUT_DIR = "out/hq_benchmark_training"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
LOOKBACK = 52

BATCH_SIZE = 32
SHUFFLE_SERIES = True
SEED = 1

HIDDEN_SIZES = (50,)
DEVICE = "cpu"  # "cpu" or "cuda"
CPU_THREADS = 1

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_WARMUP_EPOCHS = 3  # Set to 0 to disable warmup.
VALIDATION_METRIC = "log_likelihood"  # "log_likelihood" or "mse"

SIGMA_V_START = 0.5
SIGMA_V_END = 0.1
SIGMA_V_DECAY_FACTOR = 0.75

# "one_step_ahead": condition each prior prediction on its observed target and
# append the posterior to the lookback window.
# "multi_step_ahead": recursively append the prior prediction without using
# the observed target.
VALIDATION_PREDICTION_MODE = "one_step_ahead"


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    output_updater,
    train_data: DataSplit,
    epoch: int,
    sigma_v: float,
) -> None:
    model.train()
    batches = by_series_batch(
        train_data.dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_SERIES,
        seed=SEED + epoch,
    )

    first_batch = True
    for batch in batches:
        # This backend cannot reset its LSTM buffer before the first forward.
        if batch.starts_new_group and (epoch > 0 or not first_batch):
            model.reset_lstm_states()

        inputs, input_variances = prepare_inputs(
            batch.x, time_covariates=batch.time_covariates
        )
        model(inputs, input_variances)
        first_batch = False
        update_model(
            model,
            output_updater,
            batch.y,
            observation_variance=sigma_v**2,
        )


def forecast_batches(
    model,
    split: DataSplit,
    prediction_mode: str,
    sigma_v: float,
):
    """Run stateful forecasts with a posterior or recursive lookback."""
    model.eval()
    lookback = None

    for batch in by_series_batch(split.dataset, BATCH_SIZE):
        if batch.starts_new_group:
            model.reset_lstm_states()
            lookback = ForecastLookback(batch.x)

        inputs, input_variances = prepare_inputs(
            lookback.means,
            lookback.variances,
            batch.time_covariates,
        )
        prior_means, prior_variances = model(inputs, input_variances)
        prior_means = np.asarray(prior_means).reshape(-1)
        prior_variances = np.asarray(prior_variances).reshape(-1)

        lookback.update(
            prior_means=prior_means,
            prior_variances=prior_variances,
            targets=batch.y,
            active=batch.series_ids >= 0,
            mode=prediction_mode,
            observation_variance=sigma_v**2,
        )
        yield batch, prior_means, prior_variances


def calculate_validation_metric(
    targets: np.ndarray,
    predicted_means: np.ndarray,
    predicted_stds: np.ndarray,
    metric_name: str,
) -> float:
    """Calculate a PyTAGI metric from aligned prediction arrays."""
    targets = np.asarray(targets)
    predicted_means = np.asarray(predicted_means)
    predicted_stds = np.asarray(predicted_stds)
    if not (targets.shape == predicted_means.shape == predicted_stds.shape):
        raise ValueError("Targets, predicted means, and stds must have equal shapes.")

    valid = np.isfinite(targets) & np.isfinite(predicted_means)
    if metric_name == "log_likelihood":
        valid &= np.isfinite(predicted_stds) & (predicted_stds > 0)
        if not np.any(valid):
            raise RuntimeError("The validation split contains no usable predictions.")
        return float(
            metric.log_likelihood(
                predicted_means[valid], targets[valid], predicted_stds[valid]
            )
        )
    if metric_name == "mse":
        if not np.any(valid):
            raise RuntimeError("The validation split contains no usable predictions.")
        return float(metric.mse(predicted_means[valid], targets[valid]))
    raise ValueError(
        "VALIDATION_METRIC must be either 'log_likelihood' or 'mse'; "
        f"received {metric_name!r}."
    )


def validation_metric_mode(metric_name: str) -> str:
    """Return the early-stopping direction for a supported metric."""
    if metric_name == "log_likelihood":
        return "max"
    if metric_name == "mse":
        return "min"
    raise ValueError(
        "VALIDATION_METRIC must be either 'log_likelihood' or 'mse'; "
        f"received {metric_name!r}."
    )


def predict(
    model,
    split: DataSplit,
    sigma_v: float,
    prediction_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a split; None keeps teacher-forced lookback inputs."""
    model.eval()
    saved_batches = []
    observation_variance = sigma_v**2

    if prediction_mode is None:
        predictions = teacher_forced_batches(model, split)
    else:
        predictions = forecast_batches(model, split, prediction_mode, sigma_v)

    for batch, prior_means, prior_variances in predictions:
        validate_nonnegative_variances(
            prior_variances,
            active=batch.series_ids >= 0,
            name="Prior prediction variances",
        )
        predictive_variances = prior_variances + observation_variance
        saved_batches.append((batch, prior_means, predictive_variances))

    return predictions_to_original_scale(split, saved_batches)


def teacher_forced_batches(model, split: DataSplit):
    """Use observed values in every lookback, matching the training inputs."""
    for batch in by_series_batch(split.dataset, BATCH_SIZE):
        if batch.starts_new_group:
            model.reset_lstm_states()

        inputs, input_variances = prepare_inputs(
            batch.x, time_covariates=batch.time_covariates
        )
        prior_means, prior_variances = model(inputs, input_variances)
        yield (
            batch,
            np.asarray(prior_means).reshape(-1),
            np.asarray(prior_variances).reshape(-1),
        )


@dataclass
class TrainingRun:
    """The best model found by early stopping and its per-epoch history."""

    model: Sequential
    best_sigma_v: float
    best_validation_metric: float
    sigma_v_history: list[float]
    validation_metric_history: list[float]


def train_with_early_stopping(
    train_data: DataSplit,
    validation_data: DataSplit,
    hidden_sizes: tuple[int, ...],
) -> TrainingRun:
    """Train until early stopping and restore the best validation model."""
    model, output_updater = build_model(
        input_size=LOOKBACK + len(TIME_COVARIATES),
        hidden_sizes=hidden_sizes,
        seed=SEED,
        device=DEVICE,
        cpu_threads=CPU_THREADS,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode=validation_metric_mode(VALIDATION_METRIC),
        warmup_epochs=EARLY_STOPPING_WARMUP_EPOCHS,
    )
    scheduled_sigma_v = sigma_v_schedule(
        num_epochs=MAX_EPOCHS,
        start=SIGMA_V_START,
        end=SIGMA_V_END,
        decay_factor=SIGMA_V_DECAY_FACTOR,
    )
    sigma_v_history = []
    validation_metric_history = []

    for epoch in range(MAX_EPOCHS):
        sigma_v = float(scheduled_sigma_v[epoch])
        sigma_v_history.append(sigma_v)
        train_one_epoch(model, output_updater, train_data, epoch, sigma_v)
        validation_mean, validation_std = predict(
            model,
            validation_data,
            sigma_v=sigma_v,
            prediction_mode=VALIDATION_PREDICTION_MODE,
        )
        validation_score = calculate_validation_metric(
            validation_data.values,
            validation_mean,
            validation_std,
            VALIDATION_METRIC,
        )
        validation_metric_history.append(validation_score)
        print(
            f"Epoch {epoch + 1:03d} | validation {VALIDATION_METRIC}: "
            f"{validation_score:.6f} "
            f"| sigma_v: {sigma_v:.6f}"
            + (
                " | early stopping warmup"
                if epoch < EARLY_STOPPING_WARMUP_EPOCHS
                else ""
            )
        )

        if early_stopping.update(validation_score, model, sigma_v):
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    best_sigma_v = early_stopping.restore_best(model)
    return TrainingRun(
        model=model,
        best_sigma_v=best_sigma_v,
        best_validation_metric=float(early_stopping.best_score),
        sigma_v_history=sigma_v_history,
        validation_metric_history=validation_metric_history,
    )


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, validation_data, _, column_names = prepare_data(
        values_file=VALUES_FILE,
        datetime_file=DATETIMES_FILE,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        lookback=LOOKBACK,
        time_covariates=TIME_COVARIATES,
    )
    run = train_with_early_stopping(train_data, validation_data, HIDDEN_SIZES)
    run.model.save(str(output_dir / "model.bin"))

    train_mean, train_std = predict(run.model, train_data, run.best_sigma_v)
    validation_mean, validation_std = predict(
        run.model,
        validation_data,
        run.best_sigma_v,
        VALIDATION_PREDICTION_MODE,
    )

    np.savez(
        output_dir / "predictions.npz",
        column_names=column_names,
        validation_prediction_mode=VALIDATION_PREDICTION_MODE,
        sigma_v_history=np.asarray(run.sigma_v_history, dtype=np.float32),
        validation_metric=VALIDATION_METRIC,
        validation_metric_history=np.asarray(
            run.validation_metric_history, dtype=np.float64
        ),
        best_validation_metric=np.float64(run.best_validation_metric),
        early_stopping_warmup_epochs=np.int32(EARLY_STOPPING_WARMUP_EPOCHS),
        best_sigma_v=np.float32(run.best_sigma_v),
        time_covariates=np.asarray(TIME_COVARIATES),
        train_target=train_data.values,
        train_datetimes=train_data.datetimes,
        train_mean=train_mean,
        train_std=train_std,
        validation_target=validation_data.values,
        validation_datetimes=validation_data.datetimes,
        validation_mean=validation_mean,
        validation_std=validation_std,
    )
    print(f"Saved model and predictions to {output_dir}.")


if __name__ == "__main__":
    main()

"""Train one global stateful LSTM with ``by_series_batch``.

Edit the constants in the first section, then run:

    python examples/global_model_training.py
"""

from pathlib import Path

import numpy as np

from global_model_utils import (
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
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

VALUES_FILE = "data/toy_embedding/time_series_values.csv"
OUTPUT_DIR = "out/global_stateful_lstm"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
LOOKBACK = 24

BATCH_SIZE = 8
SHUFFLE_SERIES = True
SEED = 1

HIDDEN_SIZES = (40, 40)
DEVICE = "cpu"  # "cpu" or "cuda"
CPU_THREADS = 1

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

SIGMA_V_START = 0.3
SIGMA_V_END = 0.05
SIGMA_V_DECAY_FACTOR = 0.99

# "one_step_ahead": condition each prior prediction on its observed target and
# append the posterior to the lookback window.
# "multi_step_ahead": recursively append the prior prediction without using
# the observed target.
VALIDATION_PREDICTION_MODE = "one_step_ahead"
TEST_PREDICTION_MODE = "one_step_ahead"


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

        inputs, input_variances = prepare_inputs(batch.x)
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

        inputs, input_variances = prepare_inputs(lookback.means, lookback.variances)
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


def validation_rmse(
    model,
    validation_data: DataSplit,
    prediction_mode: str,
    sigma_v: float,
) -> float:
    squared_error = 0.0
    observation_count = 0

    for batch, predicted_means, _ in forecast_batches(
        model, validation_data, prediction_mode, sigma_v
    ):
        targets = batch.y.reshape(-1)
        active = (batch.series_ids >= 0) & np.isfinite(targets)

        errors = predicted_means[active] - targets[active]
        squared_error += float(np.sum(errors**2))
        observation_count += int(np.sum(active))

    if observation_count == 0:
        raise RuntimeError("The validation split contains no usable targets.")
    return float(np.sqrt(squared_error / observation_count))


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
        predictive_variances = prior_variances + observation_variance
        saved_batches.append((batch, prior_means, predictive_variances))

    return predictions_to_original_scale(split, saved_batches)


def teacher_forced_batches(model, split: DataSplit):
    """Use observed values in every lookback, matching the training inputs."""
    for batch in by_series_batch(split.dataset, BATCH_SIZE):
        if batch.starts_new_group:
            model.reset_lstm_states()

        inputs, input_variances = prepare_inputs(batch.x)
        prior_means, prior_variances = model(inputs, input_variances)
        yield (
            batch,
            np.asarray(prior_means).reshape(-1),
            np.asarray(prior_variances).reshape(-1),
        )


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, validation_data, test_data, column_names = prepare_data(
        values_file=VALUES_FILE,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        lookback=LOOKBACK,
    )
    model, output_updater = build_model(
        input_size=LOOKBACK,
        hidden_sizes=HIDDEN_SIZES,
        seed=SEED,
        device=DEVICE,
        cpu_threads=CPU_THREADS,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )
    scheduled_sigma_v = sigma_v_schedule(
        num_epochs=MAX_EPOCHS,
        start=SIGMA_V_START,
        end=SIGMA_V_END,
        decay_factor=SIGMA_V_DECAY_FACTOR,
    )
    sigma_v_history = []

    for epoch in range(MAX_EPOCHS):
        sigma_v = float(scheduled_sigma_v[epoch])
        sigma_v_history.append(sigma_v)
        train_one_epoch(model, output_updater, train_data, epoch, sigma_v)
        score = validation_rmse(
            model,
            validation_data,
            prediction_mode=VALIDATION_PREDICTION_MODE,
            sigma_v=sigma_v,
        )
        print(
            f"Epoch {epoch + 1:03d} | validation RMSE: {score:.6f} "
            f"| sigma_v: {sigma_v:.6f}"
        )

        if early_stopping.update(score, model, sigma_v):
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    best_sigma_v = early_stopping.restore_best(model)
    model.save(str(output_dir / "model.bin"))

    train_mean, train_std = predict(model, train_data, best_sigma_v)
    validation_mean, validation_std = predict(
        model,
        validation_data,
        best_sigma_v,
        VALIDATION_PREDICTION_MODE,
    )
    test_mean, test_std = predict(
        model,
        test_data,
        best_sigma_v,
        TEST_PREDICTION_MODE,
    )

    np.savez(
        output_dir / "predictions.npz",
        column_names=column_names,
        validation_prediction_mode=VALIDATION_PREDICTION_MODE,
        test_prediction_mode=TEST_PREDICTION_MODE,
        sigma_v_history=np.asarray(sigma_v_history, dtype=np.float32),
        best_sigma_v=np.float32(best_sigma_v),
        train_target=train_data.values,
        train_mean=train_mean,
        train_std=train_std,
        validation_target=validation_data.values,
        validation_mean=validation_mean,
        validation_std=validation_std,
        test_target=test_data.values,
        test_mean=test_mean,
        test_std=test_std,
    )
    print(f"Saved model and predictions to {output_dir}.")


if __name__ == "__main__":
    main()

"""Train one global stateful LSTM with ``by_series_batch``.

Edit the constants in the first section, then run:

    python examples/global_model_training.py
"""

from pathlib import Path

import numpy as np

from global_model_utils import (
    DataSplit,
    EarlyStopping,
    build_model,
    by_series_batch,
    predictions_to_original_scale,
    prepare_data,
    prepare_inputs,
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
OBSERVATION_STD = 0.3
DEVICE = "cpu"  # "cpu" or "cuda"
CPU_THREADS = 1

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------


def train_one_epoch(model, output_updater, train_data: DataSplit, epoch: int) -> None:
    model.train()
    batches = by_series_batch(
        train_data.dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_SERIES,
        seed=SEED + epoch,
    )

    for batch in batches:
        if batch.starts_new_group:
            model.reset_lstm_states()

        inputs, input_variances = prepare_inputs(batch.x)
        model(inputs, input_variances)
        update_model(
            model,
            output_updater,
            batch.y,
            observation_variance=OBSERVATION_STD**2,
        )


def validation_rmse(model, validation_data: DataSplit) -> float:
    model.eval()
    squared_error = 0.0
    observation_count = 0

    for batch in by_series_batch(validation_data.dataset, BATCH_SIZE):
        if batch.starts_new_group:
            model.reset_lstm_states()

        inputs, input_variances = prepare_inputs(batch.x)
        predicted_means, _ = model(inputs, input_variances)
        predicted_means = np.asarray(predicted_means).reshape(-1)
        targets = batch.y.reshape(-1)
        active = (batch.series_ids >= 0) & np.isfinite(targets)

        errors = predicted_means[active] - targets[active]
        squared_error += float(np.sum(errors**2))
        observation_count += int(np.sum(active))

    if observation_count == 0:
        raise RuntimeError("The validation split contains no usable targets.")
    return float(np.sqrt(squared_error / observation_count))


def predict(model, split: DataSplit) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    saved_batches = []
    observation_variance = OBSERVATION_STD**2

    for batch in by_series_batch(split.dataset, BATCH_SIZE):
        if batch.starts_new_group:
            model.reset_lstm_states()

        inputs, input_variances = prepare_inputs(batch.x)
        means, variances = model(inputs, input_variances)
        means = np.asarray(means).reshape(-1)
        variances = np.asarray(variances).reshape(-1) + observation_variance
        saved_batches.append((batch, means, variances))

    return predictions_to_original_scale(split, saved_batches)


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

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(model, output_updater, train_data, epoch)
        score = validation_rmse(model, validation_data)
        print(f"Epoch {epoch + 1:03d} | validation RMSE: {score:.6f}")

        if early_stopping.update(score, model):
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    early_stopping.restore_best(model)
    model.save(str(output_dir / "model.bin"))

    train_mean, train_std = predict(model, train_data)
    validation_mean, validation_std = predict(model, validation_data)
    test_mean, test_std = predict(model, test_data)

    np.savez(
        output_dir / "predictions.npz",
        column_names=column_names,
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

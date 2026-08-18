"""Grid search the global LSTM hidden size on the validation split.

Every setting other than the hidden size is taken from
``examples/global_model_training.py``. Edit ``HIDDEN_SIZE_GRID`` below, then run:

    python examples/global_model_grid_search.py
"""

import csv
from pathlib import Path

from examples.global_model_training import (
    DATETIMES_FILE,
    LOOKBACK,
    TIME_COVARIATES,
    TRAIN_RATIO,
    VALIDATION_METRIC,
    VALIDATION_RATIO,
    VALUES_FILE,
    train_with_early_stopping,
    validation_metric_mode,
)
from examples.global_model_utils import prepare_data


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

HIDDEN_SIZE_GRID = ((32,), (64,), (128,), (256,), (512,))
OUTPUT_DIR = "out/hq_benchmark_grid_search"


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, validation_data, _, _ = prepare_data(
        values_file=VALUES_FILE,
        datetime_file=DATETIMES_FILE,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        lookback=LOOKBACK,
        time_covariates=TIME_COVARIATES,
    )

    results = []
    for hidden_sizes in HIDDEN_SIZE_GRID:
        print(f"\nTraining hidden sizes {hidden_sizes}.")
        run = train_with_early_stopping(train_data, validation_data, hidden_sizes)
        results.append(
            {
                "hidden_sizes": "-".join(str(size) for size in hidden_sizes),
                "epochs_run": len(run.validation_metric_history),
                "best_sigma_v": run.best_sigma_v,
                f"best_validation_{VALIDATION_METRIC}": run.best_validation_metric,
            }
        )

    score_key = f"best_validation_{VALIDATION_METRIC}"
    best_first = validation_metric_mode(VALIDATION_METRIC) == "max"
    results.sort(key=lambda result: result[score_key], reverse=best_first)

    results_file = output_dir / "grid_search.csv"
    with results_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRanked by validation {VALIDATION_METRIC}:")
    for result in results:
        print(
            f"  hidden sizes {result['hidden_sizes']:>12s} | "
            f"{score_key}: {result[score_key]:.6f} | "
            f"epochs: {result['epochs_run']:3d} | "
            f"sigma_v: {result['best_sigma_v']:.6f}"
        )
    print(f"\nBest hidden sizes: {results[0]['hidden_sizes']}.")
    print(f"Saved grid search results to {results_file}.")


if __name__ == "__main__":
    main()

"""Grid search the global LSTM hidden size on the validation split.

Each hidden size is trained once per seed and ranked on the seed-averaged
validation metric. Every other setting is taken from
``examples/global_model_training.py``. Edit ``HIDDEN_SIZE_GRID`` and ``SEEDS``
below, then run:

    python examples/global_model_grid_search.py
"""

import csv
from pathlib import Path

import numpy as np

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
SEEDS = (1, 2, 3, 4, 5)
OUTPUT_DIR = "out/hq_benchmark_grid_search"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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

    seed_rows = []
    results = []
    for hidden_sizes in HIDDEN_SIZE_GRID:
        label = "-".join(str(size) for size in hidden_sizes)
        scores = []
        epochs = []
        sigma_vs = []
        for seed in SEEDS:
            print(f"\nTraining hidden sizes {hidden_sizes} with seed {seed}.")
            run = train_with_early_stopping(
                train_data, validation_data, hidden_sizes, seed
            )
            scores.append(run.best_validation_metric)
            epochs.append(len(run.validation_metric_history))
            sigma_vs.append(run.best_sigma_v)
            seed_rows.append(
                {
                    "hidden_sizes": label,
                    "seed": seed,
                    "epochs_run": epochs[-1],
                    "best_sigma_v": sigma_vs[-1],
                    f"best_validation_{VALIDATION_METRIC}": scores[-1],
                }
            )

        results.append(
            {
                "hidden_sizes": label,
                "num_seeds": len(SEEDS),
                "mean_epochs_run": float(np.mean(epochs)),
                "mean_best_sigma_v": float(np.mean(sigma_vs)),
                f"mean_validation_{VALIDATION_METRIC}": float(np.mean(scores)),
                f"std_validation_{VALIDATION_METRIC}": float(np.std(scores)),
            }
        )

    score_key = f"mean_validation_{VALIDATION_METRIC}"
    std_key = f"std_validation_{VALIDATION_METRIC}"
    best_first = validation_metric_mode(VALIDATION_METRIC) == "max"
    results.sort(key=lambda result: result[score_key], reverse=best_first)

    results_file = output_dir / "grid_search.csv"
    seed_results_file = output_dir / "grid_search_per_seed.csv"
    write_csv(results_file, results)
    write_csv(seed_results_file, seed_rows)

    print(f"\nRanked by seed-averaged validation {VALIDATION_METRIC}:")
    for result in results:
        print(
            f"  hidden sizes {result['hidden_sizes']:>12s} | "
            f"{score_key}: {result[score_key]:.6f} "
            f"(std {result[std_key]:.6f}) | "
            f"mean epochs: {result['mean_epochs_run']:6.1f} | "
            f"mean sigma_v: {result['mean_best_sigma_v']:.6f}"
        )
    print(f"\nBest hidden sizes: {results[0]['hidden_sizes']}.")
    print(f"Saved grid search results to {results_file} and {seed_results_file}.")


if __name__ == "__main__":
    main()

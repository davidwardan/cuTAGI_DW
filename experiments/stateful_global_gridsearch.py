import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.config import Config
from experiments.wandb_helpers import finish_run, init_run
from experiments import stateful_global as base_script
from experiments.utils import load_true_split_arrays

from pytagi import Normalizer as normalizer
from pytagi import cuda
import pytagi.metric as metric

DEFAULT_SEEDS: Sequence[int] = (
    11,
    42,
    235,
)
DEFAULT_EXPERIMENTS: Sequence[str] = ("train100",)


def _trim_trailing_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)
    if x.size == 0:
        return x.astype(np.float32)
    valid = ~np.isnan(x)
    if not np.any(valid):
        return np.array([], dtype=np.float32)
    last = np.where(valid)[0][-1]
    return x[: last + 1].astype(np.float32)


def _compute_validation_metrics(
    config: Config,
    experiment_name: str,
) -> Dict[str, float]:
    input_dir = Path("out") / experiment_name
    val_states = np.load(input_dir / "val_states.npz")

    true_train, true_val, _ = load_true_split_arrays(**config.true_split_kwargs())

    val_rmse_list = []
    val_log_lik_list = []
    val_mae_list = []

    all_stand_y_true = []
    all_stand_y_pred = []
    all_stand_s_pred = []

    train_offset = config.split_target_offset("train")
    val_offset = config.split_target_offset("val")

    for ts_idx in tqdm(config.ts_to_use, desc="Scoring validation series", leave=False):
        local_idx = config.ts_to_use.index(ts_idx)

        yt_train = _trim_trailing_nans(true_train[train_offset:, local_idx])
        yt_val = _trim_trailing_nans(true_val[val_offset:, local_idx])

        if len(yt_val) == 0:
            continue

        ypred_val = val_states["mu"][local_idx][: len(yt_val)]
        spred_val = val_states["std"][local_idx][: len(yt_val)]

        if config.data.loader.scale_method == "standard":
            train_mean = np.nanmean(yt_train)
            train_std = np.nanstd(yt_train)
        else:
            train_mean = 0.0
            train_std = 1.0

        stand_y_true = normalizer.standardize(yt_val, train_mean, train_std)
        stand_y_pred = normalizer.standardize(ypred_val, train_mean, train_std)
        stand_s_pred = normalizer.standardize_std(spred_val, train_std)

        val_rmse_list.append(metric.rmse(stand_y_pred, stand_y_true))
        val_log_lik_list.append(
            metric.log_likelihood(stand_y_pred, stand_y_true, stand_s_pred)
        )
        val_mae_list.append(metric.mae(stand_y_pred, stand_y_true))

        all_stand_y_true.append(stand_y_true)
        all_stand_y_pred.append(stand_y_pred)
        all_stand_s_pred.append(stand_s_pred)

    if not val_rmse_list:
        raise ValueError("Validation search produced no valid validation targets.")

    full_stand_y_true = np.concatenate(all_stand_y_true)
    full_stand_y_pred = np.concatenate(all_stand_y_pred)
    full_stand_s_pred = np.concatenate(all_stand_s_pred)

    return {
        "macro_rmse": float(np.nanmean(val_rmse_list)),
        "macro_log_lik": float(np.nanmean(val_log_lik_list)),
        "macro_mae": float(np.nanmean(val_mae_list)),
        "micro_rmse": float(metric.rmse(full_stand_y_pred, full_stand_y_true)),
        "micro_log_lik": float(
            metric.log_likelihood(
                full_stand_y_pred,
                full_stand_y_true,
                full_stand_s_pred,
            )
        ),
        "micro_mae": float(metric.mae(full_stand_y_pred, full_stand_y_true)),
    }


def _select_best_trial(results_df: pd.DataFrame, metric_name: str) -> pd.Series:
    metric_aliases = {
        "rmse": "micro_rmse",
        "loglik": "micro_log_lik",
        "log_lik": "micro_log_lik",
        "mae": "micro_mae",
        "macro_rmse": "macro_rmse",
        "macro_log_lik": "macro_log_lik",
        "macro_mae": "macro_mae",
        "micro_rmse": "micro_rmse",
        "micro_log_lik": "micro_log_lik",
        "micro_mae": "micro_mae",
    }
    metric_key = metric_aliases.get(metric_name.lower())
    if metric_key is None:
        supported = ", ".join(sorted(metric_aliases))
        raise ValueError(
            f"Unsupported lookback search metric '{metric_name}'. Supported values: {supported}"
        )

    ascending = metric_key != "micro_log_lik" and metric_key != "macro_log_lik"
    return results_df.sort_values(metric_key, ascending=ascending).iloc[0]


def _train_lookback_trial_worker(
    config_payload: Dict[str, Any],
    trial_experiment_name: str,
) -> None:
    trial_config = Config(**config_payload)
    base_script.train_model(
        trial_config,
        experiment_name=trial_experiment_name,
        wandb_run=None,
    )


def _run_lookback_trial_in_subprocess(
    ctx: Any,
    trial_config: Config,
    trial_experiment_name: str,
) -> None:
    process = ctx.Process(
        target=_train_lookback_trial_worker,
        args=(trial_config.model_dump(), trial_experiment_name),
    )
    process.start()
    process.join()

    if process.exitcode:
        raise RuntimeError(
            "Lookback trial failed "
            f"(look_back_len={trial_config.look_back_len}) "
            f"with exit code {process.exitcode}."
        )


def optimize_lookback(
    config: Config,
    experiment_name: Optional[str] = None,
    wandb_run: Optional[Any] = None,
    evaluate_best_on_test: Optional[bool] = None,
) -> Dict[str, Any]:
    if experiment_name is None:
        raise ValueError("experiment_name must be provided for lookback search.")

    candidate_values = list(dict.fromkeys(config.lookback_search.candidate_values))
    if not candidate_values:
        candidate_values = [config.look_back_len]

    if any(value < 2 for value in candidate_values):
        raise ValueError("All lookback candidates must be >= 2.")

    output_dir = Path("out") / experiment_name
    os.makedirs(output_dir, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")

    results = []
    ctx = mp.get_context("spawn")

    if wandb_run is not None:
        print(
            "W&B logging object detected. "
            "Lookback trials run in subprocesses and will train without live W&B trial logs."
        )

    for lookback in candidate_values:
        trial_config = config.model_copy(deep=True)
        trial_config.data.loader.look_back_len = int(lookback)
        trial_experiment_name = f"{experiment_name}/lookback_{lookback}"

        print(f"Running lookback search trial with look_back_len={lookback}")
        _run_lookback_trial_in_subprocess(
            ctx=ctx,
            trial_config=trial_config,
            trial_experiment_name=trial_experiment_name,
        )

        val_metrics = _compute_validation_metrics(
            trial_config,
            experiment_name=trial_experiment_name,
        )

        trial_result = {
            "lookback": lookback,
            "experiment_name": trial_experiment_name,
            **val_metrics,
        }
        results.append(trial_result)

        print(
            "  Validation scores:"
            f" micro_rmse={trial_result['micro_rmse']:.4f},"
            f" micro_log_lik={trial_result['micro_log_lik']:.4f},"
            f" micro_mae={trial_result['micro_mae']:.4f}"
        )

    results_df = pd.DataFrame(results)
    best_trial = _select_best_trial(results_df, config.lookback_search.metric)

    results_df.to_csv(output_dir / "lookback_search_results.csv", index=False)

    best_config = config.model_copy(deep=True)
    best_config.data.loader.look_back_len = int(best_trial["lookback"])
    best_config.to_yaml(output_dir / "best_config.yaml")

    summary = {
        "search_mode": "gridsearch",
        "selection_metric": config.lookback_search.metric,
        "candidate_values": candidate_values,
        "best_lookback": int(best_trial["lookback"]),
        "best_experiment_name": best_trial["experiment_name"],
        "best_scores": {
            "macro_rmse": float(best_trial["macro_rmse"]),
            "macro_log_lik": float(best_trial["macro_log_lik"]),
            "macro_mae": float(best_trial["macro_mae"]),
            "micro_rmse": float(best_trial["micro_rmse"]),
            "micro_log_lik": float(best_trial["micro_log_lik"]),
            "micro_mae": float(best_trial["micro_mae"]),
        },
    }
    with open(output_dir / "lookback_search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "Selected best lookback:"
        f" {summary['best_lookback']} using {summary['selection_metric']}"
    )

    should_evaluate_best = config.lookback_search.evaluate_best_on_test
    if evaluate_best_on_test is not None:
        should_evaluate_best = evaluate_best_on_test

    if should_evaluate_best:
        base_script.eval_model(
            best_config,
            experiment_name=best_trial["experiment_name"],
            wandb_run=wandb_run,
        )

    return summary


def main(
    Train: bool = True,
    Eval: bool = True,
    log_wandb: bool = False,
    seeds: Optional[Sequence[int]] = None,
    experiments: Optional[Sequence[str]] = None,
):
    list_of_seeds = list(seeds) if seeds is not None else list(DEFAULT_SEEDS)
    list_of_experiments = (
        list(experiments) if experiments is not None else list(DEFAULT_EXPERIMENTS)
    )

    if not list_of_seeds:
        raise ValueError("At least one seed must be provided.")
    if not list_of_experiments:
        raise ValueError("At least one experiment must be provided.")

    for seed in list_of_seeds:
        for exp in list_of_experiments:
            print(f"Running experiment: {exp} with seed {seed}")

            model_category = "global"
            embed_category = "no-embeddings"
            experiment_name = f"seed{seed}/{exp}/experiment01_{model_category}_{embed_category}_gridsearch"

            config = Config.from_yaml(
                "experiments/configurations/global_no-embeddings_HQ127_gridsearch.yaml"
            )

            config.seed = seed
            config.model.device = "cuda" if cuda.is_available() else "cpu"
            config.data.paths.x_train = f"data/hq/{exp}/split_train_values.csv"
            config.data.paths.dates_train = f"data/hq/{exp}/split_train_datetimes.csv"
            config.data.loader.order_mode = "by_window"

            config_dict = config.wandb_dict()
            config_dict["model_type"] = f"{model_category}_{embed_category}_gridsearch"

            config.display()

            if log_wandb:
                run_id = (
                    f"{model_category}_{embed_category}_{exp}_seed{seed}_gridsearch"
                ).replace(" ", "")
                run = init_run(
                    project="tracking_weights_lstm",
                    name=run_id,
                    group=f"{model_category}_Seed{embed_category}",
                    tags=["gridsearch"],
                    config=config_dict,
                    reinit=True,
                    save_code=True,
                )
            else:
                run = None

            if Train:
                optimize_lookback(
                    config,
                    experiment_name=experiment_name,
                    wandb_run=run,
                    evaluate_best_on_test=Eval,
                )
            elif Eval:
                summary_path = (
                    Path("out") / experiment_name / "lookback_search_summary.json"
                )
                if not summary_path.exists():
                    raise FileNotFoundError(
                        f"Missing search summary at {summary_path}. Run training first."
                    )
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                best_config = config.model_copy(deep=True)
                best_config.data.loader.look_back_len = int(summary["best_lookback"])
                base_script.eval_model(
                    best_config,
                    experiment_name=summary["best_experiment_name"],
                    wandb_run=run,
                )

            if log_wandb:
                finish_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stateful global lookback gridsearch")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="One or more random seeds to run (e.g. --seeds 11 42 235).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        help="One or more train split folders under data/hq (e.g. train60 train100).",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training and only run evaluation from existing summaries.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation of the selected best lookback.",
    )
    args = parser.parse_args()

    main(
        Train=not args.no_train,
        Eval=not args.no_eval,
        log_wandb=args.log_wandb,
        seeds=args.seeds,
        experiments=args.experiments,
    )

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.config import Config
from experiments.wandb_helpers import finish_run, init_run
from experiments import stateful_global as base_script

from pytagi import Normalizer as normalizer
from pytagi import cuda
import pytagi.metric as metric


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

    true_train = pd.read_csv(
        config.x_file[0],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.ts_to_use,
    ).values
    true_val = pd.read_csv(
        config.x_file[1],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.ts_to_use,
    ).values

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
        candidate_values = [config.data.loader.input_seq_len]

    if any(value < 2 for value in candidate_values):
        raise ValueError("All lookback candidates must be >= 2.")

    output_dir = Path("out") / experiment_name
    os.makedirs(output_dir, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")

    results = []

    for lookback in candidate_values:
        trial_config = config.model_copy(deep=True)
        trial_config.data.loader.input_seq_len = int(lookback)
        trial_experiment_name = f"{experiment_name}/lookback_{lookback}"

        print(f"Running lookback search trial with input_seq_len={lookback}")
        base_script.train_model(
            trial_config,
            experiment_name=trial_experiment_name,
            wandb_run=wandb_run,
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
    best_config.data.loader.input_seq_len = int(best_trial["lookback"])
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


def main(Train=True, Eval=True, log_wandb=False):
    list_of_seeds = [11, 42, 27, 3, 99]
    list_of_experiments = ["train30", "train40", "train60", "train80", "train100"]

    for seed in list_of_seeds:
        for exp in list_of_experiments:
            print(f"Running experiment: {exp} with seed {seed}")

            model_category = "global"
            embed_category = "no-embeddings"
            experiment_name = (
                f"seed{seed}/{exp}/experiment01_{model_category}_{embed_category}_gridsearch"
            )

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
                summary_path = Path("out") / experiment_name / "lookback_search_summary.json"
                if not summary_path.exists():
                    raise FileNotFoundError(
                        f"Missing search summary at {summary_path}. Run training first."
                    )
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                best_config = config.model_copy(deep=True)
                best_config.data.loader.input_seq_len = int(summary["best_lookback"])
                base_script.eval_model(
                    best_config,
                    experiment_name=summary["best_experiment_name"],
                    wandb_run=run,
                )

            if log_wandb:
                finish_run(run)


if __name__ == "__main__":
    main(True, True)

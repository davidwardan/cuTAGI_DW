import os
import json
import argparse
import multiprocessing as mp
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.config import Config
from experiments.wandb_helpers import finish_run, init_run
from experiments import stateful_global as base_script
from experiments.utils import load_predictive_uncertainty, load_true_split_arrays

from pytagi import Normalizer as normalizer
from pytagi import cuda
import pytagi.metric as metric

DEFAULT_SEEDS: Sequence[int] = (
    1,
    2,
    3,
    4,
    5,
)
DEFAULT_TRAIN_USE_RATIOS: Sequence[float] = (1.0,)
GRIDSEARCH_OUTPUT_ROOT = Path("experiments/out/gridsearch")


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


def _gridsearch_output_path(experiment_name: str) -> Path:
    experiment_path = Path(experiment_name)
    if experiment_path.parts and experiment_path.parts[0] == "gridsearch":
        return Path("experiments/out") / experiment_path
    return GRIDSEARCH_OUTPUT_ROOT / experiment_path


def _gridsearch_experiment_name(experiment_name: str) -> str:
    experiment_path = Path(experiment_name)
    if experiment_path.parts and experiment_path.parts[0] == "gridsearch":
        return experiment_name
    return str(Path("gridsearch") / experiment_path)


def _compute_validation_metrics(
    config: Config,
    experiment_name: str,
) -> Dict[str, float]:
    input_dir = _gridsearch_output_path(experiment_name)
    val_states = np.load(input_dir / "val_states.npz")
    val_total_std, _, _ = load_predictive_uncertainty(val_states)
    if val_total_std is None:
        raise ValueError(
            "Missing predictive uncertainty in val_states.npz. "
            "Expected either `std` or both `epistemic_std` and `aleatoric_std`."
        )

    true_train, true_val, _ = load_true_split_arrays(**config.true_split_kwargs())

    val_rmse_list = []
    val_log_lik_list = []
    val_mae_list = []

    all_stand_y_true = []
    all_stand_y_pred = []
    all_stand_s_pred = []

    train_offset = config.true_split_target_offset("train")
    val_offset = config.true_split_target_offset("val")

    for ts_idx in tqdm(config.ts_to_use, desc="Scoring validation series", leave=False):
        local_idx = config.ts_to_use.index(ts_idx)

        yt_train = _trim_trailing_nans(true_train[train_offset:, local_idx])
        yt_val = _trim_trailing_nans(true_val[val_offset:, local_idx])

        if len(yt_val) == 0:
            continue

        ypred_val = val_states["mu"][local_idx][: len(yt_val)]
        spred_val = val_total_std[local_idx][: len(yt_val)]

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
            f"(look_back_len={trial_config.look_back_len}, "
            f"hidden_sizes={trial_config.model.hidden_sizes}) "
            f"with exit code {process.exitcode}."
        )


def _apply_hidden_size(config: Config, hidden_size: int) -> None:
    if hidden_size <= 0:
        raise ValueError("All hidden_size candidates must be > 0.")

    config.model.hidden_sizes = [int(hidden_size)]


def _resolve_search_space(
    config: Config,
    search_target: Optional[str] = None,
    lookback_candidates: Optional[Sequence[int]] = None,
    hidden_size_candidates: Optional[Sequence[int]] = None,
) -> Tuple[str, List[int], List[int]]:
    resolved_search_target = (
        search_target or config.lookback_search.search_target or "lookback"
    ).lower()
    valid_targets = {"lookback", "hidden_size", "both"}
    if resolved_search_target not in valid_targets:
        valid_targets_str = ", ".join(sorted(valid_targets))
        raise ValueError(
            f"Unsupported search target '{resolved_search_target}'. "
            f"Expected one of: {valid_targets_str}."
        )

    default_hidden = (
        int(config.model.hidden_sizes[0]) if config.model.hidden_sizes else 1
    )

    resolved_lookbacks = list(
        dict.fromkeys(
            list(lookback_candidates)
            if lookback_candidates is not None
            else list(config.lookback_search.candidate_values)
        )
    )
    if not resolved_lookbacks:
        resolved_lookbacks = [int(config.look_back_len)]
    if any(int(value) < 2 for value in resolved_lookbacks):
        raise ValueError("All lookback candidates must be >= 2.")
    resolved_lookbacks = [int(value) for value in resolved_lookbacks]

    resolved_hidden_sizes = list(
        dict.fromkeys(
            list(hidden_size_candidates)
            if hidden_size_candidates is not None
            else list(config.lookback_search.hidden_size_candidate_values)
        )
    )
    if not resolved_hidden_sizes:
        resolved_hidden_sizes = [default_hidden]
    if any(int(value) <= 0 for value in resolved_hidden_sizes):
        raise ValueError("All hidden_size candidates must be > 0.")
    resolved_hidden_sizes = [int(value) for value in resolved_hidden_sizes]

    if resolved_search_target == "lookback":
        resolved_hidden_sizes = [default_hidden]
    elif resolved_search_target == "hidden_size":
        resolved_lookbacks = [int(config.look_back_len)]

    return resolved_search_target, resolved_lookbacks, resolved_hidden_sizes


def optimize_lookback(
    config: Config,
    experiment_name: Optional[str] = None,
    wandb_run: Optional[Any] = None,
    evaluate_best_on_test: Optional[bool] = None,
    search_target: Optional[str] = None,
    lookback_candidates: Optional[Sequence[int]] = None,
    hidden_size_candidates: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if experiment_name is None:
        raise ValueError("experiment_name must be provided for lookback search.")
    experiment_name = _gridsearch_experiment_name(experiment_name)

    (
        resolved_search_target,
        lookback_candidate_values,
        hidden_size_candidate_values,
    ) = _resolve_search_space(
        config,
        search_target=search_target,
        lookback_candidates=lookback_candidates,
        hidden_size_candidates=hidden_size_candidates,
    )
    search_hidden_size = resolved_search_target in {"hidden_size", "both"}

    output_dir = _gridsearch_output_path(experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")

    results = []
    ctx = mp.get_context("spawn")

    if wandb_run is not None:
        print(
            "W&B logging object detected. "
            "Lookback trials run in subprocesses and will train without live W&B trial logs."
        )

    for lookback, hidden_size in product(
        lookback_candidate_values,
        hidden_size_candidate_values,
    ):
        trial_config = config.model_copy(deep=True)
        trial_config.data.loader.look_back_len = int(lookback)
        if search_hidden_size:
            _apply_hidden_size(trial_config, int(hidden_size))
            trial_hidden_size = int(hidden_size)
            trial_experiment_name = (
                f"{experiment_name}/lookback_{lookback}_hidden_size_{hidden_size}"
            )
        else:
            trial_hidden_size = (
                int(trial_config.model.hidden_sizes[0])
                if trial_config.model.hidden_sizes
                else 1
            )
            trial_experiment_name = f"{experiment_name}/lookback_{lookback}"

        print(
            "Running grid search trial with "
            f"look_back_len={lookback}, hidden_size={trial_hidden_size}"
        )
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
            "hidden_size": trial_hidden_size,
            "hidden_sizes": list(trial_config.model.hidden_sizes),
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

    results_df.to_csv(output_dir / "gridsearch_results.csv", index=False)
    results_df.to_csv(output_dir / "lookback_search_results.csv", index=False)

    best_config = config.model_copy(deep=True)
    best_config.data.loader.look_back_len = int(best_trial["lookback"])
    if search_hidden_size:
        _apply_hidden_size(best_config, int(best_trial["hidden_size"]))
    best_config.to_yaml(output_dir / "best_config.yaml")

    best_hidden_size = (
        int(best_config.model.hidden_sizes[0]) if best_config.model.hidden_sizes else 1
    )

    summary = {
        "search_mode": "gridsearch",
        "search_target": resolved_search_target,
        "selection_metric": config.lookback_search.metric,
        "lookback_candidate_values": lookback_candidate_values,
        "hidden_size_candidate_values": hidden_size_candidate_values,
        "candidate_values": lookback_candidate_values,
        "best_lookback": int(best_trial["lookback"]),
        "best_hidden_size": best_hidden_size,
        "best_hidden_sizes": list(best_config.model.hidden_sizes),
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
    with open(output_dir / "gridsearch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "lookback_search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "Selected best trial:"
        f" look_back_len={summary['best_lookback']},"
        f" hidden_size={summary['best_hidden_size']}"
        f" using {summary['selection_metric']}"
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
    train_use_ratios: Optional[Sequence[float]] = None,
    search_target: Optional[str] = None,
    lookback_candidates: Optional[Sequence[int]] = None,
    hidden_size_candidates: Optional[Sequence[int]] = None,
):
    list_of_seeds = list(seeds) if seeds is not None else list(DEFAULT_SEEDS)
    list_of_train_use_ratios = (
        list(train_use_ratios)
        if train_use_ratios is not None
        else list(DEFAULT_TRAIN_USE_RATIOS)
    )

    if not list_of_seeds:
        raise ValueError("At least one seed must be provided.")
    if not list_of_train_use_ratios:
        raise ValueError("At least one train_use_ratio must be provided.")

    for seed in list_of_seeds:
        for train_use_ratio in list_of_train_use_ratios:
            ratio_tag = f"train_use_{int(round(train_use_ratio * 100)):03d}"
            print(f"Running experiment: {ratio_tag} with seed {seed}")

            model_category = "global"
            embed_category = "no-embeddings"
            experiment_name = (
                f"seed{seed}/{ratio_tag}/"
                f"experiment01_{model_category}_{embed_category}_gridsearch_tagiv"
            )
            experiment_name = _gridsearch_experiment_name(experiment_name)

            config = Config.from_yaml(
                f"experiments/config/{model_category}_{embed_category}_HQ127_gridsearch.yaml"
            )

            config.seed = seed
            config.model.device = "cuda" if cuda.is_available() else "cpu"
            config.data.loader.train_use_ratio = train_use_ratio
            # config.data.loader.order_mode = "by_window"

            config_dict = config.wandb_dict()
            config_dict["model_type"] = f"{model_category}_{embed_category}_gridsearch"

            config.display()

            if log_wandb:
                run_id = (
                    f"{model_category}_{embed_category}_{ratio_tag}_seed{seed}_gridsearch"
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
                    search_target=search_target,
                    lookback_candidates=lookback_candidates,
                    hidden_size_candidates=hidden_size_candidates,
                )
            elif Eval:
                summary_dir = _gridsearch_output_path(experiment_name)
                summary_path = summary_dir / "gridsearch_summary.json"
                if not summary_path.exists():
                    summary_path = summary_dir / "lookback_search_summary.json"
                if not summary_path.exists():
                    raise FileNotFoundError(
                        f"Missing search summary at {summary_path}. Run training first."
                    )
                with open(summary_path, "r") as f:
                    summary = json.load(f)

                best_config = config.model_copy(deep=True)
                best_config.data.loader.look_back_len = int(summary["best_lookback"])
                if "best_hidden_sizes" in summary and summary["best_hidden_sizes"]:
                    best_config.model.hidden_sizes = [
                        int(v) for v in summary["best_hidden_sizes"]
                    ]
                elif "best_hidden_size" in summary:
                    _apply_hidden_size(best_config, int(summary["best_hidden_size"]))
                base_script.eval_model(
                    best_config,
                    experiment_name=summary["best_experiment_name"],
                    wandb_run=run,
                )

            if log_wandb:
                finish_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stateful global gridsearch (lookback, hidden_size, or both)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="One or more random seeds to run (e.g. --seeds 11 42 235).",
    )
    parser.add_argument(
        "--train-use-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_TRAIN_USE_RATIOS),
        help="One or more train_use_ratio values to run (e.g. 0.3 0.6 1.0).",
    )
    parser.add_argument(
        "--search-target",
        choices=["lookback", "hidden_size", "both"],
        default=None,
        help=(
            "What to search: lookback_len only, hidden_size only, or both. "
            "If omitted, uses lookback_search.search_target from the config."
        ),
    )
    parser.add_argument(
        "--lookback-candidates",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional override for lookback candidates (e.g. --lookback-candidates 12 26 52)."
        ),
    )
    parser.add_argument(
        "--hidden-size-candidates",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional override for hidden_size candidates "
            "(e.g. --hidden-size-candidates 32 50 64)."
        ),
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
        help="Skip evaluation of the selected best trial.",
    )
    args = parser.parse_args()

    main(
        Train=not args.no_train,
        Eval=not args.no_eval,
        log_wandb=args.log_wandb,
        seeds=args.seeds,
        train_use_ratios=args.train_use_ratios,
        search_target=args.search_target,
        lookback_candidates=args.lookback_candidates,
        hidden_size_candidates=args.hidden_size_candidates,
    )

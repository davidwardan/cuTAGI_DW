import multiprocessing as mp
from typing import Iterable, Sequence

from experiments.wandb_helpers import finish_run, init_run


DEFAULT_SEEDS: Sequence[int] = (3, 67, 9, 17, 99)  # (1, 42, 235, 1234, 2024)
DEFAULT_EXPERIMENTS: Sequence[str] = (
    "train30",
    "train40",
    "train60",
    "train80",
    "train100",
)


def _run_experiment(
    seed: int,
    experiment: str,
    train: bool,
    evaluate: bool,
    log_wandb: bool,
) -> None:
    from experiments import time_series_global as tsg

    torch = tsg.torch

    # Model category
    model_category = "global"
    embed_category = "simple embedding"

    # Define experiment name
    experiment_name = (
        f"seed{seed}/{experiment}/experiment01_{model_category}_{embed_category}"
    )

    config = tsg.Config()
    config.seed = seed
    config.batch_size = 16
    config.embedding_size = 15
    # config.embedding_map_dir = "data/hq/ts_embedding_map.csv"
    config.eval_plots = False
    config.embed_plots = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.x_train = f"data/hq/{experiment}/split_train_values.csv"
    config.dates_train = f"data/hq/{experiment}/split_train_datetimes.csv"

    config_dict = config.wandb_dict()
    config_dict["model_type"] = f"{model_category}_{embed_category}"

    wandb_run = None
    if log_wandb:
        # Initialize W&B run
        run_id = f"{model_category}_{embed_category}_{experiment}_seed{seed}".replace(
            " ", ""
        )
        wandb_run = init_run(
            project="Experiment_01_Forecasting",
            name=run_id,
            group=f"{experiment}_Seed{seed}",
            tags=[model_category, embed_category],
            config=config_dict,
            reinit=True,
            save_code=True,
        )

    try:
        if train:
            tsg.train_global_model(
                config,
                experiment_name=experiment_name,
                wandb_run=wandb_run,
            )
        if evaluate:
            tsg.eval_global_model(
                config,
                experiment_name=experiment_name,
                wandb_run=wandb_run,
            )
    finally:
        finish_run(wandb_run)


def run_experiments(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    experiments: Iterable[str] = DEFAULT_EXPERIMENTS,
    *,
    train: bool = True,
    evaluate: bool = True,
    log_wandb: bool = False,
) -> None:
    ctx = mp.get_context("spawn")

    for seed in seeds:
        for experiment in experiments:
            print(f"Launching experiment '{experiment}' with seed {seed}")
            process = ctx.Process(
                target=_run_experiment,
                args=(seed, experiment, train, evaluate, log_wandb),
            )
            process.start()
            process.join()

            if process.exitcode:
                raise RuntimeError(
                    f"Experiment '{experiment}' with seed {seed} failed "
                    f"with exit code {process.exitcode}"
                )


if __name__ == "__main__":
    run_experiments()

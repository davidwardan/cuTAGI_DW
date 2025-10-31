import multiprocessing as mp
from typing import Iterable, Sequence


DEFAULT_SEEDS: Sequence[int] = (1, 42, 235, 1234, 2024)
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

    experiment_name = f"seed{seed}/{experiment}/experiment01_global_model"

    config = tsg.Config()
    config.seed = seed
    config.batch_size = 64
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.x_train = f"data/hq/{experiment}/split_train_values.csv"
    config.dates_train = f"data/hq/{experiment}/split_train_datetimes.csv"

    config_dict = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith("_") and not callable(getattr(config, key))
    }

    wandb_run = None
    if log_wandb:
        wandb_run = tsg.wandb.init(
            project="Forecasting_SHM",
            name=experiment_name,
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
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run_experiments(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    experiments: Iterable[str] = DEFAULT_EXPERIMENTS,
    *,
    train: bool = True,
    evaluate: bool = True,
    log_wandb: bool = True,
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

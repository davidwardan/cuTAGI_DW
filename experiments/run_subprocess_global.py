import multiprocessing as mp
from typing import Iterable, Sequence, List, Tuple

from experiments.wandb_helpers import finish_run, init_run


DEFAULT_SEEDS: Sequence[int] = (3, 67, 9, 17, 99)
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
    device: str,
) -> None:
    """
    Run a single experiment on a specific device (e.g. "cuda:0" or "cuda:1").
    """
    from experiments import time_series_global as tsg

    torch = tsg.torch

    # Make sure Torch is using the right GPU inside this process
    if device.startswith("cuda") and torch.cuda.is_available():
        # e.g. "cuda:0" -> 0
        dev_idx = int(device.split(":")[1])
        torch.cuda.set_device(dev_idx)
    else:
        device = "cpu"

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
    config.eval_plots = False
    config.embed_plots = False
    config.device = device  # <-- important: use the device passed in
    config.x_train = f"data/hq/{experiment}/split_train_values.csv"
    config.dates_train = f"data/hq/{experiment}/split_train_datetimes.csv"

    config_dict = config.wandb_dict()
    config_dict["model_type"] = f"{model_category}_{embed_category}"
    config_dict["device"] = device

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
            tags=[model_category, embed_category, device],
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
    device_ids: Sequence[str] = ("cuda:0", "cuda:1"),  # two GPUs by default
) -> None:
    """
    Run all (seed, experiment) combinations using at most len(device_ids)
    parallel subprocesses, each pinned to a specific CUDA device.
    """
    from experiments.runner_utils import run_experiments_parallel

    # Build the work list
    tasks: List[Tuple[int, str]] = [
        (seed, experiment) for seed in seeds for experiment in experiments
    ]

    def worker_args_mapper(task: Tuple[int, str], device: str) -> Tuple:
        seed, experiment = task
        print(
            f"Launching experiment '{experiment}' with seed {seed} on device {device}"
        )
        return (seed, experiment, train, evaluate, log_wandb, device)

    run_experiments_parallel(
        tasks=tasks,
        worker_fn=_run_experiment,
        worker_args_mapper=worker_args_mapper,
        device_ids=device_ids,
    )


if __name__ == "__main__":
    run_experiments()

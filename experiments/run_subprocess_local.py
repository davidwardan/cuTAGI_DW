import multiprocessing as mp
from typing import Iterable, List, Sequence, Tuple

from experiments.wandb_helpers import finish_run, init_run

DEFAULT_SEEDS: Sequence[int] = (3, 67, 9, 17, 99)
DEFAULT_EXPERIMENTS: Sequence[str] = (
    "train30",
    "train40",
    "train60",
    "train80",
    "train100",
)


def _run_local_experiment(
    seed: int,
    experiment: str,
    train: bool,
    evaluate: bool,
    log_wandb: bool,
    device: str,
) -> None:
    """Run a single local-model experiment bound to a specific device."""
    from experiments import time_series_locals as tsl

    torch = tsl.torch

    # Ensure Torch operations target the requested device
    if device.startswith("cuda") and torch.cuda.is_available():
        dev_idx = int(device.split(":")[1])
        torch.cuda.set_device(dev_idx)
        config_device = "cuda"
    else:
        device = "cpu"
        config_device = "cpu"

    experiment_name = f"seed{seed}/{experiment}/experiment01_local"

    config = tsl.Config()
    config.seed = seed
    config.warmup_epochs = 5
    config.eval_plots = False
    config.device = config_device

    base_path = f"data/hq/{experiment}"
    config.x_train = f"{base_path}/split_train_values.csv"
    config.dates_train = f"{base_path}/split_train_datetimes.csv"
    config.x_val = f"{base_path}/split_val_values.csv"
    config.dates_val = f"{base_path}/split_val_datetimes.csv"
    config.x_test = f"{base_path}/split_test_values.csv"
    config.dates_test = f"{base_path}/split_test_datetimes.csv"

    config_dict = {
        name: getattr(config, name)
        for name in dir(config)
        if not name.startswith("_") and not callable(getattr(config, name))
    }
    config_dict["device_id"] = device

    wandb_run = None
    if log_wandb:
        run_id = f"local_{experiment}_seed{seed}".replace(" ", "")
        wandb_run = init_run(
            project="Local_Model_Run",
            group=f"{experiment}_Seed{seed}",
            name=run_id,
            tags=["local", device],
            config=config_dict,
            reinit=True,
            save_code=True,
        )

    try:
        if train:
            tsl.train_local_models(
                config,
                experiment_name=experiment_name,
                wandb_run=wandb_run,
            )
        if evaluate:
            tsl.eval_local_models(
                config,
                experiment_name=experiment_name,
            )
    finally:
        finish_run(wandb_run)


def run_local_experiments(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    experiments: Iterable[str] = DEFAULT_EXPERIMENTS,
    *,
    train: bool = True,
    evaluate: bool = True,
    log_wandb: bool = False,
    device_ids: Sequence[str] = ("cuda:0", "cuda:1"),
) -> None:
    """Run every (seed, experiment) pair using the provided pool of devices."""
    from experiments.runner_utils import run_experiments_parallel

    tasks: List[Tuple[int, str]] = [
        (seed, experiment) for seed in seeds for experiment in experiments
    ]

    def worker_args_mapper(task: Tuple[int, str], device: str) -> Tuple:
        seed, experiment = task
        print(
            f"Launching local experiment '{experiment}' with seed {seed} on device {device}"
        )
        return (seed, experiment, train, evaluate, log_wandb, device)

    run_experiments_parallel(
        tasks=tasks,
        worker_fn=_run_local_experiment,
        worker_args_mapper=worker_args_mapper,
        device_ids=device_ids,
    )


if __name__ == "__main__":
    run_local_experiments()

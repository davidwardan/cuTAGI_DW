import multiprocessing as mp
from typing import Iterable, Sequence

from experiments.config import Config

DEFAULT_SEEDS: Sequence[int] = (1, 3, 17, 42, 99)
DEFAULT_EXPERIMENTS: Sequence[str] = (
    "train30",
    "train40",
    "train60",
    "train80",
    "train100",
)


def _run_experiment(
    seed: int,
    exp: str,
    train: bool,
    evaluate: bool,
) -> None:
    from experiments import time_series_global as tsg

    torch = tsg.torch

    # Model category
    model_category = "global"
    embed_category = "no-embeddings"

    # Define experiment name
    experiment_name = (
        f"seed{seed}/{exp}/experiment01_{model_category}_{embed_category}"
    )

    # Load configuration
    config = Config.from_yaml(
        f"experiments/configurations/{model_category}_{embed_category}_HQ127.yaml"
    )

    config.seed = seed
    config.model.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.data_paths.x_train = f"data/hq/{exp}/split_train_values.csv"
    config.data_paths.dates_train = f"data/hq/{exp}/split_train_datetimes.csv"

    # Display config
    config.display()

    if train:
        tsg.train_global_model(
            config,
            experiment_name=experiment_name,
        )
    if evaluate:
        tsg.eval_global_model(
            config,
            experiment_name=experiment_name,
        )



def run_experiments(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    experiments: Iterable[str] = DEFAULT_EXPERIMENTS,
    *,
    train: bool = True,
    evaluate: bool = True,
) -> None:
    ctx = mp.get_context("spawn")

    for seed in seeds:
        for experiment in experiments:
            print(f"Launching experiment '{experiment}' with seed {seed}")
            process = ctx.Process(
                target=_run_experiment,
                args=(seed, experiment, train, evaluate),
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

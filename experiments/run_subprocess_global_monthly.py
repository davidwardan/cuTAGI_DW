import multiprocessing as mp
from typing import Iterable, Sequence

from experiments.config import Config

from pytagi import cuda

DEFAULT_SEEDS: Sequence[int] = [11, 42, 235, 2026, 3]
DEFAULT_EXPERIMENTS: Sequence[str] = (
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
    from experiments import stateful_global as parent_script

    # Model category
    model_category = "global"
    embed_category = "no-embeddings"

    # Define experiment name
    experiment_name = f"seed{seed}/{exp}/Filter_{model_category}_{embed_category}_monthly"

    # Load configuration
    config = Config.from_yaml(
        f"experiments/config/{model_category}_{embed_category}_HQ127_monthly.yaml"
    )

    config.seed = seed
    config.model.device = "cuda" if cuda.is_available() else "cpu"
    config.data.paths.x_train = f"data/hq/{exp}/split_train_values.csv"
    config.data.paths.dates_train = f"data/hq/{exp}/split_train_datetimes.csv"
    config.data.loader.train_use_ratio = int(exp.removeprefix("train")) / 100.0

    # Display config
    config.display()

    if train:
        parent_script.train_model(
            config,
            experiment_name=experiment_name,
        )
    if evaluate:
        parent_script.eval_model(
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

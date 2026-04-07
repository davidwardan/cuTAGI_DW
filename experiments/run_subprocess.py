import multiprocessing as mp
from typing import Iterable, Sequence

from experiments.config import Config

from pytagi import cuda

DEFAULT_SEEDS: Sequence[int] = [2016]
# DEFAULT_SEEDS: Sequence[int] = [2016, 2005, 2012]
DEFAULT_TRAIN_USE_RATIOS: Sequence[float] = (
    # 0.35,
    # 0.5,
    # 0.65,
    # 0.8,
    1.0,
)


def _run_experiment(
    seed: int,
    train_use_ratio: float,
    train: bool,
    evaluate: bool,
) -> None:
    from experiments import stateful_global as parent_script

    # Model category
    model_category = "global"
    embed_category = "no-embeddings"
    ratio_tag = f"train_use_{int(round(train_use_ratio * 100)):03d}"

    # Define experiment name
    experiment_name = f"seed{seed}/{ratio_tag}/Testing2_{model_category}"

    # Load configuration
    config = Config.from_yaml(
        f"experiments/config/{model_category}_{embed_category}_HQ127.yaml"
    )

    config.seed = seed
    config.model.device = "cuda" if cuda.is_available() else "cpu"
    config.data.loader.train_use_ratio = train_use_ratio

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
    train_use_ratios: Iterable[float] = DEFAULT_TRAIN_USE_RATIOS,
    *,
    train: bool = True,
    evaluate: bool = True,
) -> None:
    ctx = mp.get_context("spawn")

    for seed in seeds:
        for train_use_ratio in train_use_ratios:
            ratio_tag = f"train_use_{int(round(train_use_ratio * 100)):03d}"
            print(f"Launching experiment '{ratio_tag}' with seed {seed}")
            process = ctx.Process(
                target=_run_experiment,
                args=(seed, train_use_ratio, train, evaluate),
            )
            process.start()
            process.join()

            if process.exitcode:
                raise RuntimeError(
                    f"Experiment '{ratio_tag}' with seed {seed} failed "
                    f"with exit code {process.exitcode}"
                )


if __name__ == "__main__":
    run_experiments()

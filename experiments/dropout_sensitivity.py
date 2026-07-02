"""Series-dropout sensitivity analysis.

Trains the global stateful model at 100% training size while randomly
dropping increasing fractions of the available time series, then reports
epistemic-uncertainty calibration metrics (PICP, MPIW, NLL, CRPS) per
drop fraction.

Each (seed, drop_pct) combination runs in its own subprocess, mirroring
``run_subprocess_global``. After all runs complete, predictions are
re-loaded to build a calibration table (CSV) and reliability curve
(PGF/PDF) under ``experiments/out/dropout_sensitivity``.
"""

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf, erfinv

from experiments.config import Config

from pytagi import cuda

DEFAULT_SEEDS: Sequence[int] = (11, 3, 235)
DEFAULT_DROP_PCTS: Sequence[int] = (0, 10, 20, 30, 40, 50)

MODEL_CATEGORY = "global"
EMBED_CATEGORY = "no-embeddings"
EXPERIMENT_TAG = "dropout_sensitivity"

RUN_ROOT = Path("out") / EXPERIMENT_TAG
OUT_ROOT = Path("experiments/out") / EXPERIMENT_TAG

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)

# Nominal coverage levels used for the reliability curve.
NOMINAL_LEVELS = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)


def _experiment_name(seed: int, drop_pct: int) -> str:
    return (
        f"{EXPERIMENT_TAG}/seed{seed}/drop{drop_pct:02d}/"
        f"{MODEL_CATEGORY}_{EMBED_CATEGORY}"
    )


def _build_config(seed: int, drop_pct: int) -> Config:
    if drop_pct < 0 or drop_pct >= 100:
        raise ValueError("drop_pct must be in [0, 100).")

    config = Config.from_yaml(
        f"experiments/config/{MODEL_CATEGORY}_{EMBED_CATEGORY}_HQ127.yaml"
    )
    config.seed = seed
    config.model.device = "cuda" if cuda.is_available() else "cpu"
    # Fix training size at 100% — sensitivity is over series dropout only.
    config.data.paths.x_train = "data/hq/train100/split_train_values.csv"
    config.data.paths.dates_train = "data/hq/train100/split_train_datetimes.csv"
    config.data.loader.train_use_ratio = 1.0

    total_ts = config.data.loader.nb_ts
    config.training.series_dropout_count = int(round(total_ts * drop_pct / 100.0))
    return config


def _run_experiment(seed: int, drop_pct: int, train: bool, evaluate: bool) -> None:
    from experiments import stateful_global as parent_script

    experiment_name = _experiment_name(seed, drop_pct)
    config = _build_config(seed, drop_pct)
    config.display()

    if train:
        parent_script.train_model(config, experiment_name=experiment_name)
    if evaluate:
        parent_script.eval_model(config, experiment_name=experiment_name)


def run_experiments(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    drop_pcts: Iterable[int] = DEFAULT_DROP_PCTS,
    *,
    train: bool = True,
    evaluate: bool = True,
) -> None:
    ctx = mp.get_context("spawn")
    for seed in seeds:
        for pct in drop_pcts:
            print(f"Launching seed={seed}, drop={pct}%")
            process = ctx.Process(
                target=_run_experiment,
                args=(seed, pct, train, evaluate),
            )
            process.start()
            process.join()
            if process.exitcode:
                raise RuntimeError(
                    f"Run failed: seed={seed}, drop={pct} "
                    f"(exit code {process.exitcode})"
                )


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def _gaussian_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def _gaussian_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _half_width_factor(alpha: float) -> float:
    """Two-sided z for nominal coverage ``alpha``."""
    return math.sqrt(2.0) * erfinv(alpha)


def _coverage_curve(
    y: np.ndarray, mu: np.ndarray, s: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    z = np.abs(y - mu) / s
    return np.array([float(np.mean(z <= _half_width_factor(a))) for a in levels])


def _nll(y: np.ndarray, mu: np.ndarray, s: np.ndarray) -> float:
    return float(
        np.mean(0.5 * np.log(2.0 * math.pi * s * s) + 0.5 * ((y - mu) / s) ** 2)
    )


def _crps_gaussian(y: np.ndarray, mu: np.ndarray, s: np.ndarray) -> float:
    z = (y - mu) / s
    return float(
        np.mean(
            s * (z * (2.0 * _gaussian_cdf(z) - 1.0)
                 + 2.0 * _gaussian_pdf(z)
                 - 1.0 / math.sqrt(math.pi))
        )
    )


def _gather_run_preds(
    seed: int, drop_pct: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Concatenate (y, mu, s) over the test split of all kept series."""
    from experiments.utils import load_true_split_arrays

    run_dir = Path("out") / _experiment_name(seed, drop_pct)
    cfg_path = run_dir / "config.yaml"
    states_path = run_dir / "test_states.npz"
    if not cfg_path.exists() or not states_path.exists():
        return None

    config = Config.from_yaml(str(cfg_path))
    test_states = np.load(states_path)
    _, _, true_test = load_true_split_arrays(**config.true_split_kwargs())
    test_offset = config.split_target_offset("test")

    ys, mus, ss = [], [], []
    for pos in range(config.data.loader.nb_ts):
        yt = true_test[test_offset:, pos]
        valid = ~np.isnan(yt)
        if not np.any(valid):
            continue
        last = int(np.where(valid)[0][-1])
        yt = yt[: last + 1]
        mu_t = test_states["mu"][pos][: len(yt)]
        s_t = test_states["std"][pos][: len(yt)]

        finite = np.isfinite(yt) & np.isfinite(mu_t) & np.isfinite(s_t) & (s_t > 0)
        if not np.any(finite):
            continue
        ys.append(yt[finite])
        mus.append(mu_t[finite])
        ss.append(s_t[finite])

    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(mus), np.concatenate(ss)


def _aggregate_metrics(
    seeds: Sequence[int], drop_pcts: Sequence[int]
) -> Tuple[pd.DataFrame, dict]:
    """Compute per-(seed, drop) metrics + drop-level mean/std.

    Returns the per-run metrics dataframe and a dict mapping drop_pct ->
    (nominal_levels, mean_empirical, std_empirical) for the reliability plot.
    """
    rows = []
    coverage_per_drop = {pct: [] for pct in drop_pcts}

    for seed in seeds:
        for pct in drop_pcts:
            pack = _gather_run_preds(seed, pct)
            if pack is None:
                print(f"[skip] missing outputs for seed={seed}, drop={pct}%")
                continue
            y, mu, s = pack

            picp90 = float(np.mean(np.abs(y - mu) <= _half_width_factor(0.90) * s))
            picp95 = float(np.mean(np.abs(y - mu) <= _half_width_factor(0.95) * s))
            mpiw90 = float(np.mean(2.0 * _half_width_factor(0.90) * s))
            nll = _nll(y, mu, s)
            crps = _crps_gaussian(y, mu, s)
            # Mean absolute calibration error across NOMINAL_LEVELS.
            emp = _coverage_curve(y, mu, s, NOMINAL_LEVELS)
            mace = float(np.mean(np.abs(emp - NOMINAL_LEVELS)))
            coverage_per_drop[pct].append(emp)

            rows.append({
                "seed": seed,
                "drop_pct": pct,
                "n_obs": int(y.size),
                "PICP90": picp90,
                "PICP95": picp95,
                "MPIW90": mpiw90,
                "NLL": nll,
                "CRPS": crps,
                "MACE": mace,
            })

    per_run = pd.DataFrame(rows)

    coverage_summary = {}
    for pct, curves in coverage_per_drop.items():
        if not curves:
            continue
        stack = np.stack(curves, axis=0)
        coverage_summary[pct] = (
            NOMINAL_LEVELS,
            stack.mean(axis=0),
            stack.std(axis=0, ddof=0),
        )

    return per_run, coverage_summary


def _summary_table(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["PICP90", "PICP95", "MPIW90", "NLL", "CRPS", "MACE"]
    grouped = per_run.groupby("drop_pct")[metric_cols]
    summary = grouped.agg(["mean", "std"]).round(4)
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def _apply_plot_defaults() -> None:
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": (
            r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}"
        ),
        "lines.linewidth": 1,
        "figure.figsize": SINGLE_COL,
        "font.size": 9,
        "savefig.dpi": 300,
    })


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pgf"), bbox_inches="tight")


def _plot_reliability(coverage_summary: dict, stem: Path) -> None:
    if not coverage_summary:
        print("No coverage data to plot.")
        return

    _apply_plot_defaults()
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.8)

    pcts = sorted(coverage_summary.keys())
    cmap = plt.get_cmap("viridis")
    for k, pct in enumerate(pcts):
        nominal, mean_emp, std_emp = coverage_summary[pct]
        color = cmap(k / max(len(pcts) - 1, 1))
        ax.plot(nominal, mean_emp, marker="o", markersize=3, color=color,
                label=fr"drop {pct}\%")
        ax.fill_between(nominal, mean_emp - std_emp, mean_emp + std_emp,
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=7)
    fig.tight_layout()
    _save_fig(fig, stem)
    plt.close(fig)


def aggregate_and_report(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    drop_pcts: Sequence[int] = DEFAULT_DROP_PCTS,
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    per_run, coverage_summary = _aggregate_metrics(seeds, drop_pcts)
    if per_run.empty:
        print("No runs found; nothing to aggregate.")
        return

    per_run_path = OUT_ROOT / "calibration_per_run.csv"
    per_run.to_csv(per_run_path, index=False)
    print(f"Wrote {per_run_path}")

    summary = _summary_table(per_run)
    summary_path = OUT_ROOT / "calibration_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    _plot_reliability(coverage_summary, OUT_ROOT / "calibration_curve")
    print(f"Wrote reliability curve under {OUT_ROOT}")

    print("\nCalibration summary (mean over seeds):")
    print(summary.to_string(index=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--drop-pcts", type=int, nargs="+", default=list(DEFAULT_DROP_PCTS)
    )
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training subprocesses.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation inside subprocesses.")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip the final calibration aggregation step.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (args.no_train and args.no_eval):
        run_experiments(
            seeds=args.seeds,
            drop_pcts=args.drop_pcts,
            train=not args.no_train,
            evaluate=not args.no_eval,
        )
    if not args.no_aggregate:
        aggregate_and_report(seeds=args.seeds, drop_pcts=args.drop_pcts)


if __name__ == "__main__":
    main()

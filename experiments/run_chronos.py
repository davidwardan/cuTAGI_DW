"""Run Chronos-2 zero-shot forecasting on the HQ127 dataset.

For each seed / train_use_ratio combination the script:
    1. Splits each series into train / val / test (same logic as run_subprocess).
    2. Feeds [train (truncated by ratio) + val] as context to Chronos-2.
    3. Generates a multi-step probabilistic forecast over the test horizon.
    4. Evaluates with the same metrics (RMSE, LogLik, MAE, P50, P90) and
       saves ``evaluation_metrics.txt`` in the same format.

Notes:
    - Chronos-2 returns forecast quantiles through ``predict_df``.
    - The Gaussian std used for LogLik / P90 is approximated from the
      inter-quantile range: std = (q90 - q10) / (2 * z_0.9).

Usage:
    python -m experiments.run_chronos

Requires:
    pip install "chronos-forecasting>=2.0"
"""

import multiprocessing as mp
import ctypes
from pathlib import Path
import sysconfig
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _preload_torch_cuda_deps() -> None:
    """Prefer PyTorch wheel CUDA libs over an incompatible system CUDA path."""
    nvidia_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for rel_path in (
        "nvjitlink/lib/libnvJitLink.so.12",
        "cusparse/lib/libcusparse.so.12",
    ):
        lib_path = nvidia_root / rel_path
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_torch_cuda_deps()

import torch

from pytagi import Normalizer as normalizer
import pytagi.metric as metric

# ---------------------------------------------------------------------------
# Defaults (same as run_subprocess.py)
# ---------------------------------------------------------------------------
DEFAULT_SEEDS: Sequence[int] = [2016, 17, 42]
DEFAULT_TRAIN_USE_RATIOS: Sequence[float] = (0.4, 0.6, 0.8, 1.0)

# Data paths (matching locals_HQ127.yaml)
X_FULL = "data/hq/ts_weekly_values_final.csv"
SPLIT_TRAIN_RATIO = 0.7
SPLIT_VAL_RATIO = 0.15
NB_TS = 100

# Chronos settings
CHRONOS_MODEL = "amazon/chronos-2"
_Q10_COL = "0.1"
_Q50_COL = "0.5"
_Q90_COL = "0.9"
_Z90 = 1.2815515655446004  # z-score for 90th percentile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> np.ndarray:
    """Load CSV, skip header, return (T, N) array."""
    df = pd.read_csv(path, skiprows=1, delimiter=",", header=None)
    return df.values


def _trim_trailing_nans(x: np.ndarray) -> np.ndarray:
    """Trim trailing NaNs from a 1-D array."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    valid = ~np.isnan(x)
    if not np.any(valid):
        return np.array([], dtype=np.float32)
    last = np.where(valid)[0][-1]
    return x[: last + 1]


def _split_series(x_col, train_ratio, val_ratio, train_use_ratio):
    """Split one series into train / val / test.

    Mirrors ``_split_single_series`` in experiments/utils.py so the
    evaluation is on exactly the same data points.
    """
    x = _trim_trailing_nans(x_col)
    n = len(x)
    if n < 3:
        return {
            "full_train": x,
            "train": x,
            "val": np.array([], dtype=np.float32),
            "test": np.array([], dtype=np.float32),
        }

    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1

    train_end = n_train
    val_end = n_train + n_val

    full_train = x[:train_end]
    n_used = int(np.floor(len(full_train) * train_use_ratio))
    n_used = min(len(full_train), max(1, n_used))
    train = full_train[-n_used:]

    return {
        "full_train": full_train,
        "train": train,
        "val": x[train_end:val_end],
        "test": x[val_end:],
    }


def _build_context_df(ts_id: int, context: np.ndarray) -> pd.DataFrame:
    """Build the Chronos-2 dataframe input for a single weekly series."""
    return pd.DataFrame(
        {
            "id": np.repeat(str(ts_id), len(context)),
            "timestamp": pd.date_range(
                start="2000-01-02",
                periods=len(context),
                freq="W",
            ),
            "target": context.astype(np.float32),
        }
    )


def _plot_series_if_available(**plot_kwargs) -> None:
    """Plot forecasts when the optional plotting stack is installed."""
    try:
        from experiments.utils import plot_series
    except ImportError as exc:
        if not getattr(_plot_series_if_available, "_warned", False):
            print(f"Skipping Chronos plots because plotting deps are unavailable: {exc}")
            _plot_series_if_available._warned = True
        return
    plot_series(**plot_kwargs)


# ---------------------------------------------------------------------------
# Single experiment (runs inside a subprocess)
# ---------------------------------------------------------------------------
def _run_experiment(seed: int, train_use_ratio: float) -> None:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos-2 requires `chronos-forecasting>=2.0`. "
            'Install it with `pip install "chronos-forecasting>=2.0"`.'
        ) from exc

    ratio_tag = f"train_use_{int(round(train_use_ratio * 100)):03d}"
    experiment_name = f"seed{seed}/{ratio_tag}/Forecasting_CHRONOS"
    out_dir = Path(f"experiments/out/{experiment_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Chronos-2 | seed={seed}  ratio={train_use_ratio}")
    print(f"{'='*60}")

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL,
        device_map=device,
    )

    # Load data
    X_all = _load_csv(X_FULL)  # (T, N)
    ts_indices = list(range(NB_TS))

    # Per-series metric accumulators
    rmse_list, loglik_list, mae_list, p50_list, p90_list = [], [], [], [], []
    # Micro-average accumulators
    all_stand_true, all_stand_pred, all_stand_std = [], [], []
    all_true, all_pred, all_std = [], [], []

    for ts_id in ts_indices:
        split = _split_series(
            X_all[:, ts_id],
            train_ratio=SPLIT_TRAIN_RATIO,
            val_ratio=SPLIT_VAL_RATIO,
            train_use_ratio=train_use_ratio,
        )

        test = split["test"]
        prediction_length = len(test)
        if prediction_length == 0:
            # Append NaN so indexing stays aligned with ts_indices
            rmse_list.append(np.nan)
            loglik_list.append(np.nan)
            mae_list.append(np.nan)
            p50_list.append(np.nan)
            p90_list.append(np.nan)
            continue

        # Context = truncated train + val
        context = np.concatenate([split["train"], split["val"]]).astype(np.float32)
        context_df = _build_context_df(ts_id, context)

        # Forecast quantiles with the Chronos-2 dataframe API
        with torch.no_grad():
            forecast_df = pipeline.predict_df(
                context_df,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
        pred_mean = forecast_df[_Q50_COL].to_numpy(dtype=np.float32)
        q10 = forecast_df[_Q10_COL].to_numpy(dtype=np.float32)
        q90 = forecast_df[_Q90_COL].to_numpy(dtype=np.float32)
        # Derive Gaussian std from inter-quantile range
        pred_std = ((q90 - q10) / (2 * _Z90)).astype(np.float32)
        pred_std = np.maximum(pred_std, 1e-6)

        # Standardize using full training statistics (same as eval_model)
        full_train = split["full_train"]
        train_mean = np.nanmean(full_train)
        train_std_val = np.nanstd(full_train)

        stand_true = normalizer.standardize(test, train_mean, train_std_val)
        stand_pred = normalizer.standardize(pred_mean, train_mean, train_std_val)
        stand_s = normalizer.standardize_std(pred_std, train_std_val)

        # Metrics in standardized space
        test_rmse = metric.rmse(stand_pred, stand_true)
        test_loglik = metric.log_likelihood(stand_pred, stand_true, stand_s)
        test_mae = metric.mae(stand_pred, stand_true)

        # Metrics in original space (normalized pinball)
        test_p50 = metric.Np50(test, pred_mean)
        test_p90 = metric.Np90(test, pred_mean, pred_std)

        rmse_list.append(test_rmse)
        loglik_list.append(test_loglik)
        mae_list.append(test_mae)
        p50_list.append(test_p50)
        p90_list.append(test_p90)

        all_stand_true.append(stand_true)
        all_stand_pred.append(stand_pred)
        all_stand_std.append(stand_s)
        all_true.append(test)
        all_pred.append(pred_mean)
        all_std.append(pred_std)

        # Plot forecast for full training ratio
        if train_use_ratio == 1.0:
            full_y_true = np.concatenate([split["train"], split["val"], test])
            n_ctx = len(split["train"]) + len(split["val"])

            full_y_pred = np.full(len(full_y_true), np.nan)
            full_y_pred[n_ctx:] = pred_mean

            full_std = np.full(len(full_y_true), np.nan)
            full_std[n_ctx:] = pred_std

            val_test_indices = (len(split["train"]), n_ctx)

            _plot_series_if_available(
                ts_idx=ts_id,
                y_true=full_y_true,
                y_pred=full_y_pred,
                s_pred=full_std,
                epistemic_std=full_std,
                aleatoric_std=np.zeros_like(full_std),
                out_dir=out_dir / "figures",
                val_test_indices=val_test_indices,
                std_factor=1,
            )

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    macro_rmse = np.nanmean(rmse_list)
    macro_loglik = np.nanmean(loglik_list)
    macro_mae = np.nanmean(mae_list)
    macro_p50 = np.nanmean(p50_list)
    macro_p90 = np.nanmean(p90_list)

    full_stand_true = np.concatenate(all_stand_true)
    full_stand_pred = np.concatenate(all_stand_pred)
    full_stand_std = np.concatenate(all_stand_std)
    full_true = np.concatenate(all_true)
    full_pred = np.concatenate(all_pred)
    full_std = np.concatenate(all_std)

    micro_rmse = metric.rmse(full_stand_pred, full_stand_true)
    micro_loglik = metric.log_likelihood(
        full_stand_pred, full_stand_true, full_stand_std
    )
    micro_mae = metric.mae(full_stand_pred, full_stand_true)
    micro_p50 = metric.Np50(full_true, full_pred)
    micro_p90 = metric.Np90(full_true, full_pred, full_std)

    # ------------------------------------------------------------------
    # Save results (identical format to eval_model in stateful_locals.py)
    # ------------------------------------------------------------------
    with open(out_dir / "evaluation_metrics.txt", "w") as f:
        f.write("Series_ID,RMSE,LogLik,MAE,P50,P90\n")
        for i, ts_id in enumerate(ts_indices):
            f.write(
                f"{ts_id},{rmse_list[i]:.4f},{loglik_list[i]:.4f},"
                f"{mae_list[i]:.4f},{p50_list[i]:.4f},"
                f"{p90_list[i]:.4f}\n"
            )
        f.write(
            f"Macro_Average,{macro_rmse:.4f},{macro_loglik:.4f},"
            f"{macro_mae:.4f},{macro_p50:.4f},"
            f"{macro_p90:.4f}\n"
        )
        f.write(
            f"Micro_Average,{micro_rmse:.4f},{micro_loglik:.4f},"
            f"{micro_mae:.4f},{micro_p50:.4f},"
            f"{micro_p90:.4f}\n"
        )

    print(
        f"Done: {experiment_name} | "
        f"Macro RMSE={macro_rmse:.4f}  MAE={macro_mae:.4f}  LogLik={macro_loglik:.4f}"
    )


# ---------------------------------------------------------------------------
# Main loop (mirrors run_subprocess.py)
# ---------------------------------------------------------------------------
def run_experiments(
    seeds: Iterable[int] = DEFAULT_SEEDS,
    train_use_ratios: Iterable[float] = DEFAULT_TRAIN_USE_RATIOS,
) -> None:
    ctx = mp.get_context("spawn")

    for seed in seeds:
        for train_use_ratio in train_use_ratios:
            ratio_tag = f"train_use_{int(round(train_use_ratio * 100)):03d}"
            print(f"Launching Chronos experiment '{ratio_tag}' with seed {seed}")
            process = ctx.Process(
                target=_run_experiment,
                args=(seed, train_use_ratio),
            )
            process.start()
            process.join()

            if process.exitcode:
                raise RuntimeError(
                    f"Chronos experiment '{ratio_tag}' with seed {seed} failed "
                    f"with exit code {process.exitcode}"
                )


if __name__ == "__main__":
    run_experiments()

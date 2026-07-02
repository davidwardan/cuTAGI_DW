"""Architecture sensitivity for global vs local LSTM models.

Sweeps ``(units, layers)`` for both training modes to identify each
method's optimal architecture before the head-to-head comparison.

Phase 0 — Stratified subset selection (K-means on per-series mean, std,
length, lag-1 ACF). Saves a deterministic ~25-series subset.

Phase A — Local sweep on the subset: for each architecture, train one
local model per subset series and compute standardized macro validation
RMSE, NLL, and CRPS across the subset.

Phase B — Global sweep on all series: for each architecture, train one
global model on all 101 series and compute standardized macro validation
RMSE, NLL, and CRPS.

Phase C — Final comparison: re-train each method's winning architecture
with multiple seeds on all series and evaluate on the test split.

Each (mode, arch, seed[, scope]) combination runs in its own subprocess
(spawn), mirroring ``dropout_sensitivity.py``. Outputs land under
``experiments/out/arch_sensitivity``; per-run training artefacts under
``out/arch_sensitivity``.
"""

import argparse
import json
import math
import multiprocessing as mp
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.config import Config
from experiments.utils import load_true_split_arrays

import pytagi.metric as metric
from pytagi import cuda
from pytagi import Normalizer as normalizer

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

UNITS: Sequence[int] = (32, 64, 128, 256)
LAYERS: Sequence[int] = (1, 2, 3)

SWEEP_SEEDS: Sequence[int] = (1, 2, 3, 4, 5)
FINAL_SEEDS: Sequence[int] = (1, 2, 3, 4, 5)

# Subset selection
SUBSET_K_CLUSTERS = 5
SUBSET_PER_CLUSTER = 5  # K * per_cluster = 25 series
SUBSET_SEED = 0

GLOBAL_CONFIG = "experiments/config/global_no-embeddings_HQ127.yaml"
LOCAL_CONFIG = "experiments/config/locals_HQ127.yaml"

EXPERIMENT_TAG = "arch_sensitivity"
RUN_ROOT = Path("out") / EXPERIMENT_TAG
OUT_ROOT = Path("experiments/out") / EXPERIMENT_TAG

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)

VAL_METRICS = ("val_rmse", "val_nll", "val_crps")
METRIC_LABELS = {
    "val_rmse": "RMSE",
    "val_nll": "NLL",
    "val_crps": "CRPS",
}


def _arch_tag(units: int, layers: int) -> str:
    return f"u{units:03d}_l{layers}"


def _hidden_sizes(units: int, layers: int) -> List[int]:
    return [int(units)] * int(layers)


def _device_name() -> str:
    try:
        return "cuda" if cuda.is_available() else "cpu"
    except RuntimeError:
        return "cpu"


def _experiment_name(mode: str, units: int, layers: int, seed: int,
                     phase: str) -> str:
    return f"{EXPERIMENT_TAG}/{phase}/{mode}/{_arch_tag(units, layers)}/seed{seed}"


# ---------------------------------------------------------------------------
# Phase 0 — subset selection
# ---------------------------------------------------------------------------


def _per_series_features(values: np.ndarray) -> np.ndarray:
    """Return shape (n_series, 4) of (mean, std, length, lag-1 ACF).

    ``values`` is the unscaled training portion of each series (NaNs allowed).
    """
    n_series = values.shape[1]
    feats = np.zeros((n_series, 4), dtype=np.float64)
    for j in range(n_series):
        col = values[:, j]
        valid = ~np.isnan(col)
        v = col[valid]
        if v.size < 3:
            feats[j] = (np.nan, np.nan, 0.0, 0.0)
            continue
        mean = float(np.mean(v))
        std = float(np.std(v))
        length = float(v.size)
        if std > 0:
            vc = v - mean
            denom = float(np.dot(vc, vc))
            ar1 = float(np.dot(vc[:-1], vc[1:]) / denom) if denom > 0 else 0.0
        else:
            ar1 = 0.0
        feats[j] = (mean, std, length, ar1)
    return feats


def _kmeans_numpy(
    x: np.ndarray, k: int, *, seed: int, n_init: int = 10, max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ init. Returns (labels, centers).

    Hand-rolled because the local sklearn install is broken against numpy
    here. Adequate for K=5 on ~100 4-dim points.
    """
    rng = np.random.default_rng(seed)
    best_labels, best_centers, best_inertia = None, None, np.inf
    n = x.shape[0]
    for _ in range(n_init):
        # k-means++ init
        first = int(rng.integers(0, n))
        centers = [x[first]]
        dists = np.sum((x - centers[0]) ** 2, axis=1)
        for _c in range(1, k):
            probs = dists / max(dists.sum(), 1e-12)
            nxt = int(rng.choice(n, p=probs))
            centers.append(x[nxt])
            new_d = np.sum((x - centers[-1]) ** 2, axis=1)
            dists = np.minimum(dists, new_d)
        centers = np.stack(centers, axis=0)

        for _it in range(max_iter):
            # assign
            d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d, axis=1)
            # update
            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = labels == c
                if not np.any(mask):
                    # re-seed empty cluster to the farthest point
                    far = int(np.argmax(np.min(d, axis=1)))
                    new_centers[c] = x[far]
                else:
                    new_centers[c] = x[mask].mean(axis=0)
            if np.allclose(new_centers, centers, atol=1e-8):
                centers = new_centers
                break
            centers = new_centers
        inertia = float(
            np.sum((x - centers[labels]) ** 2)
        )
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels, centers
    return best_labels, best_centers


def _select_subset(
    config_path: str, *, k_clusters: int, per_cluster: int, seed: int
) -> dict:
    """Stratify the 101 series by per-series features and pick a subset."""
    config = Config.from_yaml(config_path)
    true_train, _, _ = load_true_split_arrays(
        **config.true_split_kwargs(ts_to_use=list(range(config.data.loader.nb_ts)))
    )
    feats = _per_series_features(true_train)

    # Drop series with NaN features (too short / all NaN).
    keep = np.isfinite(feats).all(axis=1)
    valid_idx = np.where(keep)[0]
    feats_valid = feats[keep]

    mu = feats_valid.mean(axis=0)
    sd = feats_valid.std(axis=0)
    sd[sd == 0] = 1.0
    x_std = (feats_valid - mu) / sd

    labels, _ = _kmeans_numpy(x_std, k=k_clusters, seed=seed)

    rng = np.random.default_rng(seed)
    target = int(k_clusters) * int(per_cluster)
    picked: List[int] = []
    leftovers: List[int] = []
    assignments = {int(c): [] for c in range(k_clusters)}
    for c in range(k_clusters):
        members = valid_idx[labels == c].tolist()
        assignments[c] = members
        rng.shuffle(members)
        picked.extend(members[:per_cluster])
        leftovers.extend(members[per_cluster:])

    # Top up to the target size from leftover members of the larger clusters
    # so a tiny cluster does not deflate the subset.
    rng.shuffle(leftovers)
    while len(picked) < target and leftovers:
        picked.append(leftovers.pop())

    picked = sorted(int(p) for p in picked)
    return {
        "subset": picked,
        "feature_names": ["mean", "std", "length", "ar1"],
        "features": feats.tolist(),
        "cluster_labels": {
            int(i): int(labels[np.where(valid_idx == i)[0][0]])
            for i in valid_idx
        },
        "cluster_members": assignments,
        "k_clusters": int(k_clusters),
        "per_cluster": int(per_cluster),
        "seed": int(seed),
    }


def _load_or_build_subset() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    subset_path = OUT_ROOT / "subset.json"
    if subset_path.exists():
        with open(subset_path) as f:
            return json.load(f)
    info = _select_subset(
        LOCAL_CONFIG,
        k_clusters=SUBSET_K_CLUSTERS,
        per_cluster=SUBSET_PER_CLUSTER,
        seed=SUBSET_SEED,
    )
    with open(subset_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {subset_path} (subset = {info['subset']})")
    return info


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _build_config(
    mode: str, units: int, layers: int, seed: int,
    ts_to_use: Optional[List[int]] = None,
) -> Config:
    base = GLOBAL_CONFIG if mode == "global" else LOCAL_CONFIG
    config = Config.from_yaml(base)
    config.seed = int(seed)
    config.model.device = _device_name()
    config.model.hidden_sizes = _hidden_sizes(units, layers)
    if ts_to_use is not None:
        config.data.loader.ts_to_use = [int(t) for t in ts_to_use]
    # Always train on full training split.
    config.data.paths.x_train = "data/hq/train100/split_train_values.csv"
    config.data.paths.dates_train = "data/hq/train100/split_train_datetimes.csv"
    config.data.loader.train_use_ratio = 1.0
    return config


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _run_in_subprocess(
    mode: str, units: int, layers: int, seed: int,
    ts_to_use: Optional[List[int]], phase: str,
    train: bool, evaluate: bool,
) -> None:
    if mode == "global":
        from experiments import stateful_global as parent
    elif mode == "local":
        from experiments import stateful_locals as parent
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    experiment_name = _experiment_name(mode, units, layers, seed, phase)
    config = _build_config(mode, units, layers, seed, ts_to_use=ts_to_use)
    config.display()

    if train:
        parent.train_model(config, experiment_name=experiment_name)
    if evaluate:
        parent.eval_model(config, experiment_name=experiment_name)


def _spawn(
    mode: str, units: int, layers: int, seed: int,
    ts_to_use: Optional[List[int]], phase: str,
    *, train: bool = True, evaluate: bool = False,
) -> None:
    ctx = mp.get_context("spawn")
    print(
        f"[run] phase={phase} mode={mode} arch={_arch_tag(units, layers)} "
        f"seed={seed} n_series={len(ts_to_use) if ts_to_use is not None else 'all'}"
    )
    proc = ctx.Process(
        target=_run_in_subprocess,
        args=(mode, units, layers, seed, ts_to_use, phase, train, evaluate),
    )
    proc.start()
    proc.join()
    if proc.exitcode:
        raise RuntimeError(
            f"Run failed: phase={phase} mode={mode} arch={_arch_tag(units, layers)} "
            f"seed={seed} (exit code {proc.exitcode})"
        )


# ---------------------------------------------------------------------------
# Validation metric re-computation from saved states
# ---------------------------------------------------------------------------


def _nll(y: np.ndarray, mu: np.ndarray, s: np.ndarray) -> float:
    return float(
        np.mean(0.5 * np.log(2.0 * math.pi * s * s) + 0.5 * ((y - mu) / s) ** 2)
    )


def _trim_trailing_nans(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    valid = ~np.isnan(x)
    if not np.any(valid):
        return np.array([], dtype=np.float32)
    return x[: int(np.where(valid)[0][-1]) + 1].astype(np.float32)


def _per_series_val_metrics(
    mode: str, run_dir: Path, ts_to_use: Optional[List[int]],
) -> Optional[List[dict]]:
    """Compute standardized per-series validation metrics from a finished run.

    ``ts_to_use`` must match what was passed to training (None = all). We
    rebuild the eval config from source rather than reading the dumped
    ``config.yaml`` because the dump emits Python-tuple tags that
    ``yaml.safe_load`` cannot parse.

    Returns a list of per-series metric rows, or None when artefacts are
    missing.
    """
    val_path = run_dir / "val_states.npz"
    if not val_path.exists():
        return None

    # Arch hyperparameters do not affect the data split — pass placeholders.
    config = _build_config(mode, units=25, layers=1, seed=0, ts_to_use=ts_to_use)
    val_states = np.load(val_path)
    true_train, true_val, _ = load_true_split_arrays(**config.true_split_kwargs())
    val_offset = config.split_target_offset("val")
    train_offset = config.split_target_offset("train")

    results: List[dict] = []
    for local_idx, ts_id in enumerate(config.ts_to_use):
        yt_val = _trim_trailing_nans(true_val[val_offset:, local_idx])
        yt_train = _trim_trailing_nans(true_train[train_offset:, local_idx])
        if yt_val.size == 0 or yt_train.size == 0:
            continue

        # Global stores states at local position; locals at global ts_id.
        state_idx = local_idx if mode == "global" else int(ts_id)
        mu = val_states["mu"][state_idx][: len(yt_val)]
        s = val_states["std"][state_idx][: len(yt_val)]

        finite = (
            np.isfinite(yt_val) & np.isfinite(mu) & np.isfinite(s) & (s > 0)
        )
        if not np.any(finite):
            continue
        yt_val = yt_val[finite]
        mu = mu[finite]
        s = s[finite]

        # Standardize with per-series train mean/std for cross-series comparability.
        train_mean = float(np.nanmean(yt_train))
        train_std = float(np.nanstd(yt_train))
        if train_std <= 0:
            continue
        y_std = normalizer.standardize(yt_val, train_mean, train_std)
        mu_std = normalizer.standardize(mu, train_mean, train_std)
        s_std = normalizer.standardize_std(s, train_std)

        results.append({
            "ts_id": int(ts_id),
            "val_rmse": float(metric.rmse(mu_std, y_std)),
            "val_nll": _nll(y_std, mu_std, s_std),
            "val_crps": float(metric.CRPS(y_std, mu_std, s_std)),
            "n_obs": int(yt_val.size),
        })
    return results


# ---------------------------------------------------------------------------
# Phase A & B — sweeps
# ---------------------------------------------------------------------------


def _arch_grid() -> List[Tuple[int, int]]:
    return [(int(u), int(l)) for u, l in product(UNITS, LAYERS)]


def run_sweep(
    mode: str, *, subset: Optional[List[int]],
    archs: Iterable[Tuple[int, int]], seeds: Sequence[int], train: bool,
) -> None:
    phase = "sweep"
    ts = subset if mode == "local" else None
    for seed in seeds:
        for units, layers in archs:
            _spawn(mode, units, layers, seed, ts, phase, train=train, evaluate=False)


def aggregate_sweep(
    mode: str, *, subset: Optional[List[int]],
    archs: Sequence[Tuple[int, int]], seeds: Sequence[int],
) -> pd.DataFrame:
    phase = "sweep"
    rows = []
    for seed in seeds:
        for units, layers in archs:
            run_dir = Path("out") / _experiment_name(mode, units, layers, seed, phase)
            pack = _per_series_val_metrics(mode, run_dir, ts_to_use=subset)
            if pack is None:
                print(f"[skip] missing artefacts: {run_dir}")
                continue
            for metrics_row in pack:
                rows.append({
                    "mode": mode,
                    "units": units,
                    "layers": layers,
                    "arch": _arch_tag(units, layers),
                    "seed": seed,
                    **metrics_row,
                })
    return pd.DataFrame(rows)


def _summarize_sweep(df: pd.DataFrame, sort_metric: str = "val_nll") -> pd.DataFrame:
    """Summarize per-series standardized metrics as macro averages."""
    if df.empty:
        return df
    if sort_metric not in VAL_METRICS:
        raise ValueError(f"Unknown metric {sort_metric!r}; expected one of {VAL_METRICS}.")

    agg_spec = {
        "n_series": ("ts_id", "nunique"),
    }
    for metric_col in VAL_METRICS:
        agg_spec[f"{metric_col}_macro"] = (metric_col, "mean")
        agg_spec[f"{metric_col}_median"] = (metric_col, "median")
        agg_spec[f"{metric_col}_std"] = (metric_col, "std")

    out = (
        df.groupby(["mode", "units", "layers", "arch"])
        .agg(**agg_spec)
        .reset_index()
    )
    return out.sort_values(f"{sort_metric}_macro")


def _best_by_metric(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    rows = []
    for metric_col in VAL_METRICS:
        sort_col = f"{metric_col}_macro"
        best = summary.sort_values(sort_col).iloc[0]
        rows.append({
            "mode": best["mode"],
            "metric": METRIC_LABELS[metric_col],
            "units": int(best["units"]),
            "layers": int(best["layers"]),
            "arch": best["arch"],
            "standardized_macro_value": float(best[sort_col]),
            "lower_is_better": True,
        })
    return pd.DataFrame(rows)


def _best_by_metric_payload(df: pd.DataFrame) -> dict:
    payload = {}
    for _, row in df.iterrows():
        mode = str(row["mode"])
        metric_name = str(row["metric"]).lower()
        payload.setdefault(mode, {})[metric_name] = {
            "units": int(row["units"]),
            "layers": int(row["layers"]),
            "arch": str(row["arch"]),
            "standardized_macro_value": float(row["standardized_macro_value"]),
            "lower_is_better": bool(row["lower_is_better"]),
        }
    return payload


def _rank_distribution(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """For each series, rank archs by val_nll. Report how often each arch
    lands in the top-K across series. A healthy local-tuning regime has a
    clear modal winner — if not, "one local arch fits all" is suspect.
    """
    if df.empty:
        return df
    df = df.copy()
    # Average across sweep seeds so each (series, arch) is ranked once.
    df = df.groupby(["ts_id", "arch"], as_index=False)["val_nll"].mean()
    df["rank"] = (
        df.groupby("ts_id")["val_nll"].rank(method="min", ascending=True)
    )
    n_series = df["ts_id"].nunique()
    rows = []
    for arch, grp in df.groupby("arch"):
        rank1 = int(np.sum(grp["rank"] == 1))
        topk = int(np.sum(grp["rank"] <= top_k))
        rows.append({
            "arch": arch,
            "rank1_count": rank1,
            "rank1_frac": rank1 / max(n_series, 1),
            f"top{top_k}_count": topk,
            f"top{top_k}_frac": topk / max(n_series, 1),
        })
    return pd.DataFrame(rows).sort_values("rank1_count", ascending=False)


# ---------------------------------------------------------------------------
# Phase C — final comparison
# ---------------------------------------------------------------------------


def _parse_metrics_txt(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def run_final(
    winners: dict, *, seeds: Sequence[int], train: bool, evaluate: bool,
) -> None:
    phase = "final"
    for mode, (units, layers) in winners.items():
        for seed in seeds:
            _spawn(mode, units, layers, seed, None, phase,
                   train=train, evaluate=evaluate)


def aggregate_final(winners: dict, *, seeds: Sequence[int]) -> pd.DataFrame:
    phase = "final"
    rows = []
    for mode, (units, layers) in winners.items():
        for seed in seeds:
            run_dir = Path("out") / _experiment_name(
                mode, units, layers, seed, phase
            )
            df = _parse_metrics_txt(run_dir / "evaluation_metrics.txt")
            if df is None:
                print(f"[skip] missing metrics: {run_dir}")
                continue
            df = df.copy()
            df["mode"] = mode
            df["units"] = units
            df["layers"] = layers
            df["arch"] = _arch_tag(units, layers)
            df["seed"] = seed
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


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


def _plot_arch_heatmap(summary: pd.DataFrame, stem: Path, label: str) -> None:
    if summary.empty:
        print(f"No data for {label} heatmap.")
        return
    _apply_plot_defaults()

    units_sorted = sorted(summary["units"].unique())
    layers_sorted = sorted(summary["layers"].unique())
    grid = np.full((len(layers_sorted), len(units_sorted)), np.nan)
    for _, row in summary.iterrows():
        i = layers_sorted.index(int(row["layers"]))
        j = units_sorted.index(int(row["units"]))
        grid[i, j] = row["val_nll_macro"]

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    im = ax.imshow(grid, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(range(len(units_sorted)))
    ax.set_xticklabels(units_sorted)
    ax.set_yticks(range(len(layers_sorted)))
    ax.set_yticklabels(layers_sorted)
    ax.set_xlabel("Units per layer")
    ax.set_ylabel("Layers")

    # Mark the winner.
    best = summary.iloc[0]
    bi = layers_sorted.index(int(best["layers"]))
    bj = units_sorted.index(int(best["units"]))
    ax.scatter([bj], [bi], marker="*", s=70, edgecolor="white",
               facecolor="tab:red", linewidths=0.6)

    # Annotate cells.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Standardized macro val NLL ({label})")
    fig.tight_layout()
    _save_fig(fig, stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def aggregate_and_report(
    subset: List[int], *, archs: Sequence[Tuple[int, int]],
    sweep_seeds: Sequence[int], final_seeds: Sequence[int],
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    best_tables = []

    # Phase A — local sweep
    local_df = aggregate_sweep("local", subset=subset, archs=archs, seeds=sweep_seeds)
    if not local_df.empty:
        local_df.to_csv(OUT_ROOT / "local_sweep_per_series.csv", index=False)
        local_summary = _summarize_sweep(local_df)
        local_summary.to_csv(OUT_ROOT / "local_sweep.csv", index=False)
        local_best = _best_by_metric(local_summary)
        best_tables.append(local_best)
        local_best.to_csv(OUT_ROOT / "local_best_by_metric.csv", index=False)
        rank_df = _rank_distribution(local_df)
        rank_df.to_csv(OUT_ROOT / "local_rank_distribution.csv", index=False)
        _plot_arch_heatmap(local_summary, OUT_ROOT / "arch_heatmap_local", "local")
        print("\nLocal sweep summary (sorted by standardized macro val NLL):")
        print(local_summary.to_string(index=False))
        print("\nLocal best architecture by standardized macro metric:")
        print(local_best.to_string(index=False))
        print("\nLocal rank distribution (rank-1 frequency across subset):")
        print(rank_df.to_string(index=False))

    # Phase B — global sweep
    global_df = aggregate_sweep("global", subset=None, archs=archs, seeds=sweep_seeds)
    if not global_df.empty:
        global_df.to_csv(OUT_ROOT / "global_sweep_per_series.csv", index=False)
        global_summary = _summarize_sweep(global_df)
        global_summary.to_csv(OUT_ROOT / "global_sweep.csv", index=False)
        global_best = _best_by_metric(global_summary)
        best_tables.append(global_best)
        global_best.to_csv(OUT_ROOT / "global_best_by_metric.csv", index=False)
        _plot_arch_heatmap(global_summary, OUT_ROOT / "arch_heatmap_global", "global")
        print("\nGlobal sweep summary (sorted by standardized macro val NLL):")
        print(global_summary.to_string(index=False))
        print("\nGlobal best architecture by standardized macro metric:")
        print(global_best.to_string(index=False))

    if best_tables:
        best_by_metric = pd.concat(best_tables, ignore_index=True)
        best_by_metric.to_csv(OUT_ROOT / "best_by_metric.csv", index=False)
        with open(OUT_ROOT / "best_by_metric.json", "w") as f:
            json.dump(_best_by_metric_payload(best_by_metric), f, indent=2)

    # Winners
    winners: dict = {}
    if not local_df.empty:
        best = _summarize_sweep(local_df).iloc[0]
        winners["local"] = (int(best["units"]), int(best["layers"]))
    if not global_df.empty:
        best = _summarize_sweep(global_df).iloc[0]
        winners["global"] = (int(best["units"]), int(best["layers"]))
    if winners:
        with open(OUT_ROOT / "winners.json", "w") as f:
            json.dump(
                {m: {"units": u, "layers": l} for m, (u, l) in winners.items()},
                f, indent=2,
            )
        print(f"\nNLL winners used for Phase C: {winners}")

    # Phase C — final comparison
    final_df = aggregate_final(winners, seeds=final_seeds)
    if not final_df.empty:
        final_df.to_csv(OUT_ROOT / "final_comparison.csv", index=False)
        print("\nFinal comparison head:")
        print(final_df.head().to_string(index=False))


def main() -> None:
    args = _parse_args()

    # Phase 0
    info = _load_or_build_subset()
    subset: List[int] = list(info["subset"])
    print(f"Using subset ({len(subset)} series): {subset}")

    archs = _arch_grid()
    if args.archs:
        wanted = {tuple(int(x) for x in a.split("x")) for a in args.archs}
        archs = [a for a in archs if a in wanted]
        if not archs:
            raise ValueError(f"No matching archs in {args.archs}")

    if args.phases is None or "A" in args.phases:
        run_sweep("local", subset=subset, archs=archs, seeds=SWEEP_SEEDS,
                  train=not args.no_train)
    if args.phases is None or "B" in args.phases:
        run_sweep("global", subset=None, archs=archs, seeds=SWEEP_SEEDS,
                  train=not args.no_train)

    # Determine winners from disk to allow re-running Phase C in isolation.
    winners: dict = {}
    local_df = aggregate_sweep("local", subset=subset, archs=archs, seeds=SWEEP_SEEDS)
    if not local_df.empty:
        best = _summarize_sweep(local_df).iloc[0]
        winners["local"] = (int(best["units"]), int(best["layers"]))
    global_df = aggregate_sweep("global", subset=None, archs=archs, seeds=SWEEP_SEEDS)
    if not global_df.empty:
        best = _summarize_sweep(global_df).iloc[0]
        winners["global"] = (int(best["units"]), int(best["layers"]))

    if (args.phases is None or "C" in args.phases) and winners:
        run_final(
            winners, seeds=args.final_seeds,
            train=not args.no_train, evaluate=not args.no_eval,
        )

    if not args.no_aggregate:
        aggregate_and_report(
            subset, archs=archs,
            sweep_seeds=SWEEP_SEEDS, final_seeds=args.final_seeds,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases", nargs="+", choices=["A", "B", "C"], default=None,
        help="Which phases to run (default: all).",
    )
    parser.add_argument(
        "--archs", nargs="+", default=None,
        help="Restrict to archs given as UNITSxLAYERS (e.g. 50x2 100x3).",
    )
    parser.add_argument(
        "--final-seeds", type=int, nargs="+", default=list(FINAL_SEEDS),
    )
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation in Phase C subprocesses.")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate forecast csvs (ytestTr.csv, ytestPd.csv, StestPd.csv) and plot results.

- Assumes each column is a separate time series.
- Saves plots to <input-dir>/pred_plot/
- Saves metrics_per_series.csv and metrics_overall.csv in <input-dir>.

Usage:
    python evaluate_forecasts.py --input-dir . --ci 0.95
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


# Plotting defaults
import matplotlib as mpl

# Update matplotlib parameters in a single dictionary
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
        "pgf.preamble": r"\usepackage{amsmath}",
        "lines.linewidth": 1,  # Set line width to 1
    }
)

# -------------------------- Metrics Helpers -------------------------- #


def safe_mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error with epsilon to avoid div-by-zero."""
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def smape(y_true, y_pred, eps=1e-8):
    """Symmetric MAPE (0..200%)."""
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0


def r2_score(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def nrmse(y_true, y_pred, method="std"):
    """Normalized RMSE. method in {'std','range','mean'}."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if method == "std":
        denom = np.std(y_true)
    elif method == "range":
        denom = np.max(y_true) - np.min(y_true)
    elif method == "mean":
        denom = np.mean(np.abs(y_true))
    else:
        raise ValueError("method must be one of {'std','range','mean'}")
    return rmse / denom if denom > 0 else np.nan


def coverage_prob(y_true, mu, sigma, z=1.96):
    """Empirical coverage of the (mu ± z*sigma) interval."""
    lower = mu - z * sigma
    upper = mu + z * sigma
    return np.mean((y_true >= lower) & (y_true <= upper))


def mean_sharpness(sigma):
    """Average predictive std (smaller is 'sharper')."""
    return np.mean(sigma)


def computeND(y, ypred):
    return np.sum(np.abs(ypred - y)) / np.sum(np.abs(y))


def compute90QL(y, ypred, Vpred):
    ypred_90q = ypred + 1.282 * np.sqrt(Vpred)
    Iq = y > ypred_90q
    Iq_ = y <= ypred_90q
    e = y - ypred_90q
    return np.sum(2 * e * (0.9 * Iq - (1 - 0.9) * Iq_)) / np.sum(np.abs(y))


def calc_metrics(y_true, y_pred, s_pred, std_factor, test_start_idx=None):
    """Return a dict of per-series metrics evaluated on the test slice."""

    if test_start_idx is not None:
        start = int(test_start_idx)
        if start < 0:
            start = 0
        if start >= y_true.shape[0]:
            y_true = y_true[0:0]
            if y_pred is not None:
                y_pred = y_pred[0:0]
            if s_pred is not None:
                s_pred = s_pred[0:0]
        else:
            y_true = y_true[start:]
            if y_pred is not None:
                y_pred = y_pred[start:]
            if s_pred is not None:
                s_pred = s_pred[start:]

    # Clean rows where y_true is NaN; align arrays
    mask = np.isfinite(y_true)
    if y_pred is not None:
        mask &= np.isfinite(y_pred)
    if s_pred is not None:
        mask &= np.isfinite(s_pred)

    yt = y_true[mask]
    yp = y_pred[mask]
    sp = s_pred[mask]

    out = {}
    if yt.size == 0:
        # Return NaNs for all metrics
        return {
            "count": 0,
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAPE_%": np.nan,
            "sMAPE_%": np.nan,
            "R2": np.nan,
            "NRMSE_std": np.nan,
            "NRMSE_range": np.nan,
            "NRMSE_mean": np.nan,
            "Bias": np.nan,
            "Coverage": np.nan,
            "Avg_Std": np.nan,
            "ND": np.nan,
            "90QL": np.nan,
        }

    out["count"] = yt.size
    err = yp - yt
    out["MAE"] = np.mean(np.abs(err))
    out["MSE"] = np.mean(err**2)
    out["RMSE"] = np.sqrt(out["MSE"])
    out["MAPE_%"] = safe_mape(yt, yp)
    out["sMAPE_%"] = smape(yt, yp)
    out["R2"] = r2_score(yt, yp)
    out["NRMSE_std"] = nrmse(yt, yp, "std")
    out["NRMSE_range"] = nrmse(yt, yp, "range")
    out["NRMSE_mean"] = nrmse(yt, yp, "mean")
    out["Bias"] = np.mean(err)
    z = std_factor  # e.g., 1.96 for 95% if normal
    out["Coverage"] = coverage_prob(yt, yp, sp, z=z) if sp is not None else np.nan
    out["Avg_Std"] = mean_sharpness(sp) if sp is not None else np.nan
    out["ND"] = computeND(yt, yp) if yp is not None else np.nan
    out["90QL"] = compute90QL(yt, yp, sp**2) if sp is not None else np.nan
    return out


# -------------------------- Plotting -------------------------- #
def plot_embedding(mu_embedding, n_series, in_dir, out_path, labels=None):
    pca = PCA(n_components=2)
    mu_emb_2d = pca.fit_transform(mu_embedding)

    # plot embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(mu_emb_2d[:, 0], mu_emb_2d[:, 1], c="blue", alpha=0.7)
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(mu_emb_2d[i, 0], mu_emb_2d[i, 1], label)
    else:
        for i in range(n_series):
            plt.text(mu_emb_2d[i, 0], mu_emb_2d[i, 1], str(i))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    emb_plot_path = in_dir / out_path
    plt.savefig(emb_plot_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_series(
    ts_idx, y_true_col, y_pred_col, s_pred_col, out_dir, val_test_indices, std_factor=1
):
    """Plot truth, prediction, and std_factor band for a single series."""
    yt = y_true_col
    yp = y_pred_col if y_pred_col is not None else None
    sp = s_pred_col if s_pred_col is not None else None
    x = np.arange(len(yt))

    plt.figure(figsize=(10, 3))
    plt.plot(x, yt, label=r"$y_{true}$", color="red")
    if yp is not None:
        plt.plot(x, yp, label=r"$\mathbb{E}[Y']$", color="blue")
    if sp is not None and yp is not None:
        lower = yp - std_factor * sp
        upper = yp + std_factor * sp
        plt.fill_between(
            x,
            lower,
            upper,
            color="blue",
            alpha=0.3,
            label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(std_factor),
        )
    # Shade validation region
    plt.axvspan(
        val_test_indices[ts_idx, 0],
        val_test_indices[ts_idx, 1],
        color="green",
        alpha=0.15,
        label="Validation",
        linewidth=0,
    )
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,
        frameon=False,
    )
    out_path = out_dir / f"series_{ts_idx:03d}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# -------------------------- Main -------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasts and plot results.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Folder containing ytestTr.csv, ytestPd.csv, StestPd.csv",
    )
    parser.add_argument(
        "--std_factor",
        type=float,
        default=1,
        help="Confidence level for bands (default: 1).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="If set, generate plots (enabled by default).",
    )
    parser.add_argument(
        "--no-plots", dest="plots", action="store_false", help="Disable plotting."
    )
    parser.set_defaults(plots=True)

    args = parser.parse_args()
    in_dir = Path(args.input_dir).resolve()
    plot_dir = in_dir / "pred_plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    def _read_csv(name):
        path = in_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path, header=None)
        return df

    y_true_df = _read_csv("yTr.csv")
    y_pred_df = _read_csv("yPd.csv")
    s_pred_df = _read_csv("SPd.csv")
    val_test_indices = _read_csv("split_indices.csv").to_numpy(dtype=int)

    # Basic shape checks (allow extra/truncated rows; we’ll align per-column)
    if y_true_df.shape[1] != y_pred_df.shape[1]:
        raise ValueError(
            f"Column mismatch: ytestTr has {y_true_df.shape[1]} cols, "
            f"ytestPd has {y_pred_df.shape[1]} cols."
        )
    if s_pred_df.shape[1] != y_pred_df.shape[1]:
        raise ValueError(
            f"Column mismatch: StestPd has {s_pred_df.shape[1]} cols, "
            f"ytestPd has {y_pred_df.shape[1]} cols."
        )

    n_series = y_true_df.shape[1]

    # Convert to numpy
    y_true = y_true_df.to_numpy(dtype=float, copy=True)
    y_pred = y_pred_df.to_numpy(dtype=float, copy=True)
    s_pred = s_pred_df.to_numpy(dtype=float, copy=True)

    # Metrics per series
    per_series_records = []
    for j in range(n_series):
        # Align length by cutting to min rows across the three for that column
        col_len = min(
            y_true[:, j].shape[0],
            y_pred[:, j].shape[0],
            s_pred[:, j].shape[0],
        )
        yt_col = y_true[:col_len, j]
        yp_col = y_pred[:col_len, j]
        sp_col = s_pred[:col_len, j]

        test_start_idx = None
        if val_test_indices.ndim >= 2 and val_test_indices.shape[0] > j:
            if val_test_indices.shape[1] >= 2:
                test_start_idx = int(val_test_indices[j, 1])
        rec = calc_metrics(
            yt_col,
            yp_col,
            sp_col,
            std_factor=args.std_factor,
            test_start_idx=test_start_idx,
        )
        rec["series"] = j
        per_series_records.append(rec)

        if args.plots:
            plot_series(
                j,
                yt_col,
                yp_col,
                sp_col,
                plot_dir,
                val_test_indices,
                std_factor=args.std_factor,
            )

    per_series_df = pd.DataFrame(per_series_records).set_index("series").sort_index()
    per_series_df.to_csv(in_dir / "metrics_per_series.csv", float_format="%.6g")

    # Overall metrics (aggregate over all valid rows of all series)
    # Concatenate aligned columns vertically
    stacks_y, stacks_p, stacks_s = [], [], []
    for j in range(n_series):
        col_len = min(
            y_true[:, j].shape[0],
            y_pred[:, j].shape[0],
            s_pred[:, j].shape[0],
        )
        yt_col = y_true[:col_len, j]
        yp_col = y_pred[:col_len, j]
        sp_col = s_pred[:col_len, j]

        test_start_idx = 0
        if val_test_indices.ndim >= 2 and val_test_indices.shape[0] > j:
            if val_test_indices.shape[1] >= 2:
                test_start_idx = int(val_test_indices[j, 1])
        if test_start_idx < 0:
            test_start_idx = 0
        if test_start_idx > col_len:
            test_start_idx = col_len

        yt_col = yt_col[test_start_idx:]
        yp_col = yp_col[test_start_idx:]
        sp_col = sp_col[test_start_idx:]

        # mask finite jointly
        mask = np.isfinite(yt_col) & np.isfinite(yp_col) & np.isfinite(sp_col)
        if mask.any():
            stacks_y.append(yt_col[mask])
            stacks_p.append(yp_col[mask])
            stacks_s.append(sp_col[mask])

    if len(stacks_y) == 0:
        overall = {
            k: np.nan
            for k in [
                "count",
                "MAE",
                "MSE",
                "RMSE",
                "MAPE_%",
                "sMAPE_%",
                "R2",
                "NRMSE_std",
                "NRMSE_range",
                "NRMSE_mean",
                "Bias",
                "Coverage",
                "Avg_Std",
                "ND",
                "90QL",
            ]
        }
        overall["count"] = 0
    else:
        Y = np.concatenate(stacks_y)
        P = np.concatenate(stacks_p)
        S = np.concatenate(stacks_s)
        overall = calc_metrics(Y, P, S, std_factor=args.std_factor)

    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(in_dir / "metrics_overall.csv", index=False, float_format="%.6g")

    # check if embeddings file exists
    if os.path.exists(in_dir / "embeddings/embeddings_start.npz"):

        data = np.load(in_dir / "embeddings/embeddings_start.npz")
        mu_embedding = data["mu_embedding"]

        plot_embedding(mu_embedding, n_series, in_dir, "embeddings_mu_pca_start.png")

    if os.path.exists(in_dir / "embeddings/embeddings_final.npz"):

        data = np.load(in_dir / "embeddings/embeddings_final.npz")
        mu_embedding = data["mu_embedding"]

        plot_embedding(mu_embedding, n_series, in_dir, "embeddings_mu_pca_final.png")

    embeddings_dir = in_dir / "embeddings"
    required_files = [
        "dam_embeddings_start.npz",
        "dam_type_embeddings_start.npz",
        "sensor_type_embeddings_start.npz",
        "direction_embeddings_start.npz",
        "sensor_embeddings_start.npz",
    ]
    if all((embeddings_dir / fname).is_file() for fname in required_files):

        # load embeddings
        dam_embeddings = np.load(embeddings_dir / "dam_embeddings_start.npz")
        dam_embeddings_mu = dam_embeddings["mu_embedding"]
        plot_embedding(
            dam_embeddings_mu,
            dam_embeddings_mu.shape[0],
            in_dir,
            "dam_embeddings_mu_pca_start.png",
            labels=["DRU", "GOU", "LGA", "LTU", "MAT", "M5"],
        )

        dam_type_embeddings = np.load(embeddings_dir / "dam_type_embeddings_start.npz")
        dam_type_embeddings_mu = dam_type_embeddings["mu_embedding"]
        plot_embedding(
            dam_type_embeddings_mu,
            dam_type_embeddings_mu.shape[0],
            in_dir,
            "dam_type_embeddings_mu_pca_start.png",
            labels=["Run-of-River", "Reservoir"],
        )

        sensor_type_embeddings = np.load(
            embeddings_dir / "sensor_type_embeddings_start.npz"
        )
        sensor_type_embeddings_mu = sensor_type_embeddings["mu_embedding"]
        plot_embedding(
            sensor_type_embeddings_mu,
            sensor_type_embeddings_mu.shape[0],
            in_dir,
            "sensor_type_embeddings_mu_pca_start.png",
            labels=["PIZ", "EXT", "PEN"],
        )

        direction_embeddings = np.load(
            embeddings_dir / "direction_embeddings_start.npz"
        )
        direction_embeddings_mu = direction_embeddings["mu_embedding"]
        plot_embedding(
            direction_embeddings_mu,
            direction_embeddings_mu.shape[0],
            in_dir,
            "direction_embeddings_mu_pca_start.png",
            labels=["NA", "X", "Y", "Z"],
        )

        sensor_embeddings = np.load(embeddings_dir / "sensor_embeddings_start.npz")
        sensor_embeddings_mu = sensor_embeddings["mu_embedding"]
        plot_embedding(
            sensor_embeddings_mu, n_series, in_dir, "sensor_embeddings_mu_pca_start.png"
        )

        # concatenate all embeddings and plot
        embedding_mu = np.vstack(
            [
                dam_embeddings_mu,
                dam_type_embeddings_mu,
                sensor_type_embeddings_mu,
                direction_embeddings_mu,
                sensor_embeddings_mu,
            ]
        )
        plot_embedding(
            embedding_mu, n_series, in_dir, "stitched_embeddings_mu_pca_start.png"
        )

    required_files = [
        "dam_embeddings_final.npz",
        "dam_type_embeddings_final.npz",
        "sensor_type_embeddings_final.npz",
        "direction_embeddings_final.npz",
        "sensor_embeddings_final.npz",
    ]
    if all((embeddings_dir / fname).is_file() for fname in required_files):

        # load embeddings
        dam_embeddings = np.load(embeddings_dir / "dam_embeddings_final.npz")
        dam_embeddings_mu = dam_embeddings["mu_embedding"]
        plot_embedding(
            dam_embeddings_mu,
            dam_embeddings_mu.shape[0],
            in_dir,
            "dam_embeddings_mu_pca_final.png",
            labels=["DRU", "GOU", "LGA", "LTU", "MAT", "M5"],
        )

        dam_type_embeddings = np.load(embeddings_dir / "dam_type_embeddings_final.npz")
        dam_type_embeddings_mu = dam_type_embeddings["mu_embedding"]
        plot_embedding(
            dam_type_embeddings_mu,
            dam_type_embeddings_mu.shape[0],
            in_dir,
            "dam_type_embeddings_mu_pca_final.png",
            labels=["Run-of-River", "Reservoir"],
        )

        sensor_type_embeddings = np.load(
            embeddings_dir / "sensor_type_embeddings_final.npz"
        )
        sensor_type_embeddings_mu = sensor_type_embeddings["mu_embedding"]
        plot_embedding(
            sensor_type_embeddings_mu,
            sensor_type_embeddings_mu.shape[0],
            in_dir,
            "sensor_type_embeddings_mu_pca_final.png",
            labels=["PIZ", "EXT", "PEN"],
        )

        direction_embeddings = np.load(
            embeddings_dir / "direction_embeddings_final.npz"
        )
        direction_embeddings_mu = direction_embeddings["mu_embedding"]
        plot_embedding(
            direction_embeddings_mu,
            direction_embeddings_mu.shape[0],
            in_dir,
            "direction_embeddings_mu_pca_final.png",
            labels=["NA", "X", "Y", "Z"],
        )

        sensor_embeddings = np.load(embeddings_dir / "sensor_embeddings_final.npz")
        sensor_embeddings_mu = sensor_embeddings["mu_embedding"]
        plot_embedding(
            sensor_embeddings_mu, n_series, in_dir, "sensor_embeddings_mu_pca_final.png"
        )

        # concatenate all embeddings and plot
        embedding_mu = np.vstack(
            [
                dam_embeddings_mu,
                dam_type_embeddings_mu,
                sensor_type_embeddings_mu,
                direction_embeddings_mu,
                sensor_embeddings_mu,
            ]
        )
        plot_embedding(
            embedding_mu, n_series, in_dir, "stitched_embeddings_mu_pca_final.png"
        )

    print(
        f"[OK] Wrote {in_dir/'metrics_per_series.csv'} and {in_dir/'metrics_overall.csv'}"
    )
    if args.plots:
        print(f"[OK] Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()

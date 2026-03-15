"""
MCB (Multiple Comparisons with the Best) Test for model comparison.

Supports two methods:
1. "hsu" - Hsu's MCB (1996): Uses raw metric values, constructs confidence intervals
2. "nemenyi" - Nemenyi/CD test (Demšar 2006): Uses ranks, M4-competition style

Behavior:
- Walks BASE_DIR looking for Seed*/trainXX/experiment01_*_model/evaluation_metrics.txt.
- Reads per-series metrics (excluding Macro/Micro averages).
- Performs MCB test comparing models across series for each (train_size, metric).
- Plots MCB diagrams showing statistical significance.

Usage:
  python -m experiments.MCB [--method hsu|nemenyi]

Outputs:
  - out/mcb_results/plots/*.svg
  - out/mcb_results/mcb_summary.csv

References:
  Hsu, J.C. (1996). Multiple Comparisons: Theory and Methods. Chapman & Hall.
  Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import t as t_dist
from scipy.stats import studentized_range

# Plotting defaults
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,
    }
)

MODEL_NAME_REGEX = re.compile(r"^experiment01_(.+?)$")

# Metrics where HIGHER is better (will be negated for "lower is better" convention)
HIGHER_IS_BETTER_METRICS = {"LogLik", "loglik", "log_lik", "log_likelihood"}


def is_seed_dir(p: Path) -> bool:
    return p.is_dir() and re.match(r"(?i)^seed\d+$", p.name) is not None


def parse_train_size(name: str) -> int:
    m = re.match(r"^train(\d+)$", name)
    return int(m.group(1)) if m else -1


def discover_runs(base_dir: Path, debug: bool = False) -> List[Tuple[int, int, Path]]:
    """Discover all experiment runs in the base directory."""
    runs = []
    for seed_dir in sorted([p for p in base_dir.iterdir() if is_seed_dir(p)]):
        if debug:
            print(f"[debug] seed dir: {seed_dir}", file=sys.stderr)

        m = re.match(r"^Seed(\d+)$", seed_dir.name, flags=re.IGNORECASE)
        seed_val = int(m.group(1)) if m else seed_dir.name

        for train_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir()]):
            ts = parse_train_size(train_dir.name)
            if ts == -1:
                continue
            for exp_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
                if MODEL_NAME_REGEX.match(exp_dir.name):
                    runs.append((seed_val, ts, exp_dir))
    return runs


def extract_model_type(exp_dir_name: str) -> str:
    m = MODEL_NAME_REGEX.match(exp_dir_name)
    return m.group(1) if m else "unknown"


def get_model_label(model: str) -> str:
    """Convert model type to display label."""
    model_lower = model.lower()
    if model_lower == "global_no-embeddings":
        return "Global"
    elif model_lower == "global_simple-embeddings":
        return r"Global + $B_{simple}$"
    elif model_lower == "locals":
        return "Locals"
    elif model_lower == "global_hierarchical-embeddings":
        return r"Global + $B_{hierarchical}$"
    return model


def _read_metrics_table(file_path: Path) -> pd.DataFrame:
    """Robust CSV reader for evaluation_metrics.txt."""
    try:
        df = pd.read_csv(file_path, sep=None, engine="python", dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")

    if df.shape[0] == 0:
        raise RuntimeError(f"Empty metrics file: {file_path}")

    # Drop columns that are entirely empty (from trailing commas)
    empty_cols = [c for c in df.columns if df[c].astype(str).str.strip().eq("").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Strip whitespace
    df.columns = [c.strip() for c in df.columns]
    if hasattr(df, "map"):
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    else:
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def read_per_series_metrics(file_path: Path) -> pd.DataFrame:
    """
    Read per-series metrics, excluding aggregate rows (Macro/Micro/Overall).
    Returns DataFrame with Series_ID as index and metric columns.
    """
    df = _read_metrics_table(file_path)

    if "Series_ID" not in df.columns:
        return pd.DataFrame()

    # Filter out aggregate rows
    series_col = df["Series_ID"].str.lower().str.replace("_", "")
    exclude_mask = (
        series_col.eq("macroaverage")
        | series_col.eq("microaverage")
        | series_col.eq("macro")
        | series_col.eq("micro")
        | series_col.eq("overall")
    )
    df = df[~exclude_mask].copy()

    # Convert metric columns to numeric
    metric_cols = ["RMSE", "LogLik", "MAE", "P50", "P90"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Series_ID"] = df["Series_ID"].astype(str)
    return df


def collect_all_per_series(base_dir: Path, debug: bool = False) -> pd.DataFrame:
    """
    Collect per-series metrics from all runs.
    Returns DataFrame with columns: seed, train_size, model_type, Series_ID, RMSE, LogLik, etc.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    records = []

    if not base_dir.exists():
        print(f"[warn] Base directory not found: {base_dir}", file=sys.stderr)
        return pd.DataFrame()

    runs = discover_runs(base_dir, debug=debug)
    if not runs:
        print(f"[warn] No runs found under {base_dir}", file=sys.stderr)

    for seed, train_size, exp_dir in runs:
        metrics_file = exp_dir / "evaluation_metrics.txt"
        if not metrics_file.exists():
            alt = list(exp_dir.glob("evaluation_metrics.*"))
            if alt:
                metrics_file = alt[0]
            else:
                continue

        try:
            df = read_per_series_metrics(metrics_file)
        except Exception as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue

        if df.empty:
            continue

        model_type = extract_model_type(exp_dir.name)
        df["seed"] = seed
        df["train_size"] = train_size
        df["model_type"] = model_type
        records.append(df)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


# =============================================================================
# Hsu's MCB Method
# =============================================================================


def perform_hsu_mcb(error_df: pd.DataFrame, alpha: float = 0.05) -> Optional[Dict]:
    """
    Performs Hsu's MCB (Multiple Comparisons with the Best) test.

    Hsu's MCB constructs simultaneous confidence intervals for:
        θ_i = μ_i - min_{j≠i} μ_j

    A model could be the best if its interval includes 0.

    Args:
        error_df: DataFrame where Index = Blocks (e.g., Series)
                  and Columns = Models. Values should follow "lower is better".
        alpha: Significance level (default 0.05).

    Returns:
        Dict with means, intervals, best_model, indistinguishable_group
    """
    n_blocks = len(error_df)
    k_models = len(error_df.columns)

    if n_blocks < 2 or k_models < 2:
        return None

    # 1. Calculate sample means for each model
    means = error_df.mean()

    # 2. Calculate pooled variance (MSE within blocks)
    grand_mean = error_df.values.mean()
    block_means = error_df.mean(axis=1).values
    model_means = means.values

    # Calculate residuals
    residuals = error_df.values - block_means[:, np.newaxis] - model_means + grand_mean
    ss_error = np.sum(residuals**2)
    df_error = (n_blocks - 1) * (k_models - 1)
    mse = ss_error / df_error

    # Standard error for difference between two means
    se_diff = np.sqrt(2 * mse / n_blocks)

    # 3. Critical value (Bonferroni-adjusted t for Dunnett-style comparison)
    d_crit = t_dist.ppf(1 - alpha / (2 * (k_models - 1)), df_error)

    # 4. For each model, compute MCB interval
    intervals = {}
    point_estimates = {}

    for model in error_df.columns:
        other_means = [means[m] for m in error_df.columns if m != model]
        min_other = min(other_means)

        point_est = means[model] - min_other
        margin = d_crit * se_diff

        intervals[model] = (point_est - margin, point_est + margin)
        point_estimates[model] = point_est

    # 5. Identify the best model and indistinguishable group
    best_model = means.idxmin()
    best_mean = means.min()

    # A model is in the best group if its interval contains 0 or goes negative
    in_group = [m for m, (lo, hi) in intervals.items() if lo <= 0]

    return {
        "method": "hsu",
        "means": means,
        "point_estimates": pd.Series(point_estimates),
        "intervals": intervals,
        "best_model": best_model,
        "best_mean": best_mean,
        "indistinguishable_group": in_group,
        "n_blocks": n_blocks,
        "k_models": k_models,
        "se_diff": se_diff,
        "d_crit": d_crit,
        "mse": mse,
        "df_error": df_error,
    }


def plot_hsu_mcb(
    results: Dict,
    title: str,
    out_path: Path,
    alpha: float = 0.05,
) -> None:
    """Plot Hsu's MCB test results with confidence intervals."""
    from matplotlib.lines import Line2D

    means = results["means"]
    intervals = results["intervals"]
    point_estimates = results["point_estimates"]
    in_group = results["indistinguishable_group"]

    # Sort models by mean (best first)
    sorted_models = means.sort_values().index.tolist()
    labels = [get_model_label(m) for m in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_positions = range(len(sorted_models))

    for y, model in enumerate(sorted_models):
        lo, hi = intervals[model]
        point = point_estimates[model]

        color = "green" if model in in_group else "gray"
        marker = "o" if model in in_group else "s"

        # Plot confidence interval
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=2)
        ax.plot([lo, lo], [y - 0.1, y + 0.1], color=color, linewidth=2)
        ax.plot([hi, hi], [y - 0.1, y + 0.1], color=color, linewidth=2)

        # Plot point estimate
        ax.scatter(
            [point],
            [y],
            color=color,
            marker=marker,
            s=80,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add mean value annotation
        ax.annotate(
            f"μ={means[model]:.3f}",
            xy=(hi + 0.02, y),
            fontsize=8,
            va="center",
            ha="left",
            color="dimgray",
        )

    # Vertical line at 0
    ax.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="θ=0 (could be best)",
        zorder=1,
    )

    # Shade the "could be best" region
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], 0, color="green", alpha=0.05, zorder=0)
    ax.set_xlim(xlim)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\theta_i = \mu_i - \min_{j \neq i} \mu_j$ (Lower is Better)")
    ax.set_title(title)

    legend_elements = [
        Line2D(
            [0], [0], color="green", linewidth=2, label="In best group (CI includes 0)"
        ),
        Line2D([0], [0], color="gray", linewidth=2, label="Significantly worse"),
        Line2D(
            [0], [0], color="red", linestyle="--", linewidth=1.5, label="θ=0 threshold"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Nemenyi / Critical Difference Method (M4-style)
# =============================================================================


def perform_nemenyi_cd(error_df: pd.DataFrame, alpha: float = 0.05) -> Optional[Dict]:
    """
    Performs Nemenyi-style Critical Difference test using ranks.

    This is the method used in M4 competition and recommended by Demšar (2006).

    Args:
        error_df: DataFrame where Index = Blocks (e.g., Series)
                  and Columns = Models. Values should follow "lower is better".
        alpha: Significance level (default 0.05).

    Returns:
        Dict with mean_ranks, critical_distance, best_model, indistinguishable_group
    """
    n_blocks = len(error_df)
    k_models = len(error_df.columns)

    if n_blocks < 2 or k_models < 2:
        return None

    # 1. Calculate Ranks (Row-wise, 1 = best/lowest error)
    ranks = error_df.rank(axis=1)

    # 2. Calculate Mean Rank for each model
    mean_ranks = ranks.mean()

    # 3. Calculate Critical Distance using studentized range
    q_val = studentized_range.ppf(1 - alpha, k_models, np.inf)
    critical_distance = q_val * np.sqrt((k_models * (k_models + 1)) / (6 * n_blocks))

    # 4. Identify the Best Model (Lowest Mean Rank)
    best_model = mean_ranks.idxmin()
    best_rank = mean_ranks.min()

    # Identify indistinguishable models (within CD of the best)
    threshold = best_rank + critical_distance
    in_group = mean_ranks[mean_ranks <= threshold].index.tolist()

    return {
        "method": "nemenyi",
        "mean_ranks": mean_ranks,
        "critical_distance": critical_distance,
        "best_model": best_model,
        "best_rank": best_rank,
        "indistinguishable_group": in_group,
        "n_blocks": n_blocks,
        "k_models": k_models,
    }


def plot_nemenyi_cd(
    results: Dict,
    title: str,
    out_path: Path,
    alpha: float = 0.05,
) -> None:
    """Plot Nemenyi CD test results."""
    mean_ranks = results["mean_ranks"]
    critical_distance = results["critical_distance"]
    best_rank = results["best_rank"]

    # Sort models by rank
    sorted_ranks = mean_ranks.sort_values()
    labels = [get_model_label(m) for m in sorted_ranks.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_positions = range(len(sorted_ranks))

    # Plot points
    ax.scatter(sorted_ranks.values, y_positions, color="black", zorder=3, s=50)

    # Plot CD intervals (horizontal error bars)
    ax.errorbar(
        sorted_ranks.values,
        y_positions,
        xerr=critical_distance / 2,
        fmt="none",
        capsize=5,
        color="gray",
        capthick=1.5,
    )

    # Highlight the significance threshold
    ax.axvline(
        x=best_rank + critical_distance,
        color="r",
        linestyle="--",
        label=f"Significance Threshold (α={alpha})",
    )
    ax.axvline(x=best_rank, color="g", linestyle="--", alpha=0.5, label="Best Rank")

    # Fill the "Winning Zone"
    ax.axvspan(best_rank, best_rank + critical_distance, color="green", alpha=0.1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Rank (Lower is Better)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =============================================================================
# Main Analysis
# =============================================================================


def run_mcb_analysis(
    base_dir: Path = Path("./out"),
    out_dir: Path = Path("./out/mcb_results"),
    alpha: float = 0.05,
    metrics: List[str] = None,
    method: Literal["hsu", "nemenyi"] = "nemenyi",
) -> None:
    """
    Run MCB analysis across all train sizes and metrics.

    Args:
        base_dir: Directory containing experiment results.
        out_dir: Output directory for plots and summary.
        alpha: Significance level.
        metrics: List of metrics to analyze.
        method: "hsu" for Hsu's MCB, "nemenyi" for rank-based CD test.
    """
    if metrics is None:
        metrics = ["RMSE", "MAE"]

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using method: {method.upper()}")
    print("Collecting per-series metrics...")
    df = collect_all_per_series(base_dir)

    if df.empty:
        print("[warn] No data collected.")
        return

    print(f"Collected {len(df)} records from {df['model_type'].nunique()} models")

    summary_records = []
    train_sizes = sorted(df["train_size"].unique())

    for train_size in train_sizes:
        df_train = df[df["train_size"] == train_size]

        for metric in metrics:
            if metric not in df_train.columns:
                continue

            # Aggregate across seeds: average metric per (model, series)
            pivot = (
                df_train.groupby(["Series_ID", "model_type"])[metric].mean().unstack()
            )

            if pivot.empty or pivot.shape[1] < 2:
                continue

            # Drop series with any NaN
            pivot = pivot.dropna()

            if len(pivot) < 5:
                print(
                    f"[warn] Skipping train{train_size}/{metric}: only {len(pivot)} complete series"
                )
                continue

            # Handle "higher is better" metrics by negating
            display_metric = metric
            if metric in HIGHER_IS_BETTER_METRICS:
                pivot = -pivot  # Negate so lower is better
                display_metric = f"-{metric}"
                print(
                    f"  [info] Negating {metric} (higher is better → lower is better)"
                )

            # Perform test based on method
            if method == "hsu":
                results = perform_hsu_mcb(pivot, alpha=alpha)
                if results is None:
                    continue

                se = results["se_diff"]
                d_crit = results["d_crit"]
                title = (
                    f"Hsu's MCB: {display_metric} - Train Size {train_size}%\n"
                    f"(n={results['n_blocks']} series, SE={se:.4f}, d={d_crit:.3f})"
                )
                out_path = plots_dir / f"MCB_hsu_{metric}_train{train_size}.svg"
                plot_hsu_mcb(results, title, out_path, alpha=alpha)

                # Record summary
                for model in results["means"].index:
                    lo, hi = results["intervals"][model]
                    summary_records.append(
                        {
                            "train_size": train_size,
                            "metric": metric,
                            "method": "hsu",
                            "model": model,
                            "model_label": get_model_label(model),
                            "mean": results["means"][model],
                            "theta": results["point_estimates"][model],
                            "ci_lower": lo,
                            "ci_upper": hi,
                            "is_best": model == results["best_model"],
                            "in_best_group": model
                            in results["indistinguishable_group"],
                            "n_series": results["n_blocks"],
                        }
                    )

            else:  # nemenyi
                results = perform_nemenyi_cd(pivot, alpha=alpha)
                if results is None:
                    continue

                cd = results["critical_distance"]
                title = (
                    f"Nemenyi CD: {display_metric} - Train Size {train_size}%\n"
                    f"(n={results['n_blocks']} series, CD={cd:.3f})"
                )
                out_path = plots_dir / f"MCB_nemenyi_{metric}_train{train_size}.svg"
                plot_nemenyi_cd(results, title, out_path, alpha=alpha)

                # Record summary
                for model, rank in results["mean_ranks"].items():
                    summary_records.append(
                        {
                            "train_size": train_size,
                            "metric": metric,
                            "method": "nemenyi",
                            "model": model,
                            "model_label": get_model_label(model),
                            "mean_rank": rank,
                            "critical_distance": cd,
                            "is_best": model == results["best_model"],
                            "in_best_group": model
                            in results["indistinguishable_group"],
                            "n_series": results["n_blocks"],
                        }
                    )

            print(
                f"train{train_size}/{metric}: Best={get_model_label(results['best_model'])}, "
                f"Group={[get_model_label(m) for m in results['indistinguishable_group']]}"
            )

    # Save summary
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(out_dir / "mcb_summary.csv", index=False)
        print(f"\nResults saved to {out_dir}")
    else:
        print("[warn] No MCB results generated.")


def main():
    parser = argparse.ArgumentParser(
        description="MCB (Multiple Comparisons with the Best) Test"
    )
    parser.add_argument(
        "--method",
        choices=["hsu", "nemenyi"],
        default="nemenyi",
        help="Method: 'hsu' for Hsu's MCB (raw values), 'nemenyi' for rank-based CD (M4-style). Default: nemenyi",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./out"),
        help="Base directory with experiment results (default: ./out)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./out/mcb_results"),
        help="Output directory (default: ./out/mcb_results)",
    )
    args = parser.parse_args()

    run_mcb_analysis(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        alpha=args.alpha,
        metrics=["RMSE", "MAE", "LogLik", "P50", "P90"],
        method=args.method,
    )


if __name__ == "__main__":
    main()

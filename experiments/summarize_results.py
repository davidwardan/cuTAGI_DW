"""
Summarize evaluation metrics across seeds / train sizes / model types
and plot mean ± std across train sizes for each metric.

Behavior:
- Walks BASE_DIR looking for Seed*/trainXX/experiment01_*_model/evaluation_metrics.txt.
- Silently skips anything missing; collects whatever exists.
- Aggregates mean/std by (model_type, train_size). If only one sample -> std=0.
- Plots one figure per metric with mean ± std across train sizes (sparse OK).

Usage:
  python summarize_metrics.py --base /path/to/base --out ./metrics_out
  # --base defaults to current directory

Outputs:
  - collected_metrics_raw.csv
  - metrics_summary_by_model_and_train_size.csv
  - metrics_summary_by_model_and_train_size.md
  - plots/*.png
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def is_seed_dir(p: Path) -> bool:
    return p.is_dir() and re.match(r"(?i)^seed\d+$", p.name) is not None


def parse_train_size(name: str) -> int:
    m = re.match(r"^train(\d+)$", name)
    return int(m.group(1)) if m else -1


def discover_runs(base_dir: Path, debug: bool = False) -> List[Tuple[int, int, Path]]:
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


# ------- Robust metrics reading (handles trailing commas) --------
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
    else:  # pandas < 2.2 fallback
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def read_metrics_by_avg_type(file_path: Path) -> dict:
    """
    Read metrics for both macroaverage and microaverage rows.
    Returns a dict: {"macro": pd.Series, "micro": pd.Series}.
    Falls back to 'overall' or last row if new format not found.
    """
    df = _read_metrics_table(file_path)
    keep = [c for c in ["RMSE", "LogLik", "MAE", "P50", "P90"] if c in df.columns]

    def extract_row(row) -> pd.Series:
        out = {k: pd.to_numeric(row.get(k, np.nan), errors="coerce") for k in keep}
        return pd.Series(out)

    result = {}

    if "Series_ID" in df.columns:
        series_col = df["Series_ID"].str.lower().str.replace("_", "")

        # Try macroaverage (handles "Macro_Average", "macroaverage", "macro")
        macro_mask = series_col.eq("macroaverage") | series_col.eq("macro")
        if macro_mask.any():
            result["macro"] = extract_row(df.loc[macro_mask].iloc[-1])

        # Try microaverage (handles "Micro_Average", "microaverage", "micro")
        micro_mask = series_col.eq("microaverage") | series_col.eq("micro")
        if micro_mask.any():
            result["micro"] = extract_row(df.loc[micro_mask].iloc[-1])

        # Fallback to 'overall' if neither found
        if not result:
            overall_mask = series_col.eq("overall")
            if overall_mask.any():
                row = extract_row(df.loc[overall_mask].iloc[-1])
                result["macro"] = row
                result["micro"] = row

    # Final fallback: use last row for both
    if not result:
        row = extract_row(df.iloc[-1])
        result["macro"] = row
        result["micro"] = row

    return result


# -----------------------------------------------------------------


def collect_all(base_dir, debug: bool = False) -> pd.DataFrame:
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
            metrics_by_type = read_metrics_by_avg_type(metrics_file)
        except Exception as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue

        model_type = extract_model_type(exp_dir.name)

        # Create one record per avg_type (macro/micro)
        for avg_type, metrics in metrics_by_type.items():
            rec = {
                "seed": seed,
                "train_size": train_size,
                "model_type": model_type,
                "avg_type": avg_type,
            }

            for k, v in metrics.items():
                if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v):
                    rec[k] = float(v)

            if len(rec) > 4:
                records.append(rec)

    return pd.DataFrame.from_records(records) if records else pd.DataFrame()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["model_type", "train_size", "avg_type"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]
    if not metric_cols:
        return pd.DataFrame()

    agg = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten columns
    agg.columns = [
        (
            f"{a}_{b}"
            if b
            else a if a not in ["model_type", "train_size", "avg_type"] else a
        )
        for a, b in agg.columns
    ]

    # std NaN -> 0 when only one run
    for m in metric_cols:
        std_col, cnt_col = f"{m}_std", f"{m}_count"
        if std_col in agg.columns and cnt_col in agg.columns:
            mask = (agg[cnt_col] == 1) & (agg[std_col].isna())
            agg.loc[mask, std_col] = 0.0

    return agg.sort_values(by=["model_type", "train_size", "avg_type"]).reset_index(
        drop=True
    )


def plot_metrics(agg: pd.DataFrame, out_dir: Path) -> None:
    if agg.empty:
        print("[warn] Nothing to plot.")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_bases = sorted(
        set(
            c.replace("_mean", "").replace("_std", "")
            for c in agg.columns
            if c.endswith("_mean") or c.endswith("_std")
        )
    )

    # Get unique avg_types (macro, micro)
    avg_types = (
        list(agg["avg_type"].dropna().unique()) if "avg_type" in agg.columns else [""]
    )

    # Precompute shared y-limits for each metric across all avg_types
    ylim_per_metric = {}
    for metric in metric_bases:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in agg.columns or std_col not in agg.columns:
            continue

        all_means = agg[mean_col].dropna()
        all_stds = agg[std_col].fillna(0)
        if len(all_means) == 0:
            continue

        # Compute global min/max considering error bars
        y_min = (agg[mean_col] - all_stds).min()
        y_max = (agg[mean_col] + all_stds).max()
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        ylim_per_metric[metric] = (y_min - margin, y_max + margin)

    for avg_type in avg_types:
        # Filter by avg_type
        if avg_type:
            agg_filtered = agg[agg["avg_type"] == avg_type]
            suffix = f"_{avg_type}"
            title_suffix = f" ({avg_type.capitalize()})"
        else:
            agg_filtered = agg
            suffix = ""
            title_suffix = ""

        for metric in metric_bases:
            mean_col, std_col = f"{metric}_mean", f"{metric}_std"
            if (
                mean_col not in agg_filtered.columns
                or std_col not in agg_filtered.columns
            ):
                continue

            fig = plt.figure(figsize=(6, 3))
            ax = fig.gca()
            for model in agg_filtered["model_type"].dropna().unique():
                # custom model name
                if model.lower() == "global_no-embeddings":
                    label = "Global"
                elif model.lower() == "global_simple-embeddings":
                    label = r"Global + $B_{simple}$"
                elif model.lower() == "locals":
                    label = "Locals"
                elif model.lower() == "global_hierarchical-embeddings":
                    label = r"Global + $B_{hierarchical}$"
                else:
                    label = model

                sub = agg_filtered[agg_filtered["model_type"] == model].sort_values(
                    "train_size"
                )
                ax.errorbar(
                    sub["train_size"],
                    sub[mean_col],
                    yerr=sub[std_col].fillna(0),
                    fmt="-o",
                    label=label,
                )

            plt.xlabel("Train size (%)")
            if metric == "LogLik":
                plt.ylabel(r"Log-Likelihood $(\uparrow)$")
            else:
                plt.ylabel(f"{metric} $(\\downarrow)$")
            plt.title(f"{metric}{title_suffix}")

            # Apply shared y-limits
            if metric in ylim_per_metric:
                ax.set_ylim(ylim_per_metric[metric])

            legend_order = [
                "Locals",
                "Global",
                r"Global + $B_{simple}$",
                r"Global + $B_{hierarchical}$",
            ]
            handles, labels = ax.get_legend_handles_labels()
            order = sorted(
                range(len(labels)),
                key=lambda i: (
                    legend_order.index(labels[i])
                    if labels[i] in legend_order
                    else len(legend_order) + i
                ),
            )
            ax.legend(
                [handles[i] for i in order],
                [labels[i] for i in order],
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=1,
                frameon=False,
            )
            # show all x ticks
            if len(sub) > 0:
                ax.set_xticks(sub["train_size"].unique())
            fig.subplots_adjust(right=0.72)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}{suffix}.svg")
            plt.close(fig)


def main():

    base_dir = Path("./out").expanduser()
    out_dir = Path("./out/summarized_metrics").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_all(base_dir)
    if df.empty:
        (out_dir / "collected_metrics_raw.csv").write_text("")
        (out_dir / "metrics_summary_by_model_and_train_size.csv").write_text("")
        (out_dir / "metrics_summary_by_model_and_train_size.md").write_text(
            "# Metrics Summary (mean ± std)\n\n_No data collected yet._\n"
        )
        print(f"[warn] No metrics collected under {base_dir}.")
        return

    df.sort_values(by=["seed", "model_type", "train_size"]).to_csv(
        out_dir / "collected_metrics_raw.csv", index=False
    )

    agg = aggregate(df)
    if agg.empty:
        (out_dir / "metrics_summary_by_model_and_train_size.csv").write_text("")
        (out_dir / "metrics_summary_by_model_and_train_size.md").write_text(
            "# Metrics Summary (mean ± std)\n\n_No aggregatable metrics found._\n"
        )
        print(f"[warn] No aggregatable metrics found.")
        return

    agg.to_csv(out_dir / "metrics_summary_by_model_and_train_size.csv", index=False)
    plot_metrics(agg, out_dir)

    print(f"Done. Outputs written to: {out_dir}")
    print(f"- Raw collection: {out_dir / 'collected_metrics_raw.csv'}")
    print(f"- Summary CSV:   {out_dir / 'metrics_summary_by_model_and_train_size.csv'}")
    print(f"- Plots dir:     {out_dir / 'plots'}")


if __name__ == "__main__":
    main()

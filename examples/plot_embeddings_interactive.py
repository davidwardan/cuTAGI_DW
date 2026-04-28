"""Interactive explorer for Gaussian traffic sensor embeddings.

Produces a standalone HTML file with a small navigation sidebar and a minimal
Plotly view. Select a sensor to compare its nearest neighbors under several
mean- and variance-aware distances for diagonal Gaussian embeddings.

Inputs
------
saved_results/hq_embeddings_mean.csv      (N, embed_dim) embedding means
saved_results/hq_embeddings_var.csv       (N, embed_dim) embedding variances
data/hq/ts_weekly_values_final.csv        per-sensor time series
data/hq/ts_weekly_datetimes_final.csv     per-sensor dates

Output
------
experiments/out/hq_embedding_explorer.html
"""

import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import fire
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans


EMB_CSV = "saved_results/hq_embeddings_mean.csv"
VAR_CSV = "saved_results/hq_embeddings_var.csv"
TRAFFIC_CSV = "data/hq/ts_weekly_values_final.csv"
DATES_CSV = "data/hq/ts_weekly_datetimes_final.csv"
OUT_HTML = "experiments/out/hq_embedding_explorer.html"

EPS = 1e-8
DEFAULT_METRIC = "wasserstein"
DEFAULT_PROJECTION = "weighted_pca"


METRIC_SPECS = {
    "wasserstein": {
        "label": "Wasserstein-2",
        "description": "Mean distance plus standard-deviation distance.",
        "uses_variance": True,
    },
    "symmetric_kl": {
        "label": "Symmetric KL",
        "description": "Two-way Gaussian divergence for mean and variance mismatch.",
        "uses_variance": True,
    },
    "bhattacharyya": {
        "label": "Bhattacharyya",
        "description": "Distribution overlap distance. Lower means more overlap.",
        "uses_variance": True,
    },
    "variance_scaled": {
        "label": "Variance-scaled mean",
        "description": "Mean separation scaled by the pair's pooled variance.",
        "uses_variance": True,
    },
    "mean_euclidean": {
        "label": "Mean Euclidean",
        "description": "Euclidean distance between embedding means.",
        "uses_variance": False,
    },
    "mean_cosine": {
        "label": "Mean cosine",
        "description": "Cosine distance between embedding means.",
        "uses_variance": False,
    },
}


TRACE_COLORS = [
    "#111827",
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#d97706",
    "#0891b2",
    "#be185d",
    "#4b5563",
    "#65a30d",
    "#7c3aed",
]


def pca_3d(x: np.ndarray) -> np.ndarray:
    """Return a 3D PCA projection, padding with zeros for low-rank inputs."""
    x = np.asarray(x, dtype=np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)

    n_components = min(3, vt.shape[0])
    coords = np.zeros((x.shape[0], 3), dtype=np.float64)
    if n_components:
        coords[:, :n_components] = x_centered @ vt[:n_components].T
    return coords


def pca_3d_weighted(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Precision-weighted PCA.

    Confident embeddings drive the principal axes; uncertain embeddings are
    still projected onto those axes.
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        return pca_3d(x)

    weights = weights / weights.sum()
    mu = (weights[:, None] * x).sum(axis=0)
    x_centered = x - mu
    x_scaled = x_centered * np.sqrt(weights[:, None] * len(x))
    _, _, vt = np.linalg.svd(x_scaled, full_matrices=False)

    n_components = min(3, vt.shape[0])
    coords = np.zeros((x.shape[0], 3), dtype=np.float64)
    if n_components:
        coords[:, :n_components] = x_centered @ vt[:n_components].T
    return coords


def resolve_existing_path(path: str, fallbacks: Iterable[str], label: str) -> str:
    """Return the first existing path from path + fallbacks, or exit."""
    candidates = [path, *fallbacks]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            if candidate != path:
                print(f"Using {label} fallback: {candidate}")
            return candidate

    print(f"{label} file not found. Tried: {', '.join(candidates)}", file=sys.stderr)
    sys.exit(1)


def load_csv_matrix(path: str, label: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        print(f"{label} must be a 2D CSV matrix: {path}", file=sys.stderr)
        sys.exit(1)
    return arr


def replace_nonfinite_by_column_median(x: np.ndarray, label: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.all():
        return x

    clean = x.copy()
    medians = np.nanmedian(np.where(finite, clean, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    bad_rows, bad_cols = np.where(~finite)
    clean[bad_rows, bad_cols] = medians[bad_cols]
    print(f"Warning: replaced {len(bad_rows)} non-finite values in {label}.")
    return clean


def sanitize_variance(var: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    var = np.asarray(var, dtype=np.float64)
    if var.ndim == 1:
        var = var.reshape(1, -1)

    if var.shape[0] < shape[0]:
        print(
            f"Warning: variance rows ({var.shape[0]}) < embeddings rows "
            f"({shape[0]}); truncating embeddings to match."
        )
    if var.shape[1] != shape[1]:
        print(
            f"Warning: variance dim ({var.shape[1]}) != embedding dim "
            f"({shape[1]}); truncating to the shared dimension."
        )

    var = replace_nonfinite_by_column_median(var, "embedding variances")
    positive = var[np.isfinite(var) & (var > 0)]
    floor = max(float(np.percentile(positive, 1)) * 0.01, EPS) if positive.size else EPS
    var = np.where(np.isfinite(var) & (var > floor), var, floor)
    return var


def parse_sensor_meta(name: str) -> Tuple[str, str, str]:
    """Extract (site, instrument, axis) from a sensor name."""
    sensor = str(name)
    site = sensor[:3] if len(sensor) >= 3 else "?"

    axis = "none"
    for ax in ("x", "y", "z"):
        if f"_{ax}_" in sensor or sensor.endswith(f"_{ax}"):
            axis = ax
            break

    instrument = "other"
    for token in (
        "PIAEVA",
        "ESAEVA",
        "EFAP",
        "ESAP",
        "PEAP",
        "PHAP",
        "PIAP",
    ):
        if token in sensor:
            instrument = token
            break

    return site, instrument, axis


def categorical_codes(values: List[str]) -> Tuple[List[int], List[str]]:
    categories = sorted(set(values))
    codes = {category: idx for idx, category in enumerate(categories)}
    return [codes[value] for value in values], categories


def clean_distance_matrix(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    distance = np.maximum(distance, 0.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def pairwise_mean_cosine(mu: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mu, axis=1, keepdims=True)
    unit = np.divide(mu, norms, out=np.zeros_like(mu), where=norms > EPS)
    return clean_distance_matrix(1.0 - unit @ unit.T)


def pairwise_mean_euclidean(mu: np.ndarray) -> np.ndarray:
    n = mu.shape[0]
    distance = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = mu - mu[i]
        distance[i] = np.sqrt(np.sum(diff * diff, axis=1))
    return clean_distance_matrix(distance)


def pairwise_wasserstein_diag(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    n = mu.shape[0]
    std = np.sqrt(var)
    distance = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        mean_diff = mu - mu[i]
        std_diff = std - std[i]
        distance[i] = np.sqrt(
            np.sum(mean_diff * mean_diff, axis=1)
            + np.sum(std_diff * std_diff, axis=1)
        )
    return clean_distance_matrix(distance)


def pairwise_symmetric_kl_diag(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    n = mu.shape[0]
    inv_var = 1.0 / np.maximum(var, EPS)
    distance = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = mu - mu[i]
        term = (
            var[i] * inv_var
            + var * inv_var[i]
            + diff * diff * (inv_var[i] + inv_var)
            - 2.0
        )
        distance[i] = 0.25 * np.sum(term, axis=1)
    return clean_distance_matrix(distance)


def pairwise_bhattacharyya_diag(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    n = mu.shape[0]
    log_var = np.log(np.maximum(var, EPS))
    distance = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = mu - mu[i]
        var_mid = 0.5 * (var[i] + var)
        term_mean = 0.125 * np.sum(diff * diff / np.maximum(var_mid, EPS), axis=1)
        term_var = 0.5 * np.sum(
            np.log(np.maximum(var_mid, EPS)) - 0.5 * (log_var[i] + log_var),
            axis=1,
        )
        distance[i] = term_mean + term_var
    return clean_distance_matrix(distance)


def pairwise_variance_scaled_mean(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    n = mu.shape[0]
    distance = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = mu - mu[i]
        pooled_var = np.maximum(var[i] + var, EPS)
        distance[i] = np.sqrt(np.sum(diff * diff / pooled_var, axis=1))
    return clean_distance_matrix(distance)


def compute_distance_matrices(mu: np.ndarray, var: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "wasserstein": pairwise_wasserstein_diag(mu, var),
        "symmetric_kl": pairwise_symmetric_kl_diag(mu, var),
        "bhattacharyya": pairwise_bhattacharyya_diag(mu, var),
        "variance_scaled": pairwise_variance_scaled_mean(mu, var),
        "mean_euclidean": pairwise_mean_euclidean(mu),
        "mean_cosine": pairwise_mean_cosine(mu),
    }


def nearest_neighbors(
    distances: Dict[str, np.ndarray], k_neighbors: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    nn_idx = {}
    nn_dist = {}
    for key, distance in distances.items():
        order = np.argsort(distance, axis=1)[:, : k_neighbors + 1]
        nn_idx[key] = order
        nn_dist[key] = np.take_along_axis(distance, order, axis=1)
    return nn_idx, nn_dist


def step_colorscale(colors: List[str]) -> List[List[object]]:
    count = max(len(colors), 1)
    out = []
    for idx, color in enumerate(colors[:count]):
        out.append([idx / count, color])
        out.append([(idx + 1) / count, color])
    return out


def repeated_palette(colors: List[str], count: int) -> List[str]:
    return [colors[idx % len(colors)] for idx in range(max(count, 1))]


def make_color_options(
    labels: np.ndarray,
    site_ids: List[int],
    site_categories: List[str],
    type_ids: List[int],
    type_categories: List[str],
    axis_ids: List[int],
    axis_categories: List[str],
    total_var: np.ndarray,
    mean_wasserstein: np.ndarray,
) -> Dict[str, dict]:
    return {
        "cluster": {
            "label": "Cluster",
            "type": "categorical",
            "values": labels.astype(int).tolist(),
            "categories": [str(i) for i in range(int(labels.max()) + 1)],
        },
        "site": {
            "label": "Site",
            "type": "categorical",
            "values": site_ids,
            "categories": site_categories,
        },
        "type": {
            "label": "Type",
            "type": "categorical",
            "values": type_ids,
            "categories": type_categories,
        },
        "axis": {
            "label": "Axis",
            "type": "categorical",
            "values": axis_ids,
            "categories": axis_categories,
        },
        "variance": {
            "label": "Total variance",
            "type": "continuous",
            "values": total_var.astype(float).round(8).tolist(),
            "colorscale": "Inferno",
        },
        "local_distance": {
            "label": "Avg W2 neighbor distance",
            "type": "continuous",
            "values": mean_wasserstein.astype(float).round(8).tolist(),
            "colorscale": "Viridis",
        },
    }


def load_time_series(
    traffic_csv: str,
    dates_csv: str,
    n_embeddings: int,
    max_points: int,
) -> Tuple[List[str], List[List[str]], List[List[float]], List[dict], np.ndarray]:
    value_df = pd.read_csv(traffic_csv)
    sensor_names = [str(column) for column in value_df.columns]
    raw = value_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if raw.shape[1] != n_embeddings:
        print(
            f"Warning: {raw.shape[1]} series in {traffic_csv} vs "
            f"{n_embeddings} embeddings; truncating to min."
        )

    n = min(raw.shape[1], n_embeddings)
    raw = raw[:, :n]
    sensor_names = sensor_names[:n]

    date_df = pd.read_csv(dates_csv, dtype=str)
    date_df = date_df.reindex(columns=sensor_names)
    dates = date_df.to_numpy(dtype=object)
    if dates.shape[0] != raw.shape[0]:
        print(
            f"Warning: dates rows ({dates.shape[0]}) != values rows "
            f"({raw.shape[0]}); truncating to min."
        )

    rows = min(dates.shape[0], raw.shape[0])
    dates = dates[:rows, :n]
    raw = raw[:rows]

    with np.errstate(invalid="ignore"):
        series_mean = np.nanmean(raw, axis=0)
        series_std = np.nanstd(raw, axis=0)
    series_mean = np.where(np.isfinite(series_mean), series_mean, 0.0)
    series_std = np.where(np.isfinite(series_std) & (series_std > 1e-6), series_std, 1.0)
    standardized = (raw - series_mean) / series_std

    sensor_xs = []
    sensor_ys = []
    sensor_ranges = []
    for idx in range(n):
        date_col = dates[:, idx]
        value_col = standardized[:, idx]
        valid = pd.notna(date_col) & np.isfinite(value_col)

        if np.any(valid):
            xs = [str(value) for value in date_col[valid]][-max_points:]
        else:
            valid = np.isfinite(value_col)
            xs = [str(value) for value in np.flatnonzero(valid)][-max_points:]

        ys = value_col[valid].astype(float).round(4).tolist()[-max_points:]
        sensor_xs.append(xs)
        sensor_ys.append(ys)
        if xs and ys:
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))
            y_pad = max((y_max - y_min) * 0.08, 0.1)
            sensor_ranges.append(
                {
                    "x": [xs[0], xs[-1]] if len(xs) > 1 else [None, None],
                    "y": [round(y_min - y_pad, 4), round(y_max + y_pad, 4)],
                }
            )
        else:
            sensor_ranges.append({"x": [None, None], "y": [None, None]})

    return sensor_names, sensor_xs, sensor_ys, sensor_ranges, raw


def build_sidebar_html() -> str:
    return """
<aside class="sidebar">
  <div class="sidebar-header">
    <h1>Embedding Explorer</h1>
    <div id="datasetSummary" class="dataset-summary"></div>
  </div>

  <section class="control-section">
    <label for="sensorSearch">Sensor</label>
    <input id="sensorSearch" list="sensorOptions" autocomplete="off" spellcheck="false">
    <datalist id="sensorOptions"></datalist>
  </section>

  <section class="control-section control-grid">
    <label for="metricSelect">Metric</label>
    <select id="metricSelect"></select>

    <label for="projectionSelect">Projection</label>
    <select id="projectionSelect"></select>

    <label for="colorSelect">Color</label>
    <select id="colorSelect"></select>
  </section>

  <section class="control-section">
    <h2>Selection</h2>
    <div id="selectedCard" class="selected-card"></div>
  </section>

  <section class="control-section">
    <h2>Nearest Neighbors</h2>
    <div id="metricHint" class="metric-hint"></div>
    <table class="neighbor-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Sensor</th>
          <th id="metricColumn">Distance</th>
        </tr>
      </thead>
      <tbody id="neighborRows"></tbody>
    </table>
  </section>

  <section class="control-section">
    <h2>Manual Align</h2>
    <div id="alignRows" class="align-rows"></div>
    <button id="resetAlign" class="secondary-button" type="button">Reset shifts</button>
  </section>
</aside>
"""


def build_html_document(plot_html: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Embedding Explorer</title>
  <style>
    :root {
      color-scheme: light;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --soft: #f9fafb;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      background: #ffffff;
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    .app-shell {
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      border-right: 1px solid var(--line);
      background: #ffffff;
      padding: 18px 16px 24px;
    }
    .sidebar-header {
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 19px;
      font-weight: 650;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }
    label {
      display: block;
      margin: 0 0 6px;
      font-size: 12px;
      font-weight: 700;
      color: #374151;
    }
    input,
    select,
    button {
      width: 100%;
      border: 1px solid #d1d5db;
      background: #ffffff;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      outline: none;
    }
    input:focus,
    select:focus,
    button:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    input[type="range"] {
      border: 0;
      box-shadow: none;
      padding: 0;
      accent-color: var(--accent);
    }
    .dataset-summary,
    .metric-hint {
      color: var(--muted);
      font-size: 12px;
    }
    .control-section {
      padding: 14px 0;
      border-bottom: 1px solid var(--line);
    }
    .control-grid {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 9px 10px;
      align-items: center;
    }
    .control-grid label {
      margin: 0;
    }
    .selected-card {
      display: grid;
      gap: 8px;
      font-size: 13px;
    }
    .selected-name {
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .selected-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 12px;
      color: #374151;
    }
    .selected-meta span {
      color: var(--muted);
    }
    .neighbor-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 12px;
    }
    .neighbor-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #ffffff;
      color: var(--muted);
      font-weight: 700;
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 6px 4px;
    }
    .neighbor-table td {
      border-bottom: 1px solid #f3f4f6;
      padding: 7px 4px;
      vertical-align: top;
    }
    .neighbor-table tbody tr {
      cursor: pointer;
    }
    .neighbor-table tbody tr:hover {
      background: var(--soft);
    }
    .rank-cell {
      white-space: nowrap;
      color: #374151;
      font-variant-numeric: tabular-nums;
    }
    .rank-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: 1px;
    }
    .sensor-cell {
      max-width: 176px;
      overflow-wrap: anywhere;
    }
    .distance-cell {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: #374151;
    }
    .align-rows {
      display: grid;
      gap: 8px;
    }
    .align-row {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr) 50px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .align-rank,
    .align-value {
      font-variant-numeric: tabular-nums;
      color: #374151;
    }
    .secondary-button {
      margin-top: 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 650;
      color: #374151;
    }
    .secondary-button:hover {
      background: var(--soft);
    }
    .plot-panel {
      min-width: 0;
      padding: 10px 12px 20px;
    }
    #plotArea {
      width: 100%;
    }
    #plotArea .js-plotly-plot,
    #plotArea .plot-container {
      width: 100% !important;
    }
    @media (max-width: 980px) {
      .app-shell {
        grid-template-columns: 1fr;
      }
      .sidebar {
        position: relative;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .plot-panel {
        padding: 8px 6px 18px;
      }
    }
  </style>
</head>
<body>
  <main class="app-shell">
    __SIDEBAR_HTML__
    <section class="plot-panel">
      <div id="plotArea">__PLOT_HTML__</div>
    </section>
  </main>
</body>
</html>
"""
    return (
        template.replace("__SIDEBAR_HTML__", build_sidebar_html())
        .replace("__PLOT_HTML__", plot_html)
    )


def main(
    emb_csv: str = EMB_CSV,
    var_csv: str = VAR_CSV,
    traffic_csv: str = TRAFFIC_CSV,
    dates_csv: str = DATES_CSV,
    out_html: str = OUT_HTML,
    k_neighbors: int = 5,
    n_clusters: int = 8,
    max_points: int = 208,
    default_metric: str = DEFAULT_METRIC,
    default_projection: str = DEFAULT_PROJECTION,
):
    emb_csv = resolve_existing_path(
        emb_csv,
        (
            emb_csv.replace("_embeddings.csv", "_embeddings_mean.csv")
            if emb_csv.endswith("_embeddings.csv")
            else ""
        ),
        "Embeddings",
    )
    traffic_csv = resolve_existing_path(traffic_csv, (), "Time-series values")
    dates_csv = resolve_existing_path(
        dates_csv,
        ("data/hq/ts_weekly_datetimes.csv",),
        "Time-series dates",
    )

    out_dir = os.path.dirname(out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    emb = load_csv_matrix(emb_csv, "Embeddings")
    emb = replace_nonfinite_by_column_median(emb, "embedding means")

    if os.path.exists(var_csv):
        emb_var = load_csv_matrix(var_csv, "Embedding variances")
        emb_var = sanitize_variance(emb_var, emb.shape)
        variance_available = True
    else:
        print(f"Warning: variance file {var_csv} not found; using uniform variance.")
        emb_var = np.ones_like(emb)
        variance_available = False

    shared_rows = min(emb.shape[0], emb_var.shape[0])
    shared_dim = min(emb.shape[1], emb_var.shape[1])
    emb = emb[:shared_rows, :shared_dim]
    emb_var = emb_var[:shared_rows, :shared_dim]

    sensor_names, sensor_xs, sensor_ys, sensor_ranges, raw = load_time_series(
        traffic_csv=traffic_csv,
        dates_csv=dates_csv,
        n_embeddings=emb.shape[0],
        max_points=max_points,
    )
    n = len(sensor_names)
    if n == 0:
        print("No overlapping sensors to plot.", file=sys.stderr)
        sys.exit(1)

    emb = emb[:n]
    emb_var = emb_var[:n]
    embed_dim = emb.shape[1]
    print(f"Loaded embeddings: {n} x {embed_dim}")
    print(f"Loaded time series: {raw.shape[0]} rows x {raw.shape[1]} sensors")

    total_var = emb_var.sum(axis=1)
    mean_var = emb_var.mean(axis=1)
    precision = 1.0 / np.maximum(total_var, EPS)

    var_norm = (total_var - total_var.min()) / max(
        float(total_var.max() - total_var.min()), EPS
    )
    marker_sizes = (1.0 - var_norm) * 5.0 + 3.0

    pca = pca_3d(emb)
    pca_w = pca_3d_weighted(emb, precision)

    n_clusters = min(max(int(n_clusters), 1), n)
    if n_clusters == 1:
        labels = np.zeros(n, dtype=int)
    else:
        labels = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=0,
        ).fit_predict(emb)

    meta = [parse_sensor_meta(name) for name in sensor_names]
    site_ids, site_categories = categorical_codes([item[0] for item in meta])
    type_ids, type_categories = categorical_codes([item[1] for item in meta])
    axis_ids, axis_categories = categorical_codes([item[2] for item in meta])

    k_neighbors = min(max(int(k_neighbors), 0), max(n - 1, 0))
    distances = compute_distance_matrices(emb, emb_var)
    nn_idx, nn_dist = nearest_neighbors(distances, k_neighbors)

    if default_metric not in distances:
        print(f"Warning: unknown metric '{default_metric}', using {DEFAULT_METRIC}.")
        default_metric = DEFAULT_METRIC
    if default_projection not in ("standard_pca", "weighted_pca"):
        print(
            f"Warning: unknown projection '{default_projection}', "
            f"using {DEFAULT_PROJECTION}."
        )
        default_projection = DEFAULT_PROJECTION

    mean_neighbor_distance = {}
    for key, values in nn_dist.items():
        if k_neighbors:
            mean_neighbor_distance[key] = values[:, 1:].mean(axis=1)
        else:
            mean_neighbor_distance[key] = np.zeros(n, dtype=np.float64)

    color_options = make_color_options(
        labels=labels,
        site_ids=site_ids,
        site_categories=site_categories,
        type_ids=type_ids,
        type_categories=type_categories,
        axis_ids=axis_ids,
        axis_categories=axis_categories,
        total_var=total_var,
        mean_wasserstein=mean_neighbor_distance["wasserstein"],
    )

    n_ts = k_neighbors + 1
    subplot_titles = ["Embedding projection", "Selected sensor"]
    subplot_titles.extend([""] * max(n_ts - 1, 0))
    fig = make_subplots(
        rows=1 + n_ts,
        cols=1,
        specs=[[{"type": "scene"}]] + [[{"type": "xy"}]] * n_ts,
        row_heights=[0.54] + [0.46 / n_ts] * n_ts,
        vertical_spacing=0.018,
        subplot_titles=tuple(subplot_titles),
    )

    initial_projection = pca_w if default_projection == "weighted_pca" else pca
    qualitative = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#9333ea",
        "#d97706",
        "#0891b2",
        "#be185d",
        "#4b5563",
        "#65a30d",
        "#7c3aed",
        "#0f766e",
        "#b45309",
        "#4338ca",
    ]

    hover_text = [
        (
            f"{sensor_names[idx]}<br>"
            f"cluster {labels[idx]} | site {meta[idx][0]} | "
            f"type {meta[idx][1]} | axis {meta[idx][2]}<br>"
            f"total variance {total_var[idx]:.4g}"
        )
        for idx in range(n)
    ]
    cluster_categories = [str(i) for i in range(int(labels.max()) + 1)]
    fig.add_trace(
        go.Scatter3d(
            x=initial_projection[:, 0],
            y=initial_projection[:, 1],
            z=initial_projection[:, 2],
            mode="markers",
            marker=dict(
                size=marker_sizes.astype(float).round(3).tolist(),
                color=labels.astype(int).tolist(),
                colorscale=step_colorscale(
                    repeated_palette(qualitative, len(cluster_categories))
                ),
                cmin=-0.5,
                cmax=len(cluster_categories) - 0.5,
                line=dict(width=0),
                opacity=0.82,
                showscale=True,
                colorbar=dict(
                    title="Cluster",
                    tickvals=list(range(len(cluster_categories))),
                    ticktext=cluster_categories,
                    len=0.46,
                    y=0.79,
                    thickness=13,
                ),
            ),
            text=hover_text,
            hoverinfo="text",
            customdata=np.arange(n),
            name="Sensors",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker=dict(size=[], color=[], opacity=0.68, line=dict(width=0)),
            hoverinfo="skip",
            name="Selection",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for idx in range(n_ts):
        fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="lines",
                line=dict(
                    width=2.0 if idx == 0 else 1.3,
                    color=TRACE_COLORS[idx % len(TRACE_COLORS)],
                ),
                opacity=1.0,
                name="anchor" if idx == 0 else f"neighbor {idx}",
                hovertemplate="%{fullData.name}<br>%{x}<br>z=%{y:.3f}<extra></extra>",
            ),
            row=2 + idx,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=max(860, 520 + 105 * n_ts),
        showlegend=False,
        dragmode="pan",
        margin=dict(l=48, r=20, t=38, b=32),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family=(
                "Inter, ui-sans-serif, system-ui, -apple-system, "
                "BlinkMacSystemFont, 'Segoe UI', sans-serif"
            ),
            size=12,
            color="#111827",
        ),
        scene=dict(
            uirevision="camera",
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="#ffffff",
            xaxis=dict(showbackground=False, gridcolor="#e5e7eb", zeroline=False),
            yaxis=dict(showbackground=False, gridcolor="#e5e7eb", zeroline=False),
            zaxis=dict(showbackground=False, gridcolor="#e5e7eb", zeroline=False),
        ),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#111827"),
    )

    for idx in range(n_ts):
        fig.update_xaxes(
            type="date",
            title_text="date" if idx == n_ts - 1 else "",
            showticklabels=True,
            nticks=4,
            tickfont=dict(size=10),
            showgrid=True,
            gridcolor="#f3f4f6",
            zeroline=False,
            fixedrange=False,
            row=2 + idx,
            col=1,
        )
        fig.update_yaxes(
            title_text="z-score" if idx == 0 else "",
            showgrid=True,
            gridcolor="#f3f4f6",
            zeroline=False,
            fixedrange=False,
            row=2 + idx,
            col=1,
        )

    sensor_meta = [
        {
            "site": meta[idx][0],
            "type": meta[idx][1],
            "axis": meta[idx][2],
            "cluster": int(labels[idx]),
            "totalVar": float(total_var[idx]),
            "meanVar": float(mean_var[idx]),
        }
        for idx in range(n)
    ]

    data_blob = json.dumps(
        {
            "summary": {
                "n": int(n),
                "embedDim": int(embed_dim),
                "k": int(k_neighbors),
                "varianceAvailable": bool(variance_available),
            },
            "xs": sensor_xs,
            "ys": sensor_ys,
            "ranges": sensor_ranges,
            "names": sensor_names,
            "meta": sensor_meta,
            "metrics": [
                {"key": key, **METRIC_SPECS[key]} for key in METRIC_SPECS.keys()
            ],
            "defaultMetric": default_metric,
            "defaultProjection": default_projection,
            "projections": {
                "standard_pca": {
                    "label": "Standard PCA",
                    "x": pca[:, 0].astype(float).round(8).tolist(),
                    "y": pca[:, 1].astype(float).round(8).tolist(),
                    "z": pca[:, 2].astype(float).round(8).tolist(),
                },
                "weighted_pca": {
                    "label": "Variance-weighted PCA",
                    "x": pca_w[:, 0].astype(float).round(8).tolist(),
                    "y": pca_w[:, 1].astype(float).round(8).tolist(),
                    "z": pca_w[:, 2].astype(float).round(8).tolist(),
                },
            },
            "colors": color_options,
            "nn": {
                key: value.astype(int).tolist()
                for key, value in nn_idx.items()
            },
            "dist": {
                key: value.astype(float).round(6).tolist()
                for key, value in nn_dist.items()
            },
            "meanNeighborDistance": {
                key: value.astype(float).round(6).tolist()
                for key, value in mean_neighbor_distance.items()
            },
            "traceColors": TRACE_COLORS,
        },
        separators=(",", ":"),
    )

    post_script = (
        "var __EMB = "
        + data_blob
        + ";\n"
        "var gd = document.getElementById('{plot_id}');\n"
        "var __state = {\n"
        "  metric: __EMB.defaultMetric,\n"
        "  projection: __EMB.defaultProjection,\n"
        "  color: 'cluster',\n"
        "  selected: -1,\n"
        "  shifts: []\n"
        "};\n"
        "var __HIGHLIGHT_TRACE = 1;\n"
        "var __TS_TRACE_OFFSET = 2;\n"
        "var __renderToken = 0;\n"
        "var __pendingSensor = -1;\n"
        "var __renderFrame = 0;\n"
        "var __lastSpikeX = null;\n"
        "var __WEEK_MS = 7 * 24 * 60 * 60 * 1000;\n"
        "var __QUAL = ['#2563eb','#dc2626','#059669','#9333ea','#d97706',"
        "'#0891b2','#be185d','#4b5563','#65a30d','#7c3aed','#0f766e',"
        "'#b45309','#4338ca'];\n"
        "var __nameToIndex = {};\n"
        "for (var i = 0; i < __EMB.names.length; i++) __nameToIndex[__EMB.names[i]] = i;\n"
        "function __escape(value) {\n"
        "  return String(value).replace(/[&<>\"']/g, function(ch) {\n"
        "    return {'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch];\n"
        "  });\n"
        "}\n"
        "function __fmt(value) {\n"
        "  value = Number(value);\n"
        "  if (!isFinite(value)) return 'n/a';\n"
        "  if (value === 0) return '0';\n"
        "  var absValue = Math.abs(value);\n"
        "  if (absValue < 0.001 || absValue >= 10000) return value.toExponential(2);\n"
        "  if (absValue < 1) return value.toFixed(4);\n"
        "  if (absValue < 100) return value.toFixed(3);\n"
        "  return value.toFixed(1);\n"
        "}\n"
        "function __metricSpec(key) {\n"
        "  for (var i = 0; i < __EMB.metrics.length; i++) {\n"
        "    if (__EMB.metrics[i].key === key) return __EMB.metrics[i];\n"
        "  }\n"
        "  return __EMB.metrics[0];\n"
        "}\n"
        "function __axSuffix(j) { return j === 0 ? '' : (j + 1).toString(); }\n"
        "function __formatShift(value) {\n"
        "  value = Number(value || 0);\n"
        "  var fixed = Math.abs(value % 1) < 0.001 ? value.toFixed(0) : value.toFixed(2);\n"
        "  return (value > 0 ? '+' : '') + fixed + 'w';\n"
        "}\n"
        "function __shiftDateValue(value, weeks) {\n"
        "  if (!weeks || value === null || value === undefined) return value;\n"
        "  var text = String(value);\n"
        "  if (/^\\d{4}-\\d{2}-\\d{2}/.test(text)) {\n"
        "    var ts = Date.parse(text);\n"
        "    if (isFinite(ts)) return new Date(ts + weeks * __WEEK_MS).toISOString().slice(0, 10);\n"
        "  }\n"
        "  var num = Number(value);\n"
        "  return isFinite(num) ? String(num + weeks) : value;\n"
        "}\n"
        "function __shiftX(values, weeks) {\n"
        "  if (!weeks) return values;\n"
        "  return values.map(function(value) { return __shiftDateValue(value, weeks); });\n"
        "}\n"
        "function __populateSelect(el, options, selected) {\n"
        "  if (!el) return;\n"
        "  el.innerHTML = '';\n"
        "  options.forEach(function(option) {\n"
        "    var item = document.createElement('option');\n"
        "    item.value = option.key;\n"
        "    item.textContent = option.label;\n"
        "    if (option.key === selected) item.selected = true;\n"
        "    el.appendChild(item);\n"
        "  });\n"
        "}\n"
        "function __stepColorscale(count) {\n"
        "  var out = [];\n"
        "  count = Math.max(count, 1);\n"
        "  for (var i = 0; i < count; i++) {\n"
        "    var color = __QUAL[i % __QUAL.length];\n"
        "    out.push([i / count, color]);\n"
        "    out.push([(i + 1) / count, color]);\n"
        "  }\n"
        "  return out;\n"
        "}\n"
        "function __finiteMin(values) {\n"
        "  var out = Infinity;\n"
        "  values.forEach(function(value) { if (isFinite(value)) out = Math.min(out, value); });\n"
        "  return isFinite(out) ? out : 0;\n"
        "}\n"
        "function __finiteMax(values) {\n"
        "  var out = -Infinity;\n"
        "  values.forEach(function(value) { if (isFinite(value)) out = Math.max(out, value); });\n"
        "  return isFinite(out) ? out : 1;\n"
        "}\n"
        "function __applyColor(key) {\n"
        "  var spec = __EMB.colors[key];\n"
        "  if (!spec) return;\n"
        "  __state.color = key;\n"
        "  var update = {'marker.color': [spec.values], 'marker.colorbar.title.text': [spec.label]};\n"
        "  if (spec.type === 'categorical') {\n"
        "    update['marker.colorscale'] = [__stepColorscale(spec.categories.length)];\n"
        "    update['marker.cmin'] = [-0.5];\n"
        "    update['marker.cmax'] = [spec.categories.length - 0.5];\n"
        "    update['marker.colorbar.tickvals'] = [spec.categories.map(function(_, i) { return i; })];\n"
        "    update['marker.colorbar.ticktext'] = [spec.categories];\n"
        "  } else {\n"
        "    update['marker.colorscale'] = [spec.colorscale || 'Viridis'];\n"
        "    update['marker.cmin'] = [__finiteMin(spec.values)];\n"
        "    update['marker.cmax'] = [__finiteMax(spec.values)];\n"
        "    update['marker.colorbar.tickvals'] = [null];\n"
        "    update['marker.colorbar.ticktext'] = [null];\n"
        "  }\n"
        "  Plotly.restyle(gd, update, [0]);\n"
        "}\n"
        "function __applyProjection(key) {\n"
        "  var projection = __EMB.projections[key];\n"
        "  if (!projection) return;\n"
        "  __state.projection = key;\n"
        "  Plotly.restyle(gd, {x: [projection.x], y: [projection.y], z: [projection.z]}, [0])\n"
        "    .then(function() { if (__state.selected >= 0) __updateHighlightOnly(__state.selected); });\n"
        "}\n"
        "function __findSensor(value) {\n"
        "  value = String(value || '').trim();\n"
        "  if (Object.prototype.hasOwnProperty.call(__nameToIndex, value)) return __nameToIndex[value];\n"
        "  var lower = value.toLowerCase();\n"
        "  for (var i = 0; i < __EMB.names.length; i++) {\n"
        "    if (__EMB.names[i].toLowerCase().indexOf(lower) !== -1) return i;\n"
        "  }\n"
        "  return -1;\n"
        "}\n"
        "function __buildPlotPayload(i) {\n"
        "  var neighbors = __EMB.nn[__state.metric][i];\n"
        "  var distances = __EMB.dist[__state.metric][i];\n"
        "  var src = gd.data[0];\n"
        "  var hx = [], hy = [], hz = [], hc = [], hs = [];\n"
        "  var xs = [], ys = [], names = [], idx = [];\n"
        "  var layoutUpdate = {};\n"
        "  for (var j = 0; j < neighbors.length; j++) {\n"
        "    var nb = neighbors[j];\n"
        "    var shift = __state.shifts[j] || 0;\n"
        "    hx.push(src.x[nb]);\n"
        "    hy.push(src.y[nb]);\n"
        "    hz.push(src.z[nb]);\n"
        "    hc.push(__EMB.traceColors[j % __EMB.traceColors.length]);\n"
        "    hs.push(j === 0 ? 14 : 10);\n"
        "    xs.push(__shiftX(__EMB.xs[nb], shift));\n"
        "    ys.push(__EMB.ys[nb]);\n"
        "    names.push((j === 0 ? 'A  ' : j + '  ') + __EMB.names[nb]\n"
        "      + (j === 0 ? '' : '  d=' + __fmt(distances[j]))\n"
        "      + (shift ? '  shift=' + __formatShift(shift) : ''));\n"
        "    idx.push(__TS_TRACE_OFFSET + j);\n"
        "    var suffix = __axSuffix(j);\n"
        "    var range = __EMB.ranges[nb];\n"
        "    if (range && range.x && range.x[0] !== null && range.x[1] !== null) {\n"
        "      layoutUpdate['xaxis' + suffix + '.range'] = range.x;\n"
        "      layoutUpdate['xaxis' + suffix + '.autorange'] = false;\n"
        "    } else {\n"
        "      layoutUpdate['xaxis' + suffix + '.autorange'] = true;\n"
        "    }\n"
        "    if (range && range.y && range.y[0] !== null && range.y[1] !== null) {\n"
        "      layoutUpdate['yaxis' + suffix + '.range'] = range.y;\n"
        "      layoutUpdate['yaxis' + suffix + '.autorange'] = false;\n"
        "    } else {\n"
        "      layoutUpdate['yaxis' + suffix + '.autorange'] = true;\n"
        "    }\n"
        "  }\n"
        "  return {\n"
        "    highlight: {x: hx, y: hy, z: hz, color: hc, size: hs},\n"
        "    ts: {x: xs, y: ys, name: names, idx: idx},\n"
        "    layout: layoutUpdate\n"
        "  };\n"
        "}\n"
        "function __updateHighlightOnly(i) {\n"
        "  var payload = __buildPlotPayload(i);\n"
        "  return Plotly.restyle(gd,\n"
        "    {x: [payload.highlight.x], y: [payload.highlight.y], z: [payload.highlight.z],\n"
        "     'marker.color': [payload.highlight.color], 'marker.size': [payload.highlight.size]},\n"
        "    [__HIGHLIGHT_TRACE]);\n"
        "}\n"
        "function __updatePlot(i) {\n"
        "  var token = ++__renderToken;\n"
        "  var payload = __buildPlotPayload(i);\n"
        "  Plotly.update(gd,\n"
        "    {x: payload.ts.x, y: payload.ts.y, name: payload.ts.name},\n"
        "    payload.layout,\n"
        "    payload.ts.idx)\n"
        "    .then(function() {\n"
        "      if (token !== __renderToken || __state.selected !== i) return;\n"
        "      return Plotly.restyle(gd,\n"
        "        {x: [payload.highlight.x], y: [payload.highlight.y], z: [payload.highlight.z],\n"
        "         'marker.color': [payload.highlight.color], 'marker.size': [payload.highlight.size]},\n"
        "        [__HIGHLIGHT_TRACE]);\n"
        "    });\n"
        "}\n"
        "function __schedulePlotUpdate(i) {\n"
        "  __pendingSensor = i;\n"
        "  if (__renderFrame) return;\n"
        "  __renderFrame = requestAnimationFrame(function() {\n"
        "    var next = __pendingSensor;\n"
        "    __pendingSensor = -1;\n"
        "    __renderFrame = 0;\n"
        "    __updatePlot(next);\n"
        "  });\n"
        "}\n"
        "function __updateSelectionPanel(i) {\n"
        "  var selectedCard = document.getElementById('selectedCard');\n"
        "  var metricHint = document.getElementById('metricHint');\n"
        "  var metricColumn = document.getElementById('metricColumn');\n"
        "  var spec = __metricSpec(__state.metric);\n"
        "  var meta = __EMB.meta[i];\n"
        "  if (metricHint) metricHint.textContent = spec.description;\n"
        "  if (metricColumn) metricColumn.textContent = spec.label;\n"
        "  if (!selectedCard) return;\n"
        "  selectedCard.innerHTML = ''\n"
        "    + '<div class=\"selected-name\">' + __escape(__EMB.names[i]) + '</div>'\n"
        "    + '<div class=\"selected-meta\">'\n"
        "    + '<div><span>cluster</span><br>' + meta.cluster + '</div>'\n"
        "    + '<div><span>site</span><br>' + __escape(meta.site) + '</div>'\n"
        "    + '<div><span>type</span><br>' + __escape(meta.type) + '</div>'\n"
        "    + '<div><span>axis</span><br>' + __escape(meta.axis) + '</div>'\n"
        "    + '<div><span>total var</span><br>' + __fmt(meta.totalVar) + '</div>'\n"
        "    + '<div><span>mean var</span><br>' + __fmt(meta.meanVar) + '</div>'\n"
        "    + '<div><span>avg top-k</span><br>' + __fmt(__EMB.meanNeighborDistance[__state.metric][i]) + '</div>'\n"
        "    + '<div><span>metric</span><br>' + __escape(spec.label) + '</div>'\n"
        "    + '</div>';\n"
        "}\n"
        "function __updateNeighborTable(i) {\n"
        "  var body = document.getElementById('neighborRows');\n"
        "  if (!body) return;\n"
        "  var neighbors = __EMB.nn[__state.metric][i];\n"
        "  var distances = __EMB.dist[__state.metric][i];\n"
        "  body.innerHTML = '';\n"
        "  for (var j = 0; j < neighbors.length; j++) {\n"
        "    var nb = neighbors[j];\n"
        "    var row = document.createElement('tr');\n"
        "    row.setAttribute('data-index', nb);\n"
        "    var rank = j === 0 ? 'A' : String(j);\n"
        "    var color = __EMB.traceColors[j % __EMB.traceColors.length];\n"
        "    row.innerHTML = ''\n"
        "      + '<td class=\"rank-cell\"><span class=\"rank-dot\" style=\"background:' + color + '\"></span>' + rank + '</td>'\n"
        "      + '<td class=\"sensor-cell\" title=\"' + __escape(__EMB.names[nb]) + '\">' + __escape(__EMB.names[nb]) + '</td>'\n"
        "      + '<td class=\"distance-cell\">' + __fmt(distances[j]) + '</td>';\n"
        "    row.addEventListener('click', function() {\n"
        "      __showSensor(Number(this.getAttribute('data-index')));\n"
        "    });\n"
        "    body.appendChild(row);\n"
        "  }\n"
        "}\n"
        "function __resetShifts(updatePlot) {\n"
        "  __state.shifts = new Array(__EMB.summary.k + 1).fill(0);\n"
        "  if (updatePlot && __state.selected >= 0) {\n"
        "    __updateAlignControls(__state.selected);\n"
        "    __schedulePlotUpdate(__state.selected);\n"
        "  }\n"
        "}\n"
        "function __updateAlignControls(i) {\n"
        "  var rows = document.getElementById('alignRows');\n"
        "  if (!rows) return;\n"
        "  var neighbors = __EMB.nn[__state.metric][i];\n"
        "  var fragment = document.createDocumentFragment();\n"
        "  rows.innerHTML = '';\n"
        "  for (var j = 0; j < neighbors.length; j++) {\n"
        "    var nb = neighbors[j];\n"
        "    var row = document.createElement('div');\n"
        "    row.className = 'align-row';\n"
        "    var rank = j === 0 ? 'A' : String(j);\n"
        "    var value = __state.shifts[j] || 0;\n"
        "    row.innerHTML = ''\n"
        "      + '<span class=\"align-rank\"><span class=\"rank-dot\" style=\"background:' + __EMB.traceColors[j % __EMB.traceColors.length] + '\"></span>' + rank + '</span>'\n"
        "      + '<input type=\"range\" min=\"-104\" max=\"104\" step=\"0.25\" value=\"' + value + '\" data-rank=\"' + j + '\" title=\"' + __escape(__EMB.names[nb]) + '\">'\n"
        "      + '<span class=\"align-value\">' + __formatShift(value) + '</span>';\n"
        "    var slider = row.querySelector('input');\n"
        "    slider.addEventListener('input', function() {\n"
        "      var rankIdx = Number(this.getAttribute('data-rank'));\n"
        "      var shiftValue = Number(this.value);\n"
        "      __state.shifts[rankIdx] = shiftValue;\n"
        "      this.parentNode.querySelector('.align-value').textContent = __formatShift(shiftValue);\n"
        "      if (__state.selected >= 0) __schedulePlotUpdate(__state.selected);\n"
        "    });\n"
        "    fragment.appendChild(row);\n"
        "  }\n"
        "  rows.appendChild(fragment);\n"
        "}\n"
        "function __showSensor(i) {\n"
        "  if (!isFinite(i) || i < 0 || i >= __EMB.names.length) return;\n"
        "  var changed = __state.selected !== i;\n"
        "  __state.selected = i;\n"
        "  if (changed) __resetShifts(false);\n"
        "  var search = document.getElementById('sensorSearch');\n"
        "  if (search) search.value = __EMB.names[i];\n"
        "  __updateSelectionPanel(i);\n"
        "  __updateNeighborTable(i);\n"
        "  __updateAlignControls(i);\n"
        "  __schedulePlotUpdate(i);\n"
        "}\n"
        "function __drawSpike(xValue) {\n"
        "  if (__lastSpikeX === xValue) return;\n"
        "  __lastSpikeX = xValue;\n"
        "  var shapes = [];\n"
        "  for (var j = 0; j <= __EMB.summary.k; j++) {\n"
        "    var suffix = __axSuffix(j);\n"
        "    shapes.push({type: 'line', xref: 'x' + suffix, yref: 'y' + suffix + ' domain', x0: xValue, x1: xValue, y0: 0, y1: 1, line: {color: 'rgba(75,85,99,0.55)', width: 1, dash: 'dot'}, layer: 'above'});\n"
        "  }\n"
        "  Plotly.relayout(gd, {shapes: shapes});\n"
        "}\n"
        "function __clearSpike() {\n"
        "  if (__lastSpikeX === null) return;\n"
        "  __lastSpikeX = null;\n"
        "  Plotly.relayout(gd, {shapes: []});\n"
        "}\n"
        "function __isTimeSeriesPoint(point) {\n"
        "  return point && point.data && (point.data.type === 'scatter' || point.data.type === 'scattergl');\n"
        "}\n"
        "function __initControls() {\n"
        "  var summary = document.getElementById('datasetSummary');\n"
        "  if (summary) {\n"
        "    summary.textContent = __EMB.summary.n + ' sensors, ' + __EMB.summary.embedDim + ' dims, k=' + __EMB.summary.k;\n"
        "  }\n"
        "  var datalist = document.getElementById('sensorOptions');\n"
        "  if (datalist) {\n"
        "    __EMB.names.forEach(function(name) {\n"
        "      var option = document.createElement('option');\n"
        "      option.value = name;\n"
        "      datalist.appendChild(option);\n"
        "    });\n"
        "  }\n"
        "  var metricSelect = document.getElementById('metricSelect');\n"
        "  var projectionSelect = document.getElementById('projectionSelect');\n"
        "  var colorSelect = document.getElementById('colorSelect');\n"
        "  __resetShifts(false);\n"
        "  __populateSelect(metricSelect, __EMB.metrics, __state.metric);\n"
        "  __populateSelect(projectionSelect, Object.keys(__EMB.projections).map(function(key) { return {key: key, label: __EMB.projections[key].label}; }), __state.projection);\n"
        "  __populateSelect(colorSelect, Object.keys(__EMB.colors).map(function(key) { return {key: key, label: __EMB.colors[key].label}; }), __state.color);\n"
        "  if (metricSelect) metricSelect.addEventListener('change', function() { __state.metric = this.value; if (__state.selected >= 0) __showSensor(__state.selected); });\n"
        "  if (projectionSelect) projectionSelect.addEventListener('change', function() { __applyProjection(this.value); });\n"
        "  if (colorSelect) colorSelect.addEventListener('change', function() { __applyColor(this.value); });\n"
        "  var resetAlign = document.getElementById('resetAlign');\n"
        "  if (resetAlign) resetAlign.addEventListener('click', function() { __resetShifts(true); });\n"
        "  var sensorSearch = document.getElementById('sensorSearch');\n"
        "  if (sensorSearch) {\n"
        "    sensorSearch.addEventListener('keydown', function(event) {\n"
        "      if (event.key === 'Enter') {\n"
        "        var idx = __findSensor(this.value);\n"
        "        if (idx >= 0) __showSensor(idx);\n"
        "      }\n"
        "    });\n"
        "    sensorSearch.addEventListener('change', function() {\n"
        "      var idx = __findSensor(this.value);\n"
        "      if (idx >= 0) __showSensor(idx);\n"
        "    });\n"
        "  }\n"
        "}\n"
        "gd.on('plotly_click', function(event) {\n"
        "  if (!event.points || !event.points.length) return;\n"
        "  var point = event.points[0];\n"
        "  if (point.data.type === 'scatter3d' && point.curveNumber === 0) __showSensor(point.pointNumber);\n"
        "});\n"
        "gd.on('plotly_hover', function(event) {\n"
        "  if (!event.points || !event.points.length) return;\n"
        "  var point = event.points[0];\n"
        "  if (__isTimeSeriesPoint(point)) __drawSpike(point.x);\n"
        "});\n"
        "gd.on('plotly_unhover', function(event) {\n"
        "  if (event && event.points && event.points.length && __isTimeSeriesPoint(event.points[0])) __clearSpike();\n"
        "});\n"
        "window.__showSensor = __showSensor;\n"
        "__initControls();\n"
        "__applyColor(__state.color);\n"
        "__showSensor(0);\n"
    )

    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        post_script=post_script,
        full_html=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(build_html_document(plot_html))

    print(f"Saved interactive explorer -> {out_html}")


if __name__ == "__main__":
    fire.Fire(main)

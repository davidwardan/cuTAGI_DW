import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform

# This script assumes you have a directory structure like:
# out/
# └── toy_shared_embeddings/
#     ├── amp_embeddings/
#     │   ├── embeddings_mu.csv
#     │   └── embeddings_var.csv
#     ├── period_embeddings/
#     │   ├── embeddings_mu.csv
#     │   └── embeddings_var.csv
#     └── wave_embeddings/
#         ├── embeddings_mu.csv
#         └── embeddings_var.csv
#
# If your files don't exist, this script will create dummy files to run.
import os

# ----------------------------------------------------------------------
# Distance / divergence functions for diagonal‑covariance Gaussians
# ----------------------------------------------------------------------
def sym_kl(mu1, var1, mu2, var2):
    """Symmetric KL divergence between N(mu1,var1) and N(mu2,var2)."""
    def kl(a_mu, a_var, b_mu, b_var):
        ratio = a_var / b_var
        diff2 = ((b_mu - a_mu) ** 2) / b_var
        return 0.5 * np.sum(np.log(b_var / a_var) + ratio + diff2 - 1)
    return kl(mu1, var1, mu2, var2) + kl(mu2, var2, mu1, var1)

def wasserstein2(mu1, var1, mu2, var2):
    """Squared 2‑Wasserstein distance between two diagonal Gaussians."""
    return np.sum((mu1 - mu2) ** 2 + (np.sqrt(var1) - np.sqrt(var2)) ** 2)

def mahalanobis_avg(mu1, var1, mu2, var2):
    """Mahalanobis using the average variance as metric tensor."""
    avg_var = 0.5 * (var1 + var2)
    return np.sum((mu1 - mu2) ** 2 / avg_var)

def pairwise_diag_metric(mus, vars, fn):
    """Compute full pairwise matrix using the supplied fn."""
    n = mus.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(mus[i], vars[i], mus[j], vars[j])
            D[i, j] = D[j, i] = d
    return D


def create_dummy_data(embed_dir="out/toy_shared_embeddings/"):
    """Creates dummy embedding files if they don't exist."""
    dirs = [
        embed_dir + "wave_embeddings/",
        embed_dir + "amp_embeddings/",
        embed_dir + "period_embeddings/",
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
        # Create dummy files with 3 embeddings of dimension 10
        if not os.path.exists(d + "embeddings_mu.csv"):
            pd.DataFrame(np.random.rand(3, 10)).to_csv(
                d + "embeddings_mu.csv", header=False, index=False
            )
        if not os.path.exists(d + "embeddings_var.csv"):
            pd.DataFrame(np.random.rand(3, 10) * 0.1).to_csv(
                d + "embeddings_var.csv", header=False, index=False
            )
    print("Dummy data created for demonstration.")


def read_embeddings(file_path):
    """Reads wave embeddings from CSV files."""
    embeddings_mu = pd.read_csv(file_path + "embeddings_mu.csv", header=None).values
    embeddings_var = pd.read_csv(file_path + "embeddings_var.csv", header=None).values
    return embeddings_mu, embeddings_var


def main(embed_dir="out/toy_shared_embeddings/"):
    """
    Main function to load embeddings, perform PCA, and generate visualizations.
    """
    # Create dummy data if the specified directory doesn't exist
    if not os.path.exists(embed_dir):
        print(f"Directory '{embed_dir}' not found.")
        create_dummy_data(embed_dir)

    wave_names = ["Sin", "Square", "Triangular"]
    amp_names = ["Amp0.5x", "Amp1x", "Amp1.5x"]
    per_names = ["Period0.5x", "Period1x", "Period1.5x"]

    # Load embeddings for each property type
    wave_embeddings_mu, wave_embeddings_var = read_embeddings(
        embed_dir + "wave_embeddings/"
    )
    amp_embeddings_mu, amp_embeddings_var = read_embeddings(
        embed_dir + "amp_embeddings/"
    )
    period_embeddings_mu, period_embeddings_var = read_embeddings(
        embed_dir + "period_embeddings/"
    )

    # --- Part 1: 3-D PCA of individual property embeddings (Matplotlib) ---
    embeddings = np.concatenate(
        [wave_embeddings_mu, amp_embeddings_mu, period_embeddings_mu], axis=0
    )
    embeddings_var = np.concatenate(
        [wave_embeddings_var, amp_embeddings_var, period_embeddings_var], axis=0
    )  # shape (9, d)
    pca_initial = PCA(n_components=3)
    pca_result_initial = pca_initial.fit_transform(embeddings)
    expl_var = pca_initial.explained_variance_ratio_

    df_individual = pd.DataFrame(
        {
            "PC1": pca_result_initial[:, 0],
            "PC2": pca_result_initial[:, 1],
            "PC3": pca_result_initial[:, 2],
            "Label": wave_names + amp_names + per_names,
            "Category": (
                ["Wave"] * len(wave_names)
                + ["Amplitude"] * len(amp_names)
                + ["Period"] * len(per_names)
            ),
        }
    )

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")

    # colour / marker per category
    cat_colour = {"Wave": "red", "Amplitude": "blue", "Period": "green"}
    cat_marker = {"Wave": "o", "Amplitude": "^", "Period": "s"}

    for _, row in df_individual.iterrows():
        ax1.scatter(
            row["PC1"],
            row["PC2"],
            row["PC3"],
            color=cat_colour[row["Category"]],
            marker=cat_marker[row["Category"]],
            s=60,
            edgecolors="k",
            alpha=0.9,
        )
        ax1.text(
            row["PC1"],
            row["PC2"],
            row["PC3"],
            row["Label"],
            fontsize=8,
            ha="center",
            va="center",
        )

    handles = [
        Line2D([0], [0], marker=cat_marker[c], color="w",
               label=c, markerfacecolor=cat_colour[c],
               markersize=8, markeredgecolor="k")
        for c in ["Wave", "Amplitude", "Period"]
    ]
    ax1.legend(handles=handles, title="Category", loc="best")

    ax1.set_xlabel(f"PC1 ({expl_var[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({expl_var[1]*100:.1f}%)")
    ax1.set_zlabel(f"PC3 ({expl_var[2]*100:.1f}%)")
    ax1.set_title("3‑D PCA of Individual Embeddings")
    fig1.tight_layout()
    fig1.savefig("embedding_individual.png", dpi=300)
    plt.show()

    # --- Part 2: 3-D PCA of time-series triplet embeddings (Matplotlib) ---
    time_series_embeddings = []
    ts_labels, ts_wave, ts_amp, ts_per = [], [], [], []

    for wave_idx, w_name in enumerate(wave_names):
        for amp_idx, a_name in enumerate(amp_names):
            for per_idx, p_name in enumerate(per_names):
                ts_embed = np.concatenate(
                    (
                        wave_embeddings_mu[wave_idx],
                        amp_embeddings_mu[amp_idx],
                        period_embeddings_mu[per_idx],
                    ),
                    axis=0,
                )
                time_series_embeddings.append(ts_embed)
                ts_labels.append(f"{w_name}_{a_name}_{p_name}")
                ts_wave.append(w_name)
                ts_amp.append(a_name)
                ts_per.append(p_name)

    embeddings_matrix = np.array(time_series_embeddings)

    # Perform PCA on the combined time series embeddings
    pca_combined = PCA(n_components=3)
    pca_result_combined = pca_combined.fit_transform(embeddings_matrix)
    explained_variance = pca_combined.explained_variance_ratio_

    # Assemble a DataFrame for easy plotting with Matplotlib
    df_plot = pd.DataFrame(
        {
            "PC1": pca_result_combined[:, 0],
            "PC2": pca_result_combined[:, 1],
            "PC3": pca_result_combined[:, 2],
            "Wave": ts_wave,
            "Amplitude": ts_amp,
            "Period": ts_per,
            "Label": ts_labels,
        }
    )

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection="3d")

    colour_map = {"Sin": "red", "Square": "blue", "Triangular": "green"}
    marker_map = {"Amp0.5x": "o", "Amp1x": "s", "Amp1.5x": "D"}
    size_map   = {"Period0.5x": 40, "Period1x": 80, "Period1.5x": 120}

    for _, row in df_plot.iterrows():
        ax2.scatter(
            row["PC1"], row["PC2"], row["PC3"],
            color=colour_map[row["Wave"]],
            marker=marker_map[row["Amplitude"]],
            s=size_map[row["Period"]],
            edgecolors="k",
            alpha=0.8,
        )

    # Legends
    wave_handles = [Line2D([0], [0], marker="o", color="w",
                           label=w, markerfacecolor=c, markeredgecolor="k",
                           markersize=8) for w, c in colour_map.items()]
    amp_handles  = [Line2D([0], [0], marker=m, color="k",
                           label=a, markerfacecolor="k",
                           markersize=8) for a, m in marker_map.items()]
    period_handles = [Line2D([0], [0], marker="o", color="k",
                             label=p, markerfacecolor="k",
                             markersize=(s**0.5)) for p, s in size_map.items()]

    leg1 = ax2.legend(handles=wave_handles, title="Wave (colour)",
                      loc="upper left", bbox_to_anchor=(1.05, 1))
    leg2 = ax2.legend(handles=amp_handles, title="Amplitude (shape)",
                      loc="upper left", bbox_to_anchor=(1.05, 0.7))
    leg3 = ax2.legend(handles=period_handles, title="Period (size)",
                      loc="upper left", bbox_to_anchor=(1.05, 0.4))
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)

    ax2.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    ax2.set_zlabel(f"PC3 ({explained_variance[2]*100:.1f}%)")
    ax2.set_title("3‑D PCA of Time‑Series Embeddings")
    fig2.tight_layout()
    fig2.savefig("embedding_triplets.png", dpi=300)
    plt.show()

    # ------------------------------------------------------------------
    # Part 3 – Similarity metrics
    #   • mean‑only: cosine, Euclidean
    #   • mean+variance: symmetric‑KL, Wasserstein‑2, Mahalanobis
    # ------------------------------------------------------------------
    labels_individual = wave_names + amp_names + per_names

    # ===== Individual embeddings (9) =====
    cosine_ind   = 1 - squareform(pdist(embeddings, metric="cosine"))
    euc_ind      = squareform(pdist(embeddings, metric="euclidean"))
    skl_ind      = pairwise_diag_metric(embeddings, embeddings_var, sym_kl)
    w2_ind       = pairwise_diag_metric(embeddings, embeddings_var, wasserstein2)
    mahal_ind    = pairwise_diag_metric(embeddings, embeddings_var, mahalanobis_avg)

    out_tables = {
        "cosine":       pd.DataFrame(cosine_ind, index=labels_individual, columns=labels_individual),
        "euclidean":    pd.DataFrame(euc_ind,    index=labels_individual, columns=labels_individual),
        "symKL":        pd.DataFrame(skl_ind,    index=labels_individual, columns=labels_individual),
        "W2_sq":        pd.DataFrame(w2_ind,     index=labels_individual, columns=labels_individual),
        "mahalanobis":  pd.DataFrame(mahal_ind,  index=labels_individual, columns=labels_individual),
    }

    # ------------------------------------------------------------------
    #  A. Sub‑embedding (9×9) similarities
    # ------------------------------------------------------------------
    print("\n=== Individual‑embedding similarity matrices (first 3×3 preview) ===")
    for name, df_tab in out_tables.items():
        print(f"\n{name}:")
        print(df_tab.iloc[:3, :3].round(3))          # small preview
        df_tab.to_csv(f"individual_{name}.csv")      # full table to CSV

    # summary statistics across ALL upper‑triangle pairs
    upper_ind = np.triu_indices_from(cosine_ind, k=1)
    print("\n--- Individual‑embedding summary (mean ± std) ---")
    for name, df_tab in out_tables.items():
        vals = df_tab.values[upper_ind]
        print(f"{name:12}: {vals.mean():.4f} ± {vals.std():.4f}")

    # ===== Triplet embeddings (27) =====
    # build mu & var matrices for triplets
    triplet_vars = []
    for wave_idx in range(len(wave_names)):
        for amp_idx in range(len(amp_names)):
            for per_idx in range(len(per_names)):
                triplet_vars.append(
                    np.concatenate(
                        (
                            wave_embeddings_var[wave_idx],
                            amp_embeddings_var[amp_idx],
                            period_embeddings_var[per_idx],
                        ),
                        axis=0,
                    )
                )
    triplet_vars = np.array(triplet_vars)  # (27, d)

    cosine_trip = 1 - squareform(pdist(embeddings_matrix, metric="cosine"))
    euc_trip    = squareform(pdist(embeddings_matrix, metric="euclidean"))
    skl_trip    = pairwise_diag_metric(embeddings_matrix, triplet_vars, sym_kl)
    w2_trip     = pairwise_diag_metric(embeddings_matrix, triplet_vars, wasserstein2)
    mahal_trip  = pairwise_diag_metric(embeddings_matrix, triplet_vars, mahalanobis_avg)

    trip_tables = {
        "cosine":      cosine_trip,
        "euclidean":   euc_trip,
        "symKL":       skl_trip,
        "W2_sq":       w2_trip,
        "mahalanobis": mahal_trip,
    }

    upper = np.triu_indices_from(cosine_trip, k=1)
    print("\n=== Triplet‑level summary (mean±std) ===")
    for name, mat in trip_tables.items():
        print(f"{name:12}: {mat[upper].mean():.4f} ± {mat[upper].std():.4f}")
        pd.DataFrame(mat, index=ts_labels, columns=ts_labels).to_csv(f"triplet_{name}.csv")


if __name__ == "__main__":
    main()

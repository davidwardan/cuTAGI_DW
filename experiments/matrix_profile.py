import numpy as np
import pandas as pd
import stumpy
from tqdm import tqdm

from experiments.utils import plot_similarity, plot_similarity_graph


def matrix_profile_similarity(df: pd.DataFrame, m: int = 50) -> np.ndarray:
    cols = df.columns
    n = len(cols)
    sim_matrix = np.full((n, n), np.nan, dtype=float)

    for i in tqdm(range(n), desc="Computing pairwise similarities"):
        for j in range(i, n):
            x = df[cols[i]].dropna().to_numpy()
            y = df[cols[j]].dropna().to_numpy()

            if i == j:
                sim = 1.0 # faster to set self-similarity to 1
            else:
                if (len(x) < m) or (len(y) < m):
                    sim = np.nan
                else:
                    mp = stumpy.stump(T_A=x, T_B=y, m=m)
                    dist = np.nanmean(mp[:, 0])
                    sim = 1 / (1 + dist) # convert distance to similarity

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


if __name__ == "__main__":
    # read the training set data
    data_file = "data/hq/train100/split_train_values.csv"
    df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

    # get similarity matrix
    sim_matrix = matrix_profile_similarity(df, m=52)

    # plot heat map of similarity matrix
    # plot_similarity(
    #     sim_matrix,
    #     "out/matrix_profile_matrix.pdf",
    #     "Matrix Profile Similarity Matrix",
    #     vmin=0.0,
    # )

    plot_similarity_graph(sim_matrix, out_path="./out/", threshold=0.4)

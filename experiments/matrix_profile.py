import numpy as np
import pandas as pd
import stumpy
from tqdm import tqdm
import matplotlib.pyplot as plt

from experiments.utils import plot_similarity, plot_similarity_graph


def matrix_profile_similarity(
    df: pd.DataFrame, m: int = 50, adaptive: bool = True
) -> np.ndarray:
    cols = df.columns
    n = len(cols)
    sim_matrix = np.full((n, n), np.nan, dtype=float)

    for i in tqdm(range(n), desc="Computing pairwise similarities"):
        for j in range(i, n):
            x = df[cols[i]].dropna().to_numpy()
            y = df[cols[j]].dropna().to_numpy()

            if i == j:
                sim = 1.0
            else:
                m_eff = min(m, len(x) // 2, len(y) // 2) if adaptive else m

                # guard too-short series when adaptive=False (or after shrink)
                if (len(x) < m_eff) or (len(y) < m_eff) or (m_eff < 4):
                    if len(x) > 1 and len(y) > 1:
                        # fallback: correlation on resampled shapes
                        xs = np.interp(
                            np.linspace(0, 1, 100), np.linspace(0, 1, len(x)), x
                        )
                        ys = np.interp(
                            np.linspace(0, 1, 100), np.linspace(0, 1, len(y)), y
                        )
                        corr = np.corrcoef(xs, ys)[0, 1]
                        sim = (corr + 1) / 2
                    else:
                        sim = np.nan
                else:
                    # use keywords to avoid the signature clash
                    mp = stumpy.stump(T_A=x, T_B=y, m=m_eff)
                    dist = np.nanmean(mp[:, 0])
                    sim = 1 / (1 + dist)

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


if __name__ == "__main__":
    # read the training set data
    data_file = "data/hq/train100/split_train_values.csv"
    df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

    # get similarity matrix
    sim_matrix = matrix_profile_similarity(df, m=52, adaptive=True)

    # plot heat map of similarity matrix
    # plot_similarity(
    #     sim_matrix,
    #     "out/matrix_profile_matrix.pdf",
    #     "Matrix Profile Similarity Matrix",
    #     vmin=0.0,
    # )

    plot_similarity_graph(sim_matrix, out_path="./out/", threshold=0.4)

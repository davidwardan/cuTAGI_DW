import os
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytagi.nn import LSTM, Linear, Sequential


def generate_time_series(model, embedding, time_steps=100, look_back_len=24):

    ranges = np.arange(time_steps)

    look_back_mu = np.zeros(look_back_len, dtype=np.float32)
    look_back_var = np.ones(look_back_len, dtype=np.float32)

    generated_mu = np.zeros(time_steps, dtype=np.float32)
    generated_var = np.zeros(time_steps, dtype=np.float32)
    for step in ranges:
        # concatenate embedding with look_back_mu
        input_mu = np.concatenate((look_back_mu, embedding[0]), axis=0)
        input_var = np.concatenate((look_back_mu, embedding[1]), axis=0)

        # Generate the next step
        next_mu, next_var = model(np.float32(input_mu), np.float32(input_var))
        generated_mu[step] = next_mu
        generated_var[step] = next_var

        look_back_mu = np.roll(look_back_mu, -1)
        look_back_var = np.roll(look_back_var, -1)
        look_back_mu[-1] = next_mu
        look_back_var[-1] = next_var

    return generated_mu, generated_var


def init_model(params_dir: str, embedding_dim: int, look_back_len=24):
    """Initializes a model and loads its pretrained weights."""
    # The model architecture must match the one used for training
    net = Sequential(
        LSTM(1 + embedding_dim + look_back_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)

    model_path = os.path.join(params_dir, "model.path")
    if os.path.exists(model_path):
        net.load(model_path)
        print(f"Loaded pretrained model from {model_path}")
    else:
        print(f"ERROR: No pretrained model found at {model_path}. Cannot proceed.")
        net = None

    return net


def read_embedding_category(file_path):
    """Reads a single category of embeddings (mu and var) from CSV files."""
    try:
        mu_path = os.path.join(file_path, "embeddings_mu.csv")
        var_path = os.path.join(file_path, "embeddings_var.csv")
        embeddings_mu = pd.read_csv(mu_path, header=None).values
        embeddings_var = pd.read_csv(var_path, header=None).values
        return embeddings_mu, embeddings_var
    except FileNotFoundError:
        print(f"ERROR: Embedding files not found in {file_path}")
        return None, None


def main(time_steps=250):
    """Main function to generate and plot time series."""
    # --- Configuration ---
    look_back_len = 24
    embedding_dim_shared = 15  # 5 * 3 categories
    embedding_dim_standard = 15

    # --- Initialize Models ---
    print("Initializing models...")
    shared_model = init_model(
        "out/toy2_shared_embeddings", embedding_dim_shared, look_back_len
    )
    standard_model = init_model(
        "out/toy2_embeddings", embedding_dim_standard, look_back_len
    )

    if not all([shared_model, standard_model]):
        print("A model could not be loaded. Exiting.")
        return

    # --- Load Embeddings ---
    print("\nLoading embeddings...")
    # Load the three separate embedding categories for the shared model
    wave_emb = read_embedding_category("out/toy2_shared_embeddings/wave_embeddings")
    amp_emb = read_embedding_category("out/toy2_shared_embeddings/amp_embeddings")
    period_emb = read_embedding_category("out/toy2_shared_embeddings/period_embeddings")

    # Load the single embedding set for the standard model
    standard_emb = read_embedding_category("out/toy2_embeddings/embeddings")

    if not all(
        [
            wave_emb[0] is not None,
            amp_emb[0] is not None,
            period_emb[0] is not None,
            standard_emb[0] is not None,
        ]
    ):
        print("An embedding file could not be loaded. Exiting.")
        return

    # --- Combine Embeddings for Generation ---
    shared_embeddings = []
    for w_idx in range(3):
        for a_idx in range(3):
            for p_idx in range(3):
                mu = np.concatenate(
                    (wave_emb[0][w_idx], amp_emb[0][a_idx], period_emb[0][p_idx])
                )
                var = np.concatenate(
                    (wave_emb[1][w_idx], amp_emb[1][a_idx], period_emb[1][p_idx])
                )
                shared_embeddings.append((mu, var))

    standard_embeddings = [(standard_emb[0][i], standard_emb[1][i]) for i in range(27)]

    # --- Generate Time Series for all 27 combinations ---
    print("\nGenerating time series for 27 combinations...")
    generated_series = {"shared": [], "standard": []}
    for i in tqdm(range(27), desc="Generating Series"):
        # Generate with shared embeddings model
        mu, var = generate_time_series(
            shared_model, shared_embeddings[i], time_steps, look_back_len
        )
        generated_series["shared"].append((mu, var))

        # Generate with standard embeddings model
        mu, var = generate_time_series(
            standard_model, standard_embeddings[i], time_steps, look_back_len
        )
        generated_series["standard"].append((mu, var))

    # --- Plotting ---
    print("\nPlotting results...")
    wave_labels = ["Sine", "Square", "Triangular"]
    amp_labels = ["Amp 1x", "Amp 2x", "Amp 0.5x"]
    period_labels = ["Period 1x", "Period 2x", "Period 0.5x"]

    fig, axs = plt.subplots(9, 3, figsize=(18, 30), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        w_idx = i // 9
        a_idx = (i // 3) % 3
        p_idx = i % 3

        shared_mu, shared_var = generated_series["shared"][i]
        standard_mu, standard_var = generated_series["standard"][i]

        # Plot shared model generation
        ax.plot(shared_mu, label="Shared Model", color="royalblue")
        ax.fill_between(
            range(time_steps),
            shared_mu - np.sqrt(shared_var),
            shared_mu + np.sqrt(shared_var),
            color="royalblue",
            alpha=0.2,
        )

        # Plot standard model generation
        ax.plot(standard_mu, label="Standard Model", color="darkorange", linestyle="--")
        ax.fill_between(
            range(time_steps),
            standard_mu - np.sqrt(standard_var),
            standard_mu + np.sqrt(standard_var),
            color="darkorange",
            alpha=0.2,
        )

        title = f"{wave_labels[w_idx]}, {amp_labels[a_idx]}, {period_labels[p_idx]}"
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle("Generated Time Series: Shared vs. Standard Embeddings", fontsize=16)

    # Ensure output directory exists
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "generated_time_series_comparison.png")
    plt.savefig(save_path)
    plt.close()

    print(f"\nSuccessfully generated and saved comparison plot to {save_path}")


if __name__ == "__main__":
    main(500)

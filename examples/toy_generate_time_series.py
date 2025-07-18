import os
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


def init_model(params_dir: str, embedding_dim=10, num_features=1, look_back_len=24):
    # --- Define Model ---
    net = Sequential(
        LSTM(num_features + embedding_dim + look_back_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)

    # --- Load Pretrained Weights if available ---
    model_path = os.path.join(params_dir, "model.pth")
    if os.path.exists(model_path):
        net.load(model_path)
        print(f"Loaded pretrained model from {model_path}")
    else:
        print(f"No pretrained model found at {model_path}, initializing new model.")

    return net


def read_embeddings(file_path):
    """Reads wave embeddings from CSV files."""
    embeddings_mu = pd.read_csv(file_path + "embeddings_mu.csv", header=None).values
    embeddings_var = pd.read_csv(file_path + "embeddings_var.csv", header=None).values
    return embeddings_mu, embeddings_var


def main(time_steps=100):

    # --- Set Parameters ---
    model1 = init_model(
        "out/toy_shared_embeddings",
        embedding_dim=10,
        num_features=1,
        look_back_len=24,
    )

    model2 = init_model(
        "out/toy_embeddings",
        embedding_dim=10,
        num_features=1,
        look_back_len=24,
    )

    # --- Load Embeddings ---
    embedding_ = read_embeddings("out/toy_shared_embeddings/")
    embedding__ = read_embeddings("out/toy_embeddings/")

    # --- Rebuild Embedding Vectors for shared case ---
    pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
    embedding_shared = ()

    for i in range(4):
        mu = np.concatenate(
            (embedding_[0][pairs[i][0]], embedding_[0][pairs[i][1]]),
            axis=0,
        )
        var = np.concatenate(
            (embedding_[1][pairs[i][0]], embedding_[1][pairs[i][1]]),
            axis=0,
        )
        embedding_shared += ((mu, var),)

    # --- Rebuild Embedding Vectors for non-shared case ---
    embedding = ()

    for i in range(4):
        mu = embedding__[0][i]
        var = embedding__[1][i]
        embedding += ((mu, var),)

    # --- Generate Time Series ---
    # Generate time series for sin-amp1
    ts_mu1, ts_var1 = generate_time_series(
        model1, embedding_shared[0], time_steps=time_steps, look_back_len=24
    )
    ts_mu2, ts_var2 = generate_time_series(
        model2, embedding[0], time_steps=time_steps, look_back_len=24
    )

    # Generate time series for sin-amp1
    ts_mu3, ts_var3 = generate_time_series(
        model1, embedding_shared[1], time_steps=time_steps, look_back_len=24
    )
    ts_mu4, ts_var4 = generate_time_series(
        model2, embedding[1], time_steps=time_steps, look_back_len=24
    )

    # generate time series for square-amp1
    ts_mu5, ts_var5 = generate_time_series(
        model1, embedding_shared[2], time_steps=time_steps, look_back_len=24
    )
    ts_mu6, ts_var6 = generate_time_series(
        model2, embedding[2], time_steps=time_steps, look_back_len=24
    )
    # generate time series for square-amp2
    ts_mu7, ts_var7 = generate_time_series(
        model1, embedding_shared[3], time_steps=time_steps, look_back_len=24
    )
    ts_mu8, ts_var8 = generate_time_series(
        model2, embedding[3], time_steps=time_steps, look_back_len=24
    )

    # --- Plotting ---
    # plot each type of series in one figure per row
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    axs[0].plot(ts_mu1, label="sin-amp1 shared")
    axs[0].fill_between(
        np.arange(len(ts_mu1)), ts_mu1 - ts_var1, ts_mu1 + ts_var1, alpha=0.2
    )
    axs[0].plot(ts_mu2, label="sin-amp1")
    axs[0].fill_between(
        np.arange(len(ts_mu2)), ts_mu2 - ts_var2, ts_mu2 + ts_var2, alpha=0.2
    )
    axs[0].set_title("Sinusoidal Time Series with Amplitude 1")
    axs[0].legend()
    axs[1].plot(ts_mu3, label="sin-amp2 shared")
    axs[1].fill_between(
        np.arange(len(ts_mu3)), ts_mu3 - ts_var3, ts_mu3 + ts_var3, alpha=0.2
    )
    axs[1].plot(ts_mu4, label="sin-amp2")
    axs[1].fill_between(
        np.arange(len(ts_mu4)), ts_mu4 - ts_var4, ts_mu4 + ts_var4, alpha=0.2
    )
    axs[1].set_title("Sinusoidal Time Series with Amplitude 2")
    axs[1].legend()
    axs[2].plot(ts_mu5, label="square-amp1 shared")
    axs[2].fill_between(
        np.arange(len(ts_mu5)), ts_mu5 - ts_var5, ts_mu5 + ts_var5, alpha=0.2
    )
    axs[2].plot(ts_mu6, label="square-amp1")
    axs[2].fill_between(
        np.arange(len(ts_mu6)), ts_mu6 - ts_var6, ts_mu6 + ts_var6, alpha=0.2
    )
    axs[2].set_title("Square Wave Time Series with Amplitude 1")
    axs[2].legend()
    axs[3].plot(ts_mu7, label="square-amp2 shared")
    axs[3].fill_between(
        np.arange(len(ts_mu7)), ts_mu7 - ts_var7, ts_mu7 + ts_var7, alpha=0.2
    )
    axs[3].plot(ts_mu8, label="square-amp2")
    axs[3].fill_between(
        np.arange(len(ts_mu8)), ts_mu8 - ts_var8, ts_mu8 + ts_var8, alpha=0.2
    )
    axs[3].set_title("Square Wave Time Series with Amplitude 2")
    axs[3].legend()
    plt.tight_layout()
    plt.savefig("out/generated_time_series.png")
    plt.close()


if __name__ == "__main__":
    main(1000)

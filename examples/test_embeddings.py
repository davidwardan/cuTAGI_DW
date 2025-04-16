# import libraries
import os
import sys

import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import exponential_scheduler, manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential


from examples.data_loader import GlobalTimeSeriesDataloader
from examples.embedding_loader import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

from config.plot_config import setup_plotting

setup_plotting()


def main(
    num_epochs: int = 50,
    batch_size: int = 10,
    sigma_v: float = 0.1,
    embedding_dim: int = 10,
    seed: int = 0,
):
    """
    Run training for a time-series forecasting global model.
    Training is done on one complete time series at a time.
    """
    # Dataset
    nb_ts = 10  # number of time series
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 12
    output_seq_len = 1
    seq_stride = 1
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim),
        seed=seed,
    )  # initialize embeddings

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    for ts in pbar:
        # Load the data
        data_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/y.csv",
            date_time_file="data/toy_embedding/x.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            global_scale="deepar",
            embedding_dim=embedding_dim,
        )

        # Concatenate the time series
        if ts == 0:
            data = data_
        else:
            data = concat_ts_sample(data, data_)

    manual_seed(seed)

    # Model
    net = Sequential(
        LSTM((num_features + embedding_dim), 20, input_seq_len),
        LSTM(20, 20, input_seq_len),
        Linear(20 * input_seq_len, 1),
    )

    # net.to_device("cuda")
    net.input_state_update = True
    out_updater = OutputUpdater(net.device)

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    # Training loop
    mses = []
    for epoch in pbar:

        batch_iter = data.create_data_loader(
            batch_size=batch_size,
            shuffle=True,
        )

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        mse = []

        for x, y in batch_iter:
            # build a zero vector with the same length as x with 1 where the embedding is stored
            zero_vector = build_vector(x.shape[0], num_features, embedding_dim)
            # get the indices of the time series stored in the embedding
            x_ts_idx = reduce_vector(x, zero_vector, embedding_dim)
            # Select the first element along the embedding dimension as the index and keep as a NumPy array
            ts_indices = x_ts_idx[:, 0].astype(int)

            # replace the embedding section with the actual embedding
            x, x_var = input_embeddings(x, embeddings, num_features, embedding_dim)

            # only leave variance for the embedding section
            x_var = x_var * zero_vector

            # Feed forward
            m_pred, _ = net(x, x_var)

            # Update output layer
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Feed backward
            net.backward()
            net.step()

            # get the updated input state
            (mu_delta, var_delta) = net.get_input_states()

            # update the embedding
            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var

            # reduce the vector to get only embeddings
            x_update = reduce_vector(x_update, zero_vector, embedding_dim)
            var_update = reduce_vector(var_update, zero_vector, embedding_dim)

            # Update the embeddings in a vectorized manner (accumulation)
            np.add.at(embeddings.mu_embedding, ts_indices, x_update)
            np.add.at(embeddings.var_embedding, ts_indices, var_update)

            mse = metric.mse(m_pred, y)
            mses.append(mse)

            # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses)/len(mses):>7.4f}",
            refresh=True,
        )

    # apply PCA to embeddings
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(embeddings.mu_embedding)
    embeddings_pca = pca.transform(embeddings.mu_embedding)

    # Set figure size and plot settings
    plt.figure(figsize=(3, 3))

    # Plot the PCA results with first five embeddings in red and the rest in blue
    plt.scatter(embeddings_pca[:5, 0], embeddings_pca[:5, 1], color="r", label="Sin")
    plt.scatter(embeddings_pca[5:, 0], embeddings_pca[5:, 1], color="b", label="Cos")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc=(0.0, 1.01), edgecolor=None, frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()


def concat_ts_sample(data, data_add):
    """Concatenate two time series samples"""
    x_combined = np.concatenate(
        (data.dataset["value"][0], data_add.dataset["value"][0]), axis=0
    )
    y_combined = np.concatenate(
        (data.dataset["value"][1], data_add.dataset["value"][1]), axis=0
    )
    time_combined = np.concatenate(
        (data.dataset["date_time"], data_add.dataset["date_time"])
    )
    data.dataset["value"] = (x_combined, y_combined)
    data.dataset["date_time"] = time_combined
    return data


if __name__ == "__main__":
    main(
        num_epochs=100,
        batch_size=10,
        sigma_v=0.0,
        embedding_dim=20,
        seed=235,
    )

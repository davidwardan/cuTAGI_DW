# import libraries
import os
import sys

import fire
import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential
from pytagi import Normalizer as normalizer

from examples.data_loader import GlobalTimeSeriesDataloader
from examples.embedding import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)


def main(
    num_epochs: int = 20,
    batch_size: int = 10,
    sigma_v: float = 0.5,
):
    """
    Run training for a time-series forecasting global model.
    Training is done on one complete time series at a time.
    """
    # Dataset
    embedding_dim = 10  # dimension of the embedding
    nb_ts = 10  # for electricity 370 and 963 for traffic
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 12
    output_seq_len = 1
    seq_stride = 1
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim),
    )  # initialize embeddings

    # Model
    net = Sequential(
        LSTM((num_features + embedding_dim), 8, input_seq_len),
        LSTM(8, 8, input_seq_len),
        Linear(8 * input_seq_len, 1),
    )

    # net.to_device("cuda")
    net.input_state_update = True
    out_updater = OutputUpdater(net.device)

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    # Training loop
    mses = []
    for epoch in pbar:

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        for ts in ts_idx:
            train_dtl = GlobalTimeSeriesDataloader(
                x_file="data/toy_embedding/y.csv",
                date_time_file="data/toy_embedding/x.csv",
                output_col=output_col,
                input_seq_len=input_seq_len,
                output_seq_len=output_seq_len,
                num_features=num_features,
                stride=seq_stride,
                ts_idx=ts,
                embedding_dim=embedding_dim,
            )

            batch_iter = train_dtl.create_data_loader(batch_size)

            mse = []
            for x, y in batch_iter:
                # build a zero vector with the same length as x with 1 where the embedding is stored
                zero_vector = build_vector(x.shape[0], num_features, embedding_dim)
                # get the indices of the time series stored in the embedding
                x_ts_idx = reduce_vector(x, zero_vector, embedding_dim)
                # reduce the vector to get only the indices (needed if embedding_dim > 1)
                x_ts_idx = np.mean(x_ts_idx, axis=1).tolist()

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

                # store the updated embedding
                vec_loc = 0
                for ts_idx_ in x_ts_idx:
                    ts_idx_ = int(ts_idx_)
                    embeddings.update(ts_idx_, x_update[vec_loc], var_update[vec_loc])
                    vec_loc += 1

                mse = metric.mse(m_pred, y)
                mses.append(mse)

                # Progress bar
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses)/len(mses):>7.2f}",
                refresh=True,
            )

    # apply PCA to embeddings
    pca = PCA(n_components=2, random_state=42)
    pca.fit(embeddings.mu_embedding)
    embeddings_pca = pca.transform(embeddings.mu_embedding)

    plt.rcParams.update({"font.size": 32})
    plt.rc("text", usetex=True)
    plt.rc("mathtext", default="regular")


    # Define colormaps for blue and red shades
    n_points = 5
    blues = plt.cm.Blues(np.linspace(0.3, 1, n_points))  # Shades of blue
    reds = plt.cm.Reds(np.linspace(0.3, 1, n_points))    # Shades of red

    # Set figure size and plot settings
    plt.figure(figsize=(7, 7))

    # Plot the first 5 points with blue shades
    for i in range(n_points):
        plt.scatter(embeddings_pca[i, 0], embeddings_pca[i, 1], s=100, color=blues[i], alpha=0.7, label=f'Point {i+1}')

    # Plot the next 5 points with red shades
    for i in range(n_points, 2*n_points):
        plt.scatter(embeddings_pca[i, 0], embeddings_pca[i, 1], s=100, color=reds[i-n_points], alpha=0.7, label=f'Point {i+1}')

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.1, 1.1)

    # Add axis labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('PCA_equal.pdf', transparent=True)




if __name__ == "__main__":
    fire.Fire(main)

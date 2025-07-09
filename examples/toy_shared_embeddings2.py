import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from examples.embedding_loader import (
    TimeSeriesEmbeddings,
    build_vector,
)
from examples.data_loader import GlobalTimeSeriesDataloader
from pytagi import exponential_scheduler, manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential


def concat_ts_sample(data, data_add):
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


def main(num_epochs=40, batch_size=1, sigma_v=2.0):

    # Dataset
    nb_ts = 27
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1

    # --- Embedding configuration -------------------------------------------
    embedding_dim_each = 5  # dimensions for each category (wave, amp, period)
    embedding_dim = embedding_dim_each * 3  # total embedding size

    num_wave_types = 3  # sine | square | triangular
    num_amp_types = 3  # 1× | 2× | 0.5×
    num_period_types = 3  # 1× | 2× | 0.5×

    wave_embeddings = TimeSeriesEmbeddings(
        (num_wave_types, embedding_dim_each), encoding_type="normal"
    )
    amp_embeddings = TimeSeriesEmbeddings(
        (num_amp_types, embedding_dim_each), encoding_type="normal"
    )
    period_embeddings = TimeSeriesEmbeddings(
        (num_period_types, embedding_dim_each), encoding_type="normal"
    )

    # set seed for model initialization
    manual_seed(42)

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    covar_means = [0.0] * nb_ts
    covar_stds = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/time_series_values.csv",
            date_time_file="data/toy_embedding/time_series_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )

        # store covariate means and stds
        # covar_means[ts] = train_dtl_.covariate_means
        # covar_stds[ts] = train_dtl_.covariate_stds

        if ts == 0:
            train_dtl = train_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)

    net = Sequential(
        LSTM(num_features + embedding_dim + input_seq_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )

    # net.to_device("cuda")
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # input state update
    net.input_state_update = True

    # Create output directory
    out_dir = "out/toy_shared_embeddings"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        for x, y in batch_iter:
            # ---------------------------------------------------------------
            # Decode placeholder index into (wave, amplitude, period) indices
            # ---------------------------------------------------------------
            ts_idx_placeholder = int(np.mean(x[-embedding_dim:]).tolist())

            wave_idx = ts_idx_placeholder // 9
            amp_idx = (ts_idx_placeholder // 3) % 3
            period_idx = ts_idx_placeholder % 3

            # Retrieve embedding parameters
            wave_mu, wave_var = wave_embeddings.get_embedding(wave_idx)
            amp_mu, amp_var = amp_embeddings.get_embedding(amp_idx)
            per_mu, per_var = period_embeddings.get_embedding(period_idx)

            embed_mu = np.concatenate((wave_mu, amp_mu, per_mu), axis=0)
            embed_var = np.concatenate((wave_var, amp_var, per_var), axis=0)

            # Replace embed slots in x and x_var with embedding means and variances
            x_var = np.zeros_like(x)
            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            x = np.float32(x)
            x_var = np.float32(x_var)

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

            # Update embeddings based on posterior state updates
            embed_mu_update = x_update[-embedding_dim:]
            embed_var_update = var_update[-embedding_dim:]

            # Split posterior updates by category
            wave_mu_upd = embed_mu_update[:embedding_dim_each]
            amp_mu_upd = embed_mu_update[embedding_dim_each : 2 * embedding_dim_each]
            per_mu_upd = embed_mu_update[2 * embedding_dim_each :]

            wave_var_upd = embed_var_update[:embedding_dim_each]
            amp_var_upd = embed_var_update[embedding_dim_each : 2 * embedding_dim_each]
            per_var_upd = embed_var_update[2 * embedding_dim_each :]

            # Apply updates
            wave_embeddings.update(wave_idx, wave_mu_upd, wave_var_upd)
            amp_embeddings.update(amp_idx, amp_mu_upd, amp_var_upd)
            period_embeddings.update(period_idx, per_mu_upd, per_var_upd)

    # save embeddings
    save_dir = "out/toy_shared_embeddings/"
    sub_dirs = ["wave_embeddings", "amp_embeddings", "period_embeddings"]

    for sub_dir in sub_dirs:
        full_path = os.path.join(save_dir, sub_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    wave_embeddings.save(os.path.join(save_dir, "wave_embeddings"))
    amp_embeddings.save(os.path.join(save_dir, "amp_embeddings"))
    period_embeddings.save(os.path.join(save_dir, "period_embeddings"))

    # ----------------------------------------------------------------------
    # Visualise learned embeddings for each time series (PCA 2‑D projection)
    # ----------------------------------------------------------------------
    from sklearn.decomposition import PCA

    # Build a full embedding vector for each time‑series ID (0‒26)
    time_series_embeddings = []
    ts_labels = []
    ts_colors = []
    # Human‑readable category names
    wave_names = ["Sin", "Square", "Triangular"]
    amp_names = ["Amp1x", "Amp1.5x", "Amp0.5x"]
    per_names = ["Period1x", "Period1.5x", "Period0.5x"]
    color_map = {
        "Sin": "red",
        "Square": "blue",
        "Triangular": "green",
    }
    for ts in range(nb_ts):
        wave_idx = ts // 9
        amp_idx = (ts // 3) % 3
        period_idx = ts % 3

        wave_mu, _ = wave_embeddings.get_embedding(wave_idx)
        amp_mu, _ = amp_embeddings.get_embedding(amp_idx)
        per_mu, _ = period_embeddings.get_embedding(period_idx)

        ts_embed = np.concatenate((wave_mu, amp_mu, per_mu), axis=0)
        time_series_embeddings.append(ts_embed)
        descriptive_label = (
            f"{wave_names[wave_idx]}_{amp_names[amp_idx]}_{per_names[period_idx]}"
        )
        ts_labels.append(descriptive_label)
        ts_colors.append(color_map[wave_names[wave_idx]])

    embeddings_matrix = np.array(time_series_embeddings)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_matrix)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_result[:, 0], pca_result[:, 1], s=100, c=ts_colors, alpha=0.8
    )
    for i, lbl in enumerate(ts_labels):
        plt.annotate(lbl, (pca_result[i, 0], pca_result[i, 1]), fontsize=10)
    plt.title("PCA of Learned Embeddings per Time Series")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()

    # Add legend
    legend_labels = list(color_map.keys())
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[label],
            markersize=10,
        )
        for label in legend_labels
    ]
    plt.legend(legend_handles, legend_labels, title="Wave Types", loc="best")

    plt.show()


if __name__ == "__main__":
    main()

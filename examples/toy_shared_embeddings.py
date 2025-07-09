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


def main(num_epochs=20, batch_size=1, sigma_v=1.0):

    # Dataset
    nb_ts = 4
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1

    embedding_dim = 10
    embedding_dim_shared = embedding_dim // 2
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim_shared), encoding_type="normal"
    )  # initialize embeddings

    # set seed for model initialization
    manual_seed(235)

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    covar_means = [0.0] * nb_ts
    covar_stds = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/train_values.csv",
            date_time_file="data/toy_embedding/train_datetimes.csv",
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
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        for x, y in batch_iter:
            # get the indices of the time series stored in the embedding
            x_ts_idx = x[-embedding_dim:]
            # reduce the vector to get only the indices (needed if embedding_dim > 1)
            x_ts_idx = np.mean(x_ts_idx).tolist()

            # Build embeddings per sample and concatenate pairs
            ts = int(x_ts_idx)
            # Determine embedding pair based on ts index
            if ts == 0:
                pair = (0, 2)
            elif ts == 1:
                pair = (0, 3)
            elif ts == 2:
                pair = (1, 2)
            elif ts == 3:
                pair = (1, 3)
            else:
                raise ValueError(f"Unsupported ts_id {ts}")
            # Retrieve and concatenate embedding parameters
            mu1, var1 = embeddings.get_embedding(pair[0])
            mu2, var2 = embeddings.get_embedding(pair[1])
            embed_mu = np.concatenate((mu1, mu2), axis=0)
            embed_var = np.concatenate((var1, var2), axis=0)

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

            # use pairs previously defined to update the embeddings
            embeddings.update(
                pair[0],
                embed_mu_update[:embedding_dim_shared],
                embed_var_update[:embedding_dim_shared],
            )
            embeddings.update(
                pair[1],
                embed_mu_update[embedding_dim_shared:],
                embed_var_update[embedding_dim_shared:],
            )

    # plot each of the embeddings
    sin_embed_mu, sin_embed_var = embeddings.get_embedding(0)
    square_embed_mu, square_embed_var = embeddings.get_embedding(1)
    amp1_embed_mu, amp1_embed_var = embeddings.get_embedding(2)
    amp2_embed_mu, amp2_embed_var = embeddings.get_embedding(3)

    # check the cosine similarity between the embeddings
    from scipy.spatial.distance import cosine

    cos_sim_sin_square = 1 - cosine(sin_embed_mu, square_embed_mu)
    cos_sim_amp1_amp2 = 1 - cosine(amp1_embed_mu, amp2_embed_mu)
    cos_sim_sin_amp1 = 1 - cosine(sin_embed_mu, amp1_embed_mu)
    cos_sim_square_amp2 = 1 - cosine(square_embed_mu, amp2_embed_mu)

    print(f"Cosine Similarity between Sin and Square: {cos_sim_sin_square}")
    print(f"Cosine Similarity between Amp1 and Amp2: {cos_sim_amp1_amp2}")
    print(f"Cosine Similarity between Sin and Amp1: {cos_sim_sin_amp1}")
    print(f"Cosine Similarity between Square and Amp2: {cos_sim_square_amp2}")

    # Use PCA and then plot the embeddings
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    embeddings_matrix = np.array(
        [
            sin_embed_mu,
            square_embed_mu,
            amp1_embed_mu,
            amp2_embed_mu,
        ]
    )
    pca_result = pca.fit_transform(embeddings_matrix)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100)
    for i, txt in enumerate(["Sin", "Square", "Amp1", "Amp2"]):
        plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]), fontsize=12)
    plt.title("PCA of Time Series Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()

    # Plot PCA of pair embeddings
    pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
    labels = ["Sin-Amp1", "Sin-Amp2", "Square-Amp1", "Square-Amp2"]
    pair_embeds = []
    for i, j in pairs:
        mu1, _ = embeddings.get_embedding(i)
        mu2, _ = embeddings.get_embedding(j)
        pair_embeds.append(np.concatenate((mu1, mu2), axis=0))

    # Perform PCA on pair embeddings
    pca_pairs = PCA(n_components=2)
    pair_pca = pca_pairs.fit_transform(np.array(pair_embeds))

    plt.figure(figsize=(8, 6))
    plt.scatter(pair_pca[:, 0], pair_pca[:, 1], s=100)
    for idx, label in enumerate(labels):
        plt.annotate(label, (pair_pca[idx, 0], pair_pca[idx, 1]), fontsize=12)
    plt.title("PCA of Pair Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()

    m_preds = {i: [] for i in range(nb_ts)}
    var_preds = {i: [] for i in range(nb_ts)}
    m_true = {i: [] for i in range(nb_ts)}

    # Test the model
    pbar = tqdm(ts_idx, desc="Testing Progress")
    for ts in pbar:

        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_values.csv",
            date_time_file="data/toy_embedding/test_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            # covariate_means=covar_means[ts],
            # covariate_stds=covar_stds[ts],
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )

        # Extract sequences and datetimes
        inputs, outputs = test_dtl.dataset["value"]
        datetimes = test_dtl.dataset["date_time"][input_seq_len:]
        # Initial input sequence (exclude the embedding portion)
        seq_window = inputs[0][:input_seq_len].copy()  # shape: (input_seq_len,)

        for i in range(len(outputs)):
            # Assemble feature vector: [current input sequence | embedding]
            x = np.concatenate((seq_window, np.zeros(embedding_dim, dtype=np.float32)))
            x_var = np.zeros_like(x)

            # Build embedding vector for this series
            if ts == 0:
                pair = (0, 2)
            elif ts == 1:
                pair = (0, 3)
            elif ts == 2:
                pair = (1, 2)
            else:
                pair = (1, 3)
            mu1, var1 = embeddings.get_embedding(pair[0])
            mu2, var2 = embeddings.get_embedding(pair[1])
            embed_mu = np.concatenate((mu1, mu2), axis=0)
            embed_var = np.concatenate((var1, var2), axis=0)

            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            # Forward pass
            m_pred, var_pred = net(np.float32(x), np.float32(x_var))
            mean_pred = float(np.squeeze(m_pred))
            var_pred_scalar = float(np.squeeze(var_pred))

            # Log predictions and ground truth
            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            # Recursive update: shift window and append prediction
            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    # plot the predictions
    # get true values
    true_dates = test_dtl.dataset["date_time"][24:]

    # Plot each time series in its own subplot (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, ts in enumerate(ts_idx):
        ax = axs.flat[idx]
        ax.plot(
            true_dates,
            m_true[ts],
            color="red",
        )
        ax.plot(
            true_dates,
            m_preds[ts],
            color="blue",
        )
        ax.fill_between(
            true_dates,
            np.array(m_preds[ts]) - np.sqrt(np.array(var_preds[ts])),
            np.array(m_preds[ts]) + np.sqrt(np.array(var_preds[ts])),
            color="blue",
            alpha=0.3,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

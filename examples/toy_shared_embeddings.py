import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

# Assuming these are in a directory named 'examples'
# and 'pytagi' is installed
from examples.embedding_loader import (
    TimeSeriesEmbeddings,
)
from examples.data_loader import GlobalTimeSeriesDataloader
from pytagi import exponential_scheduler, manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential


def concat_ts_sample(data, data_add):
    """Concatenates two time series datasets."""
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


def shared_embeddings_run(num_epochs=20, batch_size=1, sigma_v=1.0, seed=235):
    """
    Runs the time series forecasting experiment with shared embeddings.
    """
    print("Running with shared embeddings...")
    # --- Dataset and Model Configuration ---
    nb_ts = 4
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1
    embedding_dim = 10
    embedding_dim_shared = embedding_dim // 2

    # --- Initialize Embeddings ---
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim_shared),
        # encoding_type="normal",
        encoding_type="sphere",
        seed=seed,
    )

    manual_seed(seed)

    # --- Load Data ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    train_dtl = None
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
        if train_dtl is None:
            train_dtl = train_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)

    # --- Define Model ---
    net = Sequential(
        LSTM(num_features + embedding_dim + input_seq_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # --- Output Directory ---
    out_dir = "out/toy_shared_embeddings"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # --- Training Loop ---
    mse_history = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)
        epoch_mse = 0
        num_samples = 0

        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        for x, y in batch_iter:
            x_ts_idx = np.mean(x[-embedding_dim:]).tolist()
            ts = int(x_ts_idx)

            pair_map = {0: (0, 2), 1: (0, 3), 2: (1, 2), 3: (1, 3)}
            if ts not in pair_map:
                raise ValueError(f"Unsupported ts_id {ts}")
            pair = pair_map[ts]

            mu1, var1 = embeddings.get_embedding(pair[0])
            mu2, var2 = embeddings.get_embedding(pair[1])
            embed_mu = np.concatenate((mu1, mu2), axis=0)
            embed_var = np.concatenate((var1, var2), axis=0)

            x_var = np.zeros_like(x)
            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            m_pred, _ = net(np.float32(x), np.float32(x_var))
            epoch_mse += np.sum((y - m_pred.flatten()) ** 2)
            num_samples += len(y)

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            net.backward()
            net.step()

            (mu_delta, var_delta) = net.get_input_states()
            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var

            embed_mu_update = x_update[-embedding_dim:]
            embed_var_update = var_update[-embedding_dim:]

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

        mse_history.append(epoch_mse / num_samples)
        pbar.set_postfix({"MSE": mse_history[-1]})

    # --- Save Model ---
    model_path = os.path.join(out_dir, "model.pth")
    net.save(model_path)

    # --- Save Embeddings ---
    embeddings.save(out_dir)

    # --- Plot and Save Results ---
    # Training MSE
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mse_history, label="Training MSE")
    plt.title("Shared Embeddings: Training MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_mse.png"))
    plt.close()

    # Cosine Similarity
    sin_embed_mu, _ = embeddings.get_embedding(0)
    square_embed_mu, _ = embeddings.get_embedding(1)
    amp1_embed_mu, _ = embeddings.get_embedding(2)
    amp2_embed_mu, _ = embeddings.get_embedding(3)

    # Cosine Similarities
    embed_mus = [
        np.concatenate((sin_embed_mu, amp1_embed_mu), axis=0),
        np.concatenate((sin_embed_mu, amp2_embed_mu), axis=0),
        np.concatenate((square_embed_mu, amp1_embed_mu), axis=0),
        np.concatenate((square_embed_mu, amp2_embed_mu), axis=0),
    ]

    print(f"Cosine Similarity 1-2: {1 - cosine(embed_mus[0], embed_mus[1])}")
    print(f"Cosine Similarity 1-3: {1 - cosine(embed_mus[0], embed_mus[2])}")
    print(f"Cosine Similarity 2-4: {1 - cosine(embed_mus[1], embed_mus[3])}")
    print(f"Cosine Similarity 3-4: {1 - cosine(embed_mus[2], embed_mus[3])}")

    # PCA of Embeddings
    pca = PCA(n_components=2)
    embeddings_matrix = np.array(
        [sin_embed_mu, square_embed_mu, amp1_embed_mu, amp2_embed_mu]
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
    plt.savefig(os.path.join(out_dir, "pca_embeddings.png"))
    plt.close()

    # PCA of Pair Embeddings
    pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
    labels = ["Sin-Amp1", "Sin-Amp2", "Square-Amp1", "Square-Amp2"]
    pair_embeds = [
        np.concatenate(embeddings.get_embedding(i) + embeddings.get_embedding(j))
        for i, j in pairs
    ]
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
    plt.savefig(os.path.join(out_dir, "pca_pair_embeddings.png"))
    plt.close()

    # --- Testing and Prediction Plotting ---
    m_preds = {i: [] for i in range(nb_ts)}
    var_preds = {i: [] for i in range(nb_ts)}
    m_true = {i: [] for i in range(nb_ts)}

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
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )
        inputs, outputs = test_dtl.dataset["value"]
        seq_window = inputs[0][:input_seq_len].copy()

        for i in range(len(outputs)):
            x = np.concatenate((seq_window, np.zeros(embedding_dim, dtype=np.float32)))
            x_var = np.zeros_like(x)

            pair = pair_map[ts]
            mu1, var1 = embeddings.get_embedding(pair[0])
            mu2, var2 = embeddings.get_embedding(pair[1])
            embed_mu = np.concatenate((mu1, mu2), axis=0)
            embed_var = np.concatenate((var1, var2), axis=0)
            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            m_pred, var_pred = net(np.float32(x), np.float32(x_var))
            mean_pred = float(np.squeeze(m_pred))
            var_pred_scalar = float(np.squeeze(var_pred))

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    true_dates = test_dtl.dataset["date_time"][input_seq_len:]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, ts in enumerate(ts_idx):
        ax = axs.flat[idx]
        ax.plot(true_dates, m_true[ts], color="red", label="True")
        ax.plot(true_dates, m_preds[ts], color="blue", label="Predicted")
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
    plt.savefig(os.path.join(out_dir, "predictions.png"))
    plt.close()


def embeddings_run(num_epochs=20, batch_size=1, sigma_v=1.0, seed=235):
    """
    Runs the time series forecasting experiment with standard embeddings.
    """
    print("\nRunning with embeddings...")
    # --- Configuration ---
    nb_ts = 4
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    seq_stride = 1
    embedding_dim = 10

    # --- Initialization ---
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim),
        # encoding_type="normal",
        encoding_type="sphere",
        seed=seed,
    )
    manual_seed(seed)

    # --- Load Data ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    train_dtl = None
    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/train_values.csv",
            date_time_file="data/toy_embedding/train_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=1,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )
        if train_dtl is None:
            train_dtl = train_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)

    # --- Model Definition ---
    net = Sequential(
        LSTM(num_features + embedding_dim + input_seq_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # --- Output Directory ---
    out_dir = "out/toy_embeddings"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # --- Training Loop ---
    mse_history = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)
        epoch_mse = 0
        num_samples = 0
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        for x, y in batch_iter:
            ts = int(np.mean(x[-embedding_dim:]).tolist())
            embed_mu, embed_var = embeddings.get_embedding(ts)
            x_var = np.zeros_like(x)
            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            m_pred, _ = net(np.float32(x), np.float32(x_var))
            epoch_mse += np.sum((y - m_pred.flatten()) ** 2)
            num_samples += len(y)

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            net.backward()
            net.step()

            (mu_delta, var_delta) = net.get_input_states()
            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var
            embed_mu_update = x_update[-embedding_dim:]
            embed_var_update = var_update[-embedding_dim:]
            embeddings.update(ts, embed_mu_update, embed_var_update)

        mse_history.append(epoch_mse / num_samples)
        pbar.set_postfix({"MSE": mse_history[-1]})

    # --- Save Model ---
    model_path = os.path.join(out_dir, "model.pth")
    net.save(model_path)

    # --- Save Embeddings ---
    embeddings.save(out_dir)

    # --- Plotting and Analysis ---
    # MSE Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mse_history, label="Training MSE")
    plt.title("Embeddings: Training MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_mse.png"))
    plt.close()

    # Cosine Similarities
    embed_mus = [embeddings.get_embedding(i)[0] for i in range(nb_ts)]
    print(f"Cosine Similarity 1-2: {1 - cosine(embed_mus[0], embed_mus[1])}")
    print(f"Cosine Similarity 1-3: {1 - cosine(embed_mus[0], embed_mus[2])}")
    print(f"Cosine Similarity 2-4: {1 - cosine(embed_mus[1], embed_mus[3])}")
    print(f"Cosine Similarity 3-4: {1 - cosine(embed_mus[2], embed_mus[3])}")

    # PCA Plot
    pca = PCA(n_components=2)
    embeddings_matrix = np.array(embed_mus)
    pca_result = pca.fit_transform(embeddings_matrix)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100)
    for i, txt in enumerate(["Sin-Amp1", "Sin-Amp2", "Square-Amp1", "Square-Amp2"]):
        plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]), fontsize=12)
    plt.title("PCA of Time Series Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "pca_embeddings.png"))
    plt.close()

    # --- Testing ---
    m_preds = {i: [] for i in range(nb_ts)}
    var_preds = {i: [] for i in range(nb_ts)}
    m_true = {i: [] for i in range(nb_ts)}
    pbar = tqdm(ts_idx, desc="Testing Progress")
    for ts in pbar:
        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_values.csv",
            date_time_file="data/toy_embedding/test_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=1,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )
        inputs, outputs = test_dtl.dataset["value"]
        seq_window = inputs[0][:input_seq_len].copy()

        for i in range(len(outputs)):
            x = np.concatenate((seq_window, np.zeros(embedding_dim, dtype=np.float32)))
            x_var = np.zeros_like(x)
            embed_mu, embed_var = embeddings.get_embedding(ts)
            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            m_pred, var_pred = net(np.float32(x), np.float32(x_var))
            mean_pred = float(np.squeeze(m_pred))
            var_pred_scalar = float(np.squeeze(var_pred))

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    true_dates = test_dtl.dataset["date_time"][input_seq_len:]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, ts in enumerate(ts_idx):
        ax = axs.flat[idx]
        ax.plot(true_dates, m_true[ts], color="red", label="True")
        ax.plot(true_dates, m_preds[ts], color="blue", label="Predicted")
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
    plt.savefig(os.path.join(out_dir, "predictions.png"))
    plt.close()


def no_embeddings_run(num_epochs=20, batch_size=1, sigma_v=1.0, seed=235):
    """
    Runs the time series forecasting experiment without embeddings.
    """
    print("\nRunning without embeddings...")
    # --- Configuration ---
    nb_ts = 4
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    seq_stride = 1

    manual_seed(seed)

    # --- Load Data ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    train_dtl = None
    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/train_values.csv",
            date_time_file="data/toy_embedding/train_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=1,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
        )
        if train_dtl is None:
            train_dtl = train_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)

    # --- Model Definition ---
    net = Sequential(
        LSTM(num_features + input_seq_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # --- Output Directory ---
    out_dir = "out/toy_no_embeddings"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # --- Training Loop ---
    mse_history = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)
        epoch_mse = 0
        num_samples = 0
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        for x, y in batch_iter:
            m_pred, _ = net(x)
            epoch_mse += np.sum((y - m_pred.flatten()) ** 2)
            num_samples += len(y)

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()

        mse_history.append(epoch_mse / num_samples)
        pbar.set_postfix({"MSE": mse_history[-1]})

    # --- Save Model ---
    model_path = os.path.join(out_dir, "model.pth")
    net.save(model_path)

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mse_history, label="Training MSE")
    plt.title("No Embeddings: Training MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_mse.png"))
    plt.close()

    # --- Testing ---
    m_preds = {i: [] for i in range(nb_ts)}
    var_preds = {i: [] for i in range(nb_ts)}
    m_true = {i: [] for i in range(nb_ts)}
    pbar = tqdm(ts_idx, desc="Testing Progress")
    for ts in pbar:
        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_values.csv",
            date_time_file="data/toy_embedding/test_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=1,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
        )
        inputs, outputs = test_dtl.dataset["value"]
        seq_window = inputs[0][:input_seq_len].copy()

        for i in range(len(outputs)):

            m_pred, var_pred = net(np.float32(seq_window))
            mean_pred = float(np.squeeze(m_pred))
            var_pred_scalar = float(np.squeeze(var_pred))

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    true_dates = test_dtl.dataset["date_time"][input_seq_len:]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, ts in enumerate(ts_idx):
        ax = axs.flat[idx]
        ax.plot(true_dates, m_true[ts], color="red", label="True")
        ax.plot(true_dates, m_preds[ts], color="blue", label="Predicted")
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
    plt.savefig(os.path.join(out_dir, "predictions.png"))
    plt.close()


def main(num_epochs=20, batch_size=1, sigma_v=1.0, seed=1):
    """
    Main function to run all experiments.
    """
    shared_embeddings_run(
        num_epochs=num_epochs, batch_size=batch_size, sigma_v=sigma_v, seed=seed
    )
    embeddings_run(
        num_epochs=num_epochs, batch_size=batch_size, sigma_v=sigma_v, seed=seed
    )
    no_embeddings_run(
        num_epochs=num_epochs, batch_size=batch_size, sigma_v=sigma_v, seed=seed
    )


if __name__ == "__main__":
    # Make sure you have the data files in a 'data/toy_embedding' directory
    # and the 'examples' module is accessible.
    if not os.path.exists("data/toy_embedding"):
        print("Please ensure the data files are in 'data/toy_embedding'")
    else:
        main()

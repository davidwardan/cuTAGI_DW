import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

# Assuming custom modules are in the python path
from examples.embedding_loader import TimeSeriesEmbeddings
from examples.data_loader import GlobalTimeSeriesDataloader
from pytagi import exponential_scheduler, manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential


def reset_lstm_states(model):
    """Resets the LSTM states of the model."""
    lstm_states = model.lstm_net.get_lstm_states()
    for key in lstm_states:
        old_tuple = lstm_states[key]
        new_tuple = tuple(np.zeros_like(np.array(v)).tolist() for v in old_tuple)
        lstm_states[key] = new_tuple
    model.lstm_net.set_lstm_states(lstm_states)


def concat_ts_sample(data, data_add):
    """Helper function to concatenate datasets."""
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


def plot_predictions(out_dir, nb_ts, true_dates, m_true, m_preds, var_preds):
    """Plots and saves the prediction results."""
    os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists
    n_cols = 3
    n_rows = math.ceil(nb_ts / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True, sharey=True
    )
    axs = axs.flatten()

    for idx in range(nb_ts):
        ax = axs[idx]
        ax.plot(true_dates, m_true[idx], color="red", label="True")
        ax.plot(true_dates, m_preds[idx], color="blue", label="Pred")
        ax.fill_between(
            true_dates,
            np.array(m_preds[idx]) - np.sqrt(np.array(var_preds[idx])),
            np.array(m_preds[idx]) + np.sqrt(np.array(var_preds[idx])),
            color="blue",
            alpha=0.3,
        )
        ax.set_title(f"TS {idx} (w{idx//9}, a{(idx//3)%3}, p{idx%3})", fontsize=9)
        ax.legend(fontsize=7)

    for ax in axs[nb_ts:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predictions.png"))
    plt.close()


def shared_embeddings_run(num_epochs=20, batch_size=1, sigma_v=2.0, seed=42):
    """Trains a model with shared, structured embeddings (wave, amp, period)."""
    print("--- Running Experiment with Shared Embeddings ---")
    out_dir = "out/toy2_shared_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    # --- Configuration ---
    nb_ts = 27
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    seq_stride = 1

    embedding_dim_each = 5
    embedding_dim = embedding_dim_each * 3

    num_wave_types, num_amp_types, num_period_types = 3, 3, 3

    wave_embeddings = TimeSeriesEmbeddings(
        (num_wave_types, embedding_dim_each), encoding_type="normal", seed=seed
    )
    amp_embeddings = TimeSeriesEmbeddings(
        (num_amp_types, embedding_dim_each), encoding_type="normal", seed=seed
    )
    period_embeddings = TimeSeriesEmbeddings(
        (num_period_types, embedding_dim_each), encoding_type="normal", seed=seed
    )

    wave_embeddings.save(os.path.join(out_dir, "wave_embeddings"))
    amp_embeddings.save(os.path.join(out_dir, "amp_embeddings"))
    period_embeddings.save(os.path.join(out_dir, "period_embeddings"))

    sys.exit(
        "Exiting early for debugging purposes. Remove this line to run the full experiment."
    )

    manual_seed(seed)

    # --- Data Loading ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    train_dtl = None
    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/train_triplets_values.csv",
            date_time_file="data/toy_embedding/train_triplets_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=1,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )
        train_dtl = train_dtl_ if ts == 0 else concat_ts_sample(train_dtl, train_dtl_)

    # --- Model Definition ---
    net = Sequential(
        LSTM(num_features + embedding_dim + input_seq_len - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # --- Training Loop ---
    mse_history = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        epoch_mse = 0
        num_samples = 0

        for x, y in batch_iter:
            ts_idx_placeholder = int(np.mean(x[-embedding_dim:]).tolist())
            wave_idx, amp_idx, period_idx = (
                ts_idx_placeholder // 9,
                (ts_idx_placeholder // 3) % 3,
                ts_idx_placeholder % 3,
            )

            wave_mu, wave_var = wave_embeddings.get_embedding(wave_idx)
            amp_mu, amp_var = amp_embeddings.get_embedding(amp_idx)
            per_mu, per_var = period_embeddings.get_embedding(period_idx)

            embed_mu = np.concatenate((wave_mu, amp_mu, per_mu), axis=0)
            embed_var = np.concatenate((wave_var, amp_var, per_var), axis=0)

            x_var = np.zeros_like(x)
            x[-embedding_dim:], x_var[-embedding_dim:] = embed_mu, embed_var

            m_pred, _ = net(np.float32(x), np.float32(x_var))
            epoch_mse += np.sum((y - m_pred.flatten()) ** 2)
            num_samples += len(y)

            out_updater.update(net.output_z_buffer, y, var_y, net.input_delta_z_buffer)
            net.backward()
            net.step()

            (mu_delta, var_delta) = net.get_input_states()
            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var
            embed_mu_update, embed_var_update = (
                x_update[-embedding_dim:],
                var_update[-embedding_dim:],
            )

            wave_mu_upd, amp_mu_upd, per_mu_upd = np.split(embed_mu_update, 3)
            wave_var_upd, amp_var_upd, per_var_upd = np.split(embed_var_update, 3)

            wave_embeddings.update(wave_idx, wave_mu_upd, wave_var_upd)
            amp_embeddings.update(amp_idx, amp_mu_upd, amp_var_upd)
            period_embeddings.update(period_idx, per_mu_upd, per_var_upd)

        mse_history.append(epoch_mse / num_samples)
        pbar.set_postfix({"MSE": mse_history[-1]})

        reset_lstm_states(net)

    # --- Save Artifacts ---
    net.save(os.path.join(out_dir, "model.path"))

    # Ensure embedding directories exist before saving
    for emb_name in ["wave_embeddings", "amp_embeddings", "period_embeddings"]:
        os.makedirs(os.path.join(out_dir, emb_name), exist_ok=True)

    wave_embeddings.save(os.path.join(out_dir, "wave_embeddings"))
    amp_embeddings.save(os.path.join(out_dir, "amp_embeddings"))
    period_embeddings.save(os.path.join(out_dir, "period_embeddings"))

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mse_history, label="Training MSE")
    plt.title("Shared Embeddings: Training MSE")
    plt.xlabel("Epoch"), plt.ylabel("MSE"), plt.legend(), plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_mse.png"))
    plt.close()

    for embeddings, name, labels in [
        (wave_embeddings, "Wave Type", ["Sine", "Square", "Triangular"]),
        (amp_embeddings, "Amplitude", ["1x", "2x", "0.5x"]),
        (period_embeddings, "Period", ["1x", "2x", "0.5x"]),
    ]:
        pca = PCA(n_components=2)
        embed_mus = np.array(
            [embeddings.get_embedding(i)[0] for i in range(len(labels))]
        )
        pca_result = pca.fit_transform(embed_mus)
        plt.figure(figsize=(7, 5))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], s=150)
        for i, txt in enumerate(labels):
            plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]), fontsize=12)
        plt.title(f"PCA of {name} Embeddings"), plt.xlabel(
            "PCA Component 1"
        ), plt.ylabel("PCA Component 2"), plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"pca_{name.lower().replace(' ', '_')}.png"))
        plt.close()

    # --- Testing ---
    m_preds, var_preds, m_true = ({i: [] for i in range(nb_ts)} for _ in range(3))
    pbar = tqdm(ts_idx, desc="Testing Progress")
    for ts in pbar:
        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_triplets_values.csv",
            date_time_file="data/toy_embedding/test_triplets_datetimes.csv",
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
        datetimes = test_dtl.dataset["date_time"][input_seq_len:]
        seq_window = inputs[0, :input_seq_len].copy()

        wave_idx, amp_idx, period_idx = ts // 9, (ts // 3) % 3, ts % 3
        wave_mu, _ = wave_embeddings.get_embedding(wave_idx)
        amp_mu, _ = amp_embeddings.get_embedding(amp_idx)
        per_mu, _ = period_embeddings.get_embedding(period_idx)
        embed_mu = np.concatenate((wave_mu, amp_mu, per_mu))
        embed_var = np.zeros_like(embed_mu)

        for i in range(len(outputs)):
            x = np.concatenate((seq_window, embed_mu))
            x_var = np.concatenate((np.zeros_like(seq_window), embed_var))
            m_pred, var_pred = net(np.float32(x), np.float32(x_var))
            mean_pred, var_pred_scalar = float(np.squeeze(m_pred)), float(
                np.squeeze(var_pred)
            )

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    plot_predictions(out_dir, nb_ts, datetimes, m_true, m_preds, var_preds)


def embeddings_run(num_epochs=20, batch_size=1, sigma_v=2.0, seed=42):
    """Trains a model with a unique embedding for each of the 27 time series."""
    print("\n--- Running Experiment with Standard Embeddings ---")
    out_dir = "out/toy2_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    # --- Configuration ---
    nb_ts = 27
    ts_idx = np.arange(0, nb_ts)
    embedding_dim = 15
    manual_seed(seed)

    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim), encoding_type="normal", seed=seed
    )

    # --- Data Loading ---
    train_dtl = None
    for ts in tqdm(ts_idx, "Loading Data"):
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/train_triplets_values.csv",
            date_time_file="data/toy_embedding/train_triplets_datetimes.csv",
            output_col=[0],
            input_seq_len=24,
            output_seq_len=1,
            num_features=1,
            stride=1,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )
        train_dtl = train_dtl_ if ts == 0 else concat_ts_sample(train_dtl, train_dtl_)

    # --- Model Definition ---
    net = Sequential(
        LSTM(1 + embedding_dim + 24 - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # --- Training Loop ---
    mse_history = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size,), sigma_v**2, dtype=np.float32)

        epoch_mse = 0
        num_samples = 0

        for x, y in batch_iter:
            ts = int(np.mean(x[-embedding_dim:]).tolist())
            embed_mu, embed_var = embeddings.get_embedding(ts)

            x_var = np.zeros_like(x)
            x[-embedding_dim:], x_var[-embedding_dim:] = embed_mu, embed_var

            m_pred, _ = net(np.float32(x), np.float32(x_var))
            epoch_mse += np.sum((y - m_pred.flatten()) ** 2)
            num_samples += len(y)

            out_updater.update(net.output_z_buffer, y, var_y, net.input_delta_z_buffer)
            net.backward()
            net.step()

            (mu_delta, var_delta) = net.get_input_states()
            x_update, var_update = mu_delta * x_var, x_var * var_delta * x_var
            embeddings.update(
                ts, x_update[-embedding_dim:], var_update[-embedding_dim:]
            )

        mse_history.append(epoch_mse / num_samples)
        pbar.set_postfix({"MSE": mse_history[-1]})

        reset_lstm_states(net)

    # --- Save Artifacts & Visualize ---
    net.save(os.path.join(out_dir, "model.path"))

    embedding_save_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(embedding_save_dir, exist_ok=True)
    embeddings.save(embedding_save_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mse_history, label="Training MSE")
    plt.title("Standard Embeddings: Training MSE"), plt.xlabel("Epoch"), plt.ylabel(
        "MSE"
    ), plt.legend(), plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_mse.png"))
    plt.close()

    pca = PCA(n_components=2)
    embed_mus = np.array([embeddings.get_embedding(i)[0] for i in range(nb_ts)])
    pca_result = pca.fit_transform(embed_mus)
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=100)
    for i in range(nb_ts):
        plt.annotate(f"ts{i}", (pca_result[i, 0], pca_result[i, 1]), fontsize=9)
    plt.title("PCA of 27 Unique Time Series Embeddings"), plt.xlabel(
        "PCA Component 1"
    ), plt.ylabel("PCA Component 2"), plt.grid(True)
    plt.savefig(os.path.join(out_dir, "pca_embeddings.png"))
    plt.close()

    # --- Testing ---
    m_preds, var_preds, m_true = ({i: [] for i in range(nb_ts)} for _ in range(3))
    for ts in tqdm(ts_idx, "Testing"):
        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_triplets_values.csv",
            date_time_file="data/toy_embedding/test_triplets_datetimes.csv",
            output_col=[0],
            input_seq_len=24,
            output_seq_len=1,
            num_features=1,
            stride=1,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )

        inputs, outputs = test_dtl.dataset["value"]
        datetimes = test_dtl.dataset["date_time"][24:]
        seq_window = inputs[0, :24].copy()
        embed_mu, embed_var = embeddings.get_embedding(ts)

        for i in range(len(outputs)):
            x = np.concatenate((seq_window, embed_mu))
            x_var = np.concatenate((np.zeros_like(seq_window), embed_var))
            m_pred, var_pred = net(np.float32(x), np.float32(x_var))
            mean_pred, var_pred_scalar = float(np.squeeze(m_pred)), float(
                np.squeeze(var_pred)
            )

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    plot_predictions(out_dir, nb_ts, datetimes, m_true, m_preds, var_preds)


def no_embeddings_run(num_epochs=20, batch_size=1, sigma_v=2.0, seed=42):
    """Trains a single model for all time series without any embeddings."""
    print("\n--- Running Experiment with No Embeddings (Baseline) ---")
    out_dir = "out/toy2_no_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    # --- Configuration ---
    nb_ts = 27
    ts_idx = np.arange(0, nb_ts)
    manual_seed(seed)

    # --- Data Loading ---
    train_dtl = None
    for ts in tqdm(ts_idx, "Loading Data"):
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/train_triplets_values.csv",
            date_time_file="data/toy_embedding/train_triplets_datetimes.csv",
            output_col=[0],
            input_seq_len=24,
            output_seq_len=1,
            num_features=1,
            stride=1,
            ts_idx=ts,
        )
        train_dtl = train_dtl_ if ts == 0 else concat_ts_sample(train_dtl, train_dtl_)

    # --- Model Definition ---
    net = Sequential(
        LSTM(1 + 24 - 1, 20, 1),
        LSTM(20, 20, 1),
        Linear(20, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # --- Training Loop ---
    mse_history = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size)
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size,), sigma_v**2, dtype=np.float32)

        epoch_mse = 0
        num_samples = 0

        for x, y in batch_iter:
            m_pred, _ = net(x)
            epoch_mse += np.sum((y - m_pred.flatten()) ** 2)
            num_samples += len(y)

            out_updater.update(net.output_z_buffer, y, var_y, net.input_delta_z_buffer)
            net.backward()
            net.step()

        mse_history.append(epoch_mse / num_samples)
        pbar.set_postfix({"MSE": mse_history[-1]})

        reset_lstm_states(net)

    # --- Save and Visualize ---
    net.save(os.path.join(out_dir, "model.path"))

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), mse_history, label="Training MSE")
    plt.title("No Embeddings: Training MSE"), plt.xlabel("Epoch"), plt.ylabel(
        "MSE"
    ), plt.legend(), plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_mse.png"))
    plt.close()

    # --- Testing ---
    m_preds, var_preds, m_true = ({i: [] for i in range(nb_ts)} for _ in range(3))
    for ts in tqdm(ts_idx, "Testing"):
        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_triplets_values.csv",
            date_time_file="data/toy_embedding/test_triplets_datetimes.csv",
            output_col=[0],
            input_seq_len=24,
            output_seq_len=1,
            num_features=1,
            stride=1,
            ts_idx=ts,
        )

        inputs, outputs = test_dtl.dataset["value"]
        datetimes = test_dtl.dataset["date_time"][24:]
        seq_window = inputs[0, :24].copy()

        for i in range(len(outputs)):
            m_pred, var_pred = net(np.float32(seq_window))
            mean_pred, var_pred_scalar = float(np.squeeze(m_pred)), float(
                np.squeeze(var_pred)
            )

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    plot_predictions(out_dir, nb_ts, datetimes, m_true, m_preds, var_preds)


def main():
    """Main function to orchestrate the experiments."""
    params = {"num_epochs": 30, "batch_size": 1, "sigma_v": 2.0, "seed": 235}
    shared_embeddings_run(**params)
    # embeddings_run(**params)
    # no_embeddings_run(**params)
    print("\nAll experiments completed.")


if __name__ == "__main__":
    # Ensure data is in the correct directory before running
    if not os.path.exists("data/toy_embedding/train_triplets_values.csv"):
        print(
            "Error: Data files not found in 'data/toy_embedding/'. Please check the path."
        )
    else:
        main()

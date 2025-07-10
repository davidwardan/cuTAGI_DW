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
            x_file="data/toy_embedding/train_triplets_values.csv",
            date_time_file="data/toy_embedding/train_triplets_datetimes.csv",
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

    # save model
    net.save(os.path.join(save_dir, "model.pth"))

    # ------------------------------------------------------------------
    # Test the model on *all* 27 (wave, amp, period) time‑series
    # ------------------------------------------------------------------
    import math

    m_preds = {i: [] for i in range(nb_ts)}
    var_preds = {i: [] for i in range(nb_ts)}
    m_true = {i: [] for i in range(nb_ts)}

    pbar = tqdm(ts_idx, desc="Testing Progress")
    for ts in pbar:
        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/toy_embedding/test_triplets_values.csv",
            date_time_file="data/toy_embedding/test_triplets_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            embedding_dim=embedding_dim,
            embed_at_end=True,
        )

        # Extract test sequences and timestamps
        inputs, outputs = test_dtl.dataset["value"]
        datetimes = test_dtl.dataset["date_time"][input_seq_len:]
        seq_window = inputs[0][:input_seq_len].copy()

        # Decode the series id -> (wave, amp, period) indices
        wave_idx = ts // 9
        amp_idx = (ts // 3) % 3
        period_idx = ts % 3

        # Pre‑fetch embedding parameters for this series
        wave_mu, wave_var = wave_embeddings.get_embedding(wave_idx)
        amp_mu, amp_var = amp_embeddings.get_embedding(amp_idx)
        per_mu, per_var = period_embeddings.get_embedding(period_idx)
        embed_mu = np.concatenate((wave_mu, amp_mu, per_mu), axis=0)
        embed_var = np.concatenate((wave_var, amp_var, per_var), axis=0)

        for i in range(len(outputs)):
            # Assemble input vector  [current window | embedding]
            x = np.concatenate((seq_window, np.zeros(embedding_dim, dtype=np.float32)))
            x_var = np.zeros_like(x)
            x[-embedding_dim:] = embed_mu
            x_var[-embedding_dim:] = embed_var

            # Forward pass
            m_pred, var_pred = net(np.float32(x), np.float32(x_var))
            mean_pred = float(np.squeeze(m_pred))
            var_pred_scalar = float(np.squeeze(var_pred))

            m_preds[ts].append(mean_pred)
            var_preds[ts].append(var_pred_scalar)
            m_true[ts].append(outputs[i][0])

            # Recursive window update
            seq_window = np.roll(seq_window, -1)
            seq_window[-1] = mean_pred

    # ------------------------------------------------------------------
    # Visualise predictions
    # ------------------------------------------------------------------
    true_dates = datetimes  # same length for all series
    n_cols = 3
    n_rows = math.ceil(nb_ts / n_cols)

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True, sharey=True
    )
    axs = axs.flatten()

    for idx, ts in enumerate(ts_idx):
        ax = axs[idx]
        ax.plot(true_dates, m_true[ts], color="red", label="True")
        ax.plot(true_dates, m_preds[ts], color="blue", label="Pred")
        ax.fill_between(
            true_dates,
            np.array(m_preds[ts]) - np.sqrt(np.array(var_preds[ts])),
            np.array(m_preds[ts]) + np.sqrt(np.array(var_preds[ts])),
            color="blue",
            alpha=0.3,
        )
        ax.set_title(f"TS {ts} (w{ts//9}, a{(ts//3)%3}, p{ts%3})", fontsize=9)
        ax.legend(fontsize=7)

    # Hide unused axes (if any)
    for ax in axs[nb_ts:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

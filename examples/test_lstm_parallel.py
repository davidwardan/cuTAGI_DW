import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from typing import Optional

from examples.embedding_loader import (
    TimeSeriesEmbeddings,
)
from examples.data_loader import GlobalTimeSeriesDataloader
from pytagi import manual_seed
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential, EvenExp

import matplotlib as mpl
from adjustText import adjust_text

# Update matplotlib parameters in a single dictionary
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
        "pgf.preamble": r"\usepackage{amsmath}",
        "lines.linewidth": 1,  # Set line width to 1
    }
)


# -- Helper --
def update_look_back_buffer(
    look_back_buffer: np.ndarray,
    ts_idx,
    m_pred,
    *,
    create_if_none: bool = False,
    nb_ts: Optional[int] = None,
    input_seq_len: Optional[int] = None,
) -> np.ndarray:
    """
    Update per-series 1D look-back buffers with the latest predictions from a batch.

    Parameters
    ----------
    look_back_buffer : np.ndarray
        Array of shape (nb_ts, input_seq_len) holding history per series.
        Will be modified in place. If None and create_if_none=True, a new buffer is created.
    ts_idx : array-like
        Series ids for each sample in the batch, length B (batch_size). Works for B=1+.
    m_pred : array-like
        Model predictions for the batch. If shape is (B,), each sample contributes one value.
        If shape is (B, T) or flat size B*T, the last column (time step) per sample is appended.
    create_if_none : bool, optional
        If True and look_back_buffer is None, create it using nb_ts and input_seq_len.
    nb_ts : int, optional
        Required when create_if_none=True. Number of series.
    input_seq_len : int, optional
        Required when create_if_none=True. Length of the per-series buffer.

    Returns
    -------
    np.ndarray
        The (possibly newly created) look_back_buffer.

    Notes
    -----
    - Handles duplicates in `ts_idx` by appending multiple times to that series in order.
    - For k new values for a series with buffer length L:
        * if k >= L: the buffer becomes the last L values of the k predictions.
        * else: shift left by k and append the k values at the end.
    """
    # Create if requested
    if look_back_buffer is None:
        if not create_if_none or nb_ts is None or input_seq_len is None:
            raise ValueError(
                "look_back_buffer is None; set create_if_none=True and provide nb_ts & input_seq_len."
            )
        look_back_buffer = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)

    # Normalize inputs
    ts_idx = np.asarray(ts_idx, dtype=int).ravel()
    B = ts_idx.size

    m_pred = np.asarray(m_pred)
    if m_pred.ndim == 0:
        # scalar -> treat as one-sample batch
        last_step = np.array([float(m_pred)], dtype=np.float32)
    elif m_pred.size == B:
        # one value per sample (B,)
        last_step = m_pred.reshape(B).astype(np.float32, copy=False)
    else:
        # assume (B, T) or flat B*T -> take last time step per sample
        last_step = m_pred.reshape(B, -1)[:, -1].astype(np.float32, copy=False)

    # Collect predictions per series in order of appearance
    preds_by_series = {}
    for b in range(B):
        s = ts_idx[b]
        preds_by_series.setdefault(s, []).append(last_step[b])

    # Update buffers once per series
    for s, preds in preds_by_series.items():
        buf = look_back_buffer[s]
        L = buf.size
        k = len(preds)
        preds_arr = np.asarray(preds, dtype=buf.dtype)

        if k >= L:
            # keep only the last L values
            look_back_buffer[s] = preds_arr[-L:]
        else:
            # shift left by k, append k values
            buf[:-k] = buf[k:]
            buf[-k:] = preds_arr

    return look_back_buffer


def calculate_gaussian_posterior(m_pred, v_pred, y, var_obs):
    if not np.isnan(y).any():
        # Ensure variances are non-negative and add a small epsilon to prevent division by zero
        K = v_pred / (v_pred + var_obs)  # Kalman gain
        m_post = m_pred + K * (y - m_pred)  # posterior mean
        v_post = (1.0 - K) * v_pred  # posterior variance

        # nan or inf reutn pred
        if (
            np.isnan(m_post).any()
            or np.isnan(v_post).any()
            or np.isinf(m_post).any()
            or np.isinf(v_post).any()
        ):
            return m_pred, v_pred
        else:
            return m_post.astype(np.float32), v_post.astype(np.float32)
    else:
        return m_pred, v_pred


def batch_get_lstm_states(ts_ids, net, lstm_states):
    """Capture the current LSTM states for each series id in the batch.

    Aggregates the hidden and cell state means/variances returned by
    ``net.get_lstm_states()`` and stores the slice relevant to every series id
    in ``ts_ids`` inside the ``lstm_states`` cache.
    """
    states = net.get_lstm_states()
    batch_size = len(ts_ids)
    if states and batch_size > 0:
        per_layer_states = {}
        for layer_idx, layer_state in states.items():
            mu_h, var_h, mu_c, var_c = (list(arr) for arr in layer_state)
            total = len(mu_h)
            chunk, rem = divmod(total, batch_size) if batch_size else (total, 0)
            per_layer_states[layer_idx] = (
                mu_h,
                var_h,
                mu_c,
                var_c,
                chunk if rem == 0 else 0,
            )

        for batch_pos, sid_val in enumerate(ts_ids):
            series_idx = int(sid_val)
            series_states = {}
            for layer_idx, (
                mu_h,
                var_h,
                mu_c,
                var_c,
                chunk,
            ) in per_layer_states.items():
                if chunk > 0 and batch_size > 1:
                    start = batch_pos * chunk
                    end = start + chunk
                    series_states[layer_idx] = (
                        mu_h[start:end],
                        var_h[start:end],
                        mu_c[start:end],
                        var_c[start:end],
                    )
                else:
                    series_states[layer_idx] = (
                        mu_h.copy(),
                        var_h.copy(),
                        mu_c.copy(),
                        var_c.copy(),
                    )
            lstm_states[series_idx] = series_states

    return lstm_states


def batch_set_lstm_states(ts_ids, net, lstm_states):
    """Restore cached LSTM states for the provided series ids onto the network."""
    batch_states = []
    for sid_val in ts_ids:
        stored_state = lstm_states[int(sid_val)]
        if stored_state is None:
            batch_states = []
            break
        batch_states.append(stored_state)

    if batch_states:
        merged_states: dict[
            int, tuple[list[float], list[float], list[float], list[float]]
        ] = {}
        for layer_idx in batch_states[0]:
            mu_h_cat: list[float] = []
            var_h_cat: list[float] = []
            mu_c_cat: list[float] = []
            var_c_cat: list[float] = []
            for series_states in batch_states:
                layer_state = series_states[layer_idx]
                mu_h_cat.extend(layer_state[0])
                var_h_cat.extend(layer_state[1])
                mu_c_cat.extend(layer_state[2])
                var_c_cat.extend(layer_state[3])
            merged_states[layer_idx] = (
                mu_h_cat,
                var_h_cat,
                mu_c_cat,
                var_c_cat,
            )

        net.set_lstm_states(merged_states)


# -- Experiments --
def normal_run(num_epochs, batch_size, seed, embedding_dim):
    """
    Runs the time series forecasting experiment without embeddings.
    """

    # --- Configuration ---
    nb_ts = 27  # 4
    output_col = [0]
    num_features = 2
    input_seq_len = 24
    seq_stride = 1
    output_seq_len = 1

    # Initialize embeddings
    if embedding_dim is not None:
        embeddings = TimeSeriesEmbeddings(
            (nb_ts, embedding_dim),
            encoding_type="sphere",
            seed=seed,
        )
    else:
        embeddings = None

    # --- Load Data ---
    train_dtl = GlobalTimeSeriesDataloader(
        x_file="data/toy_embedding/train_triplets_values.csv",
        date_time_file="data/toy_embedding/train_triplets_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["hour_of_day"],
        scale_covariates=True,
        scale_method="standard",
        order_mode="by_window",
        random_seed=seed,
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    # --- Define Model ---
    manual_seed(seed)
    if embedding_dim is not None:
        input_size = input_seq_len + num_features + embedding_dim - 1
    else:
        input_size = input_seq_len + num_features - 1
    net = Sequential(
        LSTM(input_size, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )
    net.set_threads(1)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # save for plotting
    train_mses = []
    train_log_liks = []

    # --- Training ---
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        mu_preds = [[] for _ in range(nb_ts)]
        std_preds = [[] for _ in range(nb_ts)]
        train_obs = [[] for _ in range(nb_ts)]

        batch_iter = train_dtl.create_data_loader(
            batch_size=batch_size,
            shuffle=False,  # full shuffle
            include_ids=True,
            # shuffle_series_blocks=True,  # ordered shuffle
        )

        # define the lookback buffer for recursive prediction
        look_back_buffer_mu = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        look_back_buffer_var = np.full((nb_ts, input_seq_len), 0.0, dtype=np.float32)
        lstm_states = [None for _ in range(nb_ts)]

        for x, y, ts_id, _ in batch_iter:

            # ts_idx = ts_id.item()
            B = int(len(ts_id))  # current batch size

            ts_ids = np.asarray(ts_id, dtype=np.int64).reshape(-1)
            if epoch != 0:
                net.reset_lstm_states()
            if ts_ids.size:
                batch_set_lstm_states(ts_ids, net, lstm_states)

            x = np.nan_to_num(x, nan=0.0)  # clean input from nans
            x_var = np.zeros_like(x)  # takes care of covariates
            y = np.concatenate(y, axis=0).astype(np.float32)  # shape (B,)

            # initialize look back buffer if first step for any series in batch
            if np.isnan(look_back_buffer_mu[ts_id]).all():
                look_back_buffer_mu[ts_id] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer_mu[
                    ts_id
                ]  # update input sequence

            # always update the var buffer (no nans)
            x_var[:, :input_seq_len] = look_back_buffer_var[
                ts_id
            ]  # update input sequence

            # append embeddings to each input in the batch
            if embedding_dim is not None:
                embed_mu, embed_var = embeddings(ts_id)  # shape: (B, embedding_dim)
                x = np.concatenate(
                    (x, embed_mu), axis=1
                )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
                x_var = np.concatenate(
                    (x_var, embed_var), axis=1
                )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

            # flatten input for pytagi
            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Forward
            m_pred, v_pred = net(flat_x, flat_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # Update output layer
            out_updater.update_heteros(
                output_states=net.output_z_buffer,
                mu_obs=y.flatten(),
                delta_states=net.input_delta_z_buffer,
            )

            # Backward + step
            net.backward()
            net.step()

            # store lstm states
            lstm_states = batch_get_lstm_states(ts_ids, net, lstm_states)

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            # get the posterior states
            m_pred, v_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            if embedding_dim is not None:
                # get updates for embeddings
                mu_delta, var_delta = net.get_input_states()
                mu_delta = mu_delta.reshape(B, -1)
                var_delta = var_delta.reshape(B, -1)

                x_update = mu_delta * x_var
                var_update = x_var * var_delta * x_var

                embeddings.update(
                    ts_id, x_update[:, -embedding_dim:], var_update[:, -embedding_dim:]
                )

            for b in range(B):
                sid = int(ts_id[b])

                # append to per-series lists
                mu_preds[sid].append(float(m_prior[b]))
                std_preds[sid].append(float(std_prior[b]))
                train_obs[sid].append(float(y[b]))

            look_back_buffer_mu = update_look_back_buffer(
                look_back_buffer_mu,
                ts_idx=ts_id,
                m_pred=m_pred,
            )
            look_back_buffer_var = update_look_back_buffer(
                look_back_buffer_var,
                ts_idx=ts_id,
                m_pred=v_pred + var_obs,
            )

        # get train metrics
        mses = []
        log_liks = []
        for s in range(nb_ts):
            pred = np.asarray(mu_preds[s])
            std = np.asarray(std_preds[s])
            obs = np.asarray(train_obs[s])
            train_mse = metric.mse(pred, obs)
            train_log_lik = metric.log_likelihood(
                prediction=pred, observation=obs, std=std
            )
            mses.append(train_mse)
            log_liks.append(train_log_lik)

        train_mse = np.nanmean(mses)
        train_log_lik = np.nanmean(log_liks)
        train_mses.append(train_mse)
        train_log_liks.append(train_log_lik)

        # unstandardize
        for s in range(nb_ts):
            mu_preds[s] = np.array(mu_preds[s])
            std_preds[s] = np.array(std_preds[s])
            train_obs[s] = np.array(train_obs[s])
            mu_preds[s] = normalizer.unstandardize(
                mu_preds[s], global_mean[s], global_std[s]
            )
            std_preds[s] = normalizer.unstandardize_std(std_preds[s], global_std[s])
            train_obs[s] = normalizer.unstandardize(
                train_obs[s], global_mean[s], global_std[s]
            )

    # --- Testing ---
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloader(
        x_file="data/toy_embedding/test_triplets_values.csv",
        date_time_file="data/toy_embedding/test_triplets_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["hour_of_day"],
        scale_covariates=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_series",
    )

    test_batch_iter = test_dtl.create_data_loader(
        batch_size=1, shuffle=False, include_ids=True
    )

    # Collect predictions per series
    test_mu_preds = [[] for _ in range(nb_ts)]
    test_std_preds = [[] for _ in range(nb_ts)]
    test_obs = [[] for _ in range(nb_ts)]

    # define the lookback buffer for recursive prediction (per series)
    look_back_buffer_test_mu = np.copy(look_back_buffer_mu)
    look_back_buffer_test_var = np.copy(look_back_buffer_var)
    for x, y, ts_id, _ in test_batch_iter:

        ts_idx = ts_id.item()
        ts_ids = np.asarray(ts_id).reshape(-1)

        # set LSTM states
        net.set_lstm_states(lstm_states[ts_idx])

        x_var = np.zeros_like(x)  # takes care of covariates

        # insert values from lookback_buffer
        x[:, :input_seq_len] = look_back_buffer_test_mu[ts_id]  # update input sequence
        x_var[:, :input_seq_len] = look_back_buffer_test_var[
            ts_id
        ]  # update input sequence

        if embedding_dim is not None:
            # append embeddings to each input in the batch
            embed_mu, embed_var = embeddings(ts_id)  # shape: (B, embedding_dim)
            x = np.concatenate(
                (x, embed_mu), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
            x_var = np.concatenate(
                (x_var, embed_var), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

        # flatten input for pytagi
        flat_x = np.concatenate(x, axis=0, dtype=np.float32)
        flat_var = np.concatenate(x_var, axis=0, dtype=np.float32)

        # Prediction
        m_pred, v_pred = net(flat_x, flat_var)

        flat_m = np.ravel(m_pred)
        flat_v = np.ravel(v_pred)

        m_pred = flat_m[::2]  # even indices
        v_pred = flat_v[::2]  # even indices
        var_obs = flat_m[1::2]  # odd indices var_v

        # store lstm states
        lstm_states[ts_idx] = net.get_lstm_states()

        # Unstadardize
        scaled_m_pred = normalizer.unstandardize(
            m_pred, global_mean[ts_idx], global_std[ts_idx]
        )
        scaled_std_pred = normalizer.unstandardize_std(
            (np.sqrt(v_pred + var_obs)), global_std[ts_idx]
        )
        scaled_y = normalizer.unstandardize(y, global_mean[ts_idx], global_std[ts_idx])

        test_mu_preds[ts_idx].extend(np.asarray(scaled_m_pred).ravel().tolist())
        test_std_preds[ts_idx].extend(np.asarray(scaled_std_pred).ravel().tolist())
        test_obs[ts_idx].extend(np.asarray(scaled_y).ravel().tolist())

        # Update lookback buffer with most recent prediction
        look_back_buffer_test_mu[ts_idx][:-1] = look_back_buffer_test_mu[ts_idx][1:]
        look_back_buffer_test_mu[ts_idx][-1] = float(np.ravel(m_pred)[-1])

        look_back_buffer_test_var[ts_idx][:-1] = look_back_buffer_test_var[ts_idx][1:]
        look_back_buffer_test_var[ts_idx][-1] = float(np.ravel(v_pred + var_obs)[-1])

    # --- Plotting & Metrics ---
    suffix = f"_{nb_ts}ts" if nb_ts is not None else ""
    saving_path = os.path.join("out", f"test_lstm_parallel{suffix}")
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    series_metrics = []
    header = f"{'Series':<10}{'Train MSE':>15}{'Test MSE':>15}{'Train LogLik':>18}{'Test LogLik':>18}"
    print("\nSummary Metrics")
    print(header)
    print("-" * len(header))

    def _fmt(val: float) -> str:
        return f"{val:15.4f}" if np.isfinite(val) else f"{'--':>15}"

    for ts in range(nb_ts):
        train_mu = np.asarray(mu_preds[ts], dtype=np.float32)
        train_std = np.clip(np.asarray(std_preds[ts], dtype=np.float32), 1e-6, None)
        train_y = np.asarray(train_obs[ts], dtype=np.float32)

        test_mu = np.asarray(test_mu_preds[ts], dtype=np.float32)
        test_std = np.clip(np.asarray(test_std_preds[ts], dtype=np.float32), 1e-6, None)
        test_y = np.asarray(test_obs[ts], dtype=np.float32)

        if train_mu.size and train_y.size:
            train_mse = metric.mse(train_mu, train_y)
            train_log_lik = metric.log_likelihood(train_mu, train_y, train_std)
        else:
            train_mse = np.nan
            train_log_lik = np.nan

        if test_mu.size and test_y.size:
            test_mse = metric.mse(test_mu, test_y)
            test_log_lik = metric.log_likelihood(test_mu, test_y, test_std)
        else:
            test_mse = np.nan
            test_log_lik = np.nan

        series_metrics.append((train_mse, test_mse, train_log_lik, test_log_lik))

        print(
            f"{f'TS {ts}':<10}"
            f"{_fmt(train_mse)}"
            f"{_fmt(test_mse)}"
            f"{_fmt(train_log_lik)}"
            f"{_fmt(test_log_lik)}"
        )

        # Prepare data for plotting
        mu_preds_ts = np.concatenate((train_mu, test_mu), axis=0)
        std_preds_ts = np.concatenate((train_std, test_std), axis=0)
        y_true_ts = np.concatenate((train_y, test_y), axis=0)

        total_len = mu_preds_ts.size
        if total_len == 0:
            continue

        x_all = np.arange(total_len)
        train_len = train_mu.size

        std_factor = 1
        lower_band = mu_preds_ts - std_factor * std_preds_ts
        upper_band = mu_preds_ts + std_factor * std_preds_ts

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x_all, y_true_ts, label=r"$y_{true}$", color="red", linewidth=1.5)
        ax.plot(
            x_all, mu_preds_ts, label=r"$\mathbb{E}[Y']$", color="blue", linewidth=1.5
        )
        ax.fill_between(
            x_all,
            lower_band,
            upper_band,
            color="blue",
            alpha=0.2,
            label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(std_factor),
        )
        # set y limits

        ax.set_ylim(-2.1, 2.1)
        if test_mu.size:
            ax.axvline(
                train_len - 0.5,
                color="0.3",
                linestyle="--",
                linewidth=1,
                label="Train/Test Split",
            )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=4,
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(f"{saving_path}/lstm_series_{ts}.png", dpi=600)
        plt.close(fig)

    if series_metrics:
        avg_train_mse = np.nanmean([m[0] for m in series_metrics])
        avg_test_mse = np.nanmean([m[1] for m in series_metrics])
        avg_train_log_lik = np.nanmean([m[2] for m in series_metrics])
        avg_test_log_lik = np.nanmean([m[3] for m in series_metrics])
        avg_row = (
            f"{'Average':<10}"
            f"{_fmt(avg_train_mse)}"
            f"{_fmt(avg_test_mse)}"
            f"{_fmt(avg_train_log_lik)}"
            f"{_fmt(avg_test_log_lik)}"
        )
        print("-" * len(header))
        print(avg_row)

    if embedding_dim is not None:
        embedding_mu = getattr(embeddings, "mu_embedding", None)
        if embedding_mu is not None and embedding_mu.size:
            embedding_mu = np.asarray(embedding_mu, dtype=np.float32)
            num_series = embedding_mu.shape[0]
            norms = np.linalg.norm(embedding_mu, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            cosine_sim = embedding_mu @ embedding_mu.T / (norms * norms.T)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)


            # ts_labels = [f"TS{idx}" for idx in range(num_series)]
            with open("data/toy_embedding/train_triplets_values.csv", "r") as f:
                header_line = f.readline().strip()
            ts_labels = header_line.split(",")
            if len(ts_labels) != num_series:
                ts_labels = [f"TS{idx}" for idx in range(num_series)]
            heatmap_size = max(6.0, num_series * 0.5)
            fig, ax = plt.subplots(figsize=(heatmap_size, heatmap_size))
            im = ax.imshow(cosine_sim, cmap="viridis", vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(num_series))
            ax.set_yticks(range(num_series))
            ax.set_xticklabels(ts_labels, rotation=45, ha="right")
            ax.set_yticklabels(ts_labels)
            ax.set_title("Embedding Cosine Similarity")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Cosine similarity", rotation=270, labelpad=15)
            fig.tight_layout()
            fig.savefig(f"{saving_path}/lstm_embedding_similarity.png", dpi=600)
            plt.close(fig)

            if embedding_mu.shape[1] > 2:
                centered = embedding_mu - embedding_mu.mean(axis=0, keepdims=True)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                coords = centered @ vh[:2].T
            elif embedding_mu.shape[1] == 2:
                coords = embedding_mu
            else:
                coords = np.concatenate(
                    [
                        embedding_mu,
                        np.zeros((embedding_mu.shape[0], 1), dtype=embedding_mu.dtype),
                    ],
                    axis=1,
                )

            scatter_width = max(6.0, num_series * 0.4)
            fig_scatter, ax_scatter = plt.subplots(figsize=(scatter_width, 5))
            ax_scatter.scatter(coords[:, 0], coords[:, 1], color="blue", s=25, alpha=0.7)
            texts = [
                ax_scatter.text(
                    x_val, y_val, f"TS{series_idx}", fontsize=5, ha="left", va="bottom"
                )
                for series_idx, (x_val, y_val) in enumerate(coords)
            ]
            adjust_text(
                texts,
                ax=ax_scatter,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                lim=50,
            )

            label_map = {f"TS{idx}": ts_labels[idx] for idx in range(num_series)}
            legend_texts = [f"{key}: {value}" for key, value in label_map.items()]
            legend_str = "\n".join(legend_texts)
            ax_scatter.text(
                1.02,
                0.5,
                legend_str,
                transform=ax_scatter.transAxes,
                fontsize=6,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.5),
            )
            ax_scatter.set_xlabel("Component 1")
            ax_scatter.set_ylabel("Component 2")
            fig_scatter.tight_layout()
            fig_scatter.savefig(f"{saving_path}/lstm_embeddings.png", dpi=600)
            plt.close(fig_scatter)

            embedding_var = getattr(embeddings, "var_embedding", None)
            embedding_std = None
            if embedding_var is not None and np.size(embedding_var):
                embedding_std = np.sqrt(embedding_var, dtype=np.float32)

            n_embeddings = embedding_mu.shape[0]
            ncols = min(4, max(1, int(np.ceil(np.sqrt(n_embeddings)))))
            nrows = int(np.ceil(n_embeddings / ncols))
            fig_width = max(8, 4 * ncols)
            fig_height = max(6, 3 * nrows)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
            )
            axes = np.atleast_1d(axes).ravel()
            dims = np.arange(embedding_mu.shape[1], dtype=np.int32)
            for subplot_idx, ax in enumerate(axes[:n_embeddings]):
                mean_vals = embedding_mu[subplot_idx]
                ax.plot(dims, mean_vals, marker="o", label="mean", color="tab:blue")

                if embedding_std is not None:
                    std_vals = embedding_std[subplot_idx]
                    ax.fill_between(
                        dims,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color="tab:blue",
                        alpha=0.3,
                    )

                ax.set_title(f"{ts_labels[subplot_idx]}")
                ax.set_xlabel("Dimension")
                ax.set_ylabel("Value")
                ax.set_xticks(dims)
            for ax in axes[n_embeddings:]:
                ax.axis("off")

            fig.tight_layout()
            fig.savefig(f"{saving_path}/lstm_embedding_stats.png", dpi=600)
            plt.close(fig)


def main(num_epochs=50, batch_size=27, seed=1, embedding_dim=10):
    """
    Main function to run all experiments.
    """
    normal_run(num_epochs, batch_size, seed, embedding_dim)


if __name__ == "__main__":
    main()

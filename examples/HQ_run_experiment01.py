import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
import copy

from examples.embedding_loader import (
    TimeSeriesEmbeddings,
)
from examples.data_loader import (
    TimeSeriesDataloader,
    GlobalTimeSeriesDataloader,
    GlobalTimeSeriesDataloaderV2,
)
from pytagi import exponential_scheduler, manual_seed
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential, EvenExp

import matplotlib as mpl

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


# --- Helper functions --- #
def get_combined_embeddings(
    ts_idx,
    embedding_id_map,
    dam_embeddings,
    dam_type_embeddings,
    sensor_type_embeddings,
    direction_embeddings,
    sensor_embeddings,
):
    dam_idx = np.array([embedding_id_map[i][0] for i in ts_idx], dtype=int)
    dam_type_idx = np.array([embedding_id_map[i][1] for i in ts_idx], dtype=int)
    sensor_type_idx = np.array([embedding_id_map[i][2] for i in ts_idx], dtype=int)
    direction_idx = np.array([embedding_id_map[i][3] for i in ts_idx], dtype=int)
    sensor_idx = np.array([embedding_id_map[i][4] for i in ts_idx], dtype=int)

    dam_mu, dam_var = dam_embeddings(dam_idx)
    dam_type_mu, dam_type_var = dam_type_embeddings(dam_type_idx)
    sensor_type_mu, sensor_type_var = sensor_type_embeddings(sensor_type_idx)
    direction_mu, direction_var = direction_embeddings(direction_idx)
    sensor_mu, sensor_var = sensor_embeddings(sensor_idx)

    combined_mu = np.concatenate(
        (dam_mu, dam_type_mu, sensor_type_mu, direction_mu, sensor_mu), axis=1
    )  # shape: (B, embedding_dim)
    combined_var = np.concatenate(
        (dam_var, dam_type_var, sensor_type_var, direction_var, sensor_var), axis=1
    )  # shape: (B, embedding_dim)

    return combined_mu, combined_var


def update_combined_embeddings(
    ts_idx,
    mu_delta,
    var_delta,
    embedding_id_map,
    dam_embeddings,
    dam_type_embeddings,
    sensor_type_embeddings,
    direction_embeddings,
    sensor_embeddings,
):
    ts_idx = np.asarray(ts_idx, dtype=int).ravel()
    if ts_idx.size == 0:
        return

    mu_delta = np.asarray(mu_delta)
    var_delta = np.asarray(var_delta)

    if mu_delta.ndim == 1:
        mu_delta = mu_delta.reshape(ts_idx.size, -1)
    if var_delta.ndim == 1:
        var_delta = var_delta.reshape(ts_idx.size, -1)

    dam_dim = dam_embeddings.mu_embedding.shape[1]
    dam_type_dim = dam_type_embeddings.mu_embedding.shape[1]
    sensor_type_dim = sensor_type_embeddings.mu_embedding.shape[1]
    direction_dim = direction_embeddings.mu_embedding.shape[1]
    sensor_dim = sensor_embeddings.mu_embedding.shape[1]

    total_dim = dam_dim + dam_type_dim + sensor_type_dim + direction_dim + sensor_dim
    if mu_delta.shape[1] != total_dim or var_delta.shape[1] != total_dim:
        raise ValueError("Mismatched embedding update dimensions.")

    split_points = np.cumsum(
        [dam_dim, dam_type_dim, sensor_type_dim, direction_dim], dtype=int
    )
    mu_splits = np.split(mu_delta, split_points, axis=1)
    var_splits = np.split(var_delta, split_points, axis=1)

    idx_map = np.array([embedding_id_map[int(i)] for i in ts_idx], dtype=int)
    dam_idx, dam_type_idx, sensor_type_idx, direction_idx, sensor_idx = idx_map.T

    dam_embeddings.update(dam_idx, mu_splits[0], var_splits[0])
    dam_type_embeddings.update(dam_type_idx, mu_splits[1], var_splits[1])
    sensor_type_embeddings.update(sensor_type_idx, mu_splits[2], var_splits[2])
    direction_embeddings.update(direction_idx, mu_splits[3], var_splits[3])
    sensor_embeddings.update(sensor_idx, mu_splits[4], var_splits[4])


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


def reset_lstm_states(net):
    # reset LSTM states to zeros
    lstm_states = net.get_lstm_states()
    for key in lstm_states:
        old_tuple = lstm_states[key]
        new_tuple = tuple(np.zeros_like(np.array(v)).tolist() for v in old_tuple)
        lstm_states[key] = new_tuple
    net.set_lstm_states(lstm_states)


def calculate_gaussian_posterior(m_pred, v_pred, y, var_obs):
    if not np.isnan(y).any():
        K_overfit = v_pred / (v_pred + 1e-3)  # Kalman gain with small obs noise
        K = v_pred / (v_pred + var_obs)  # Kalman gain
        m_pred = m_pred + K_overfit * (y - m_pred)  # posterior mean
        v_pred = (1.0 - K) * v_pred  # posterior variance
        return m_pred.astype(np.float32), v_pred.astype(np.float32)
    else:
        return m_pred, v_pred


def local_model_run(nb_ts, num_epochs, batch_size, seed, early_stopping_criteria):
    """ "
    Runs a seperate local model for each time series in the dataset
    """

    print("Running local models...")
    ts_idx = np.arange(0, nb_ts)

    # --- Output Directory ---
    out_dir = "out/experiment01_local"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # define placeholders for final CSVs
    horizon_cap = 2000

    # create placeholders
    ytestPd = np.full((horizon_cap, nb_ts), np.nan)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # --- Load Data ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    for ts in pbar:

        output_col = [0]
        num_features = 2
        input_seq_len = 52
        output_seq_len = 1
        seq_stride = 1
        log_lik_optim = -1e100
        mse_optim = 1e100
        epoch_optim = 0
        net_optim = []
        patience = 10
        min_epochs = 15

        # set random seed for reproducibility
        manual_seed(seed)

        train_dtl = TimeSeriesDataloader(
            x_file="data/hq/split_train_values.csv",
            date_time_file="data/hq/split_train_datetimes.csv",
            time_covariates=["week_of_year"],
            keep_last_time_cov=True,
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
        )

        val_dtl = TimeSeriesDataloader(
            x_file="data/hq/split_val_values.csv",
            date_time_file="data/hq/split_val_datetimes.csv",
            time_covariates=["week_of_year"],
            keep_last_time_cov=True,
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            ts_idx=ts,
        )

        test_dtl = TimeSeriesDataloader(
            x_file="data/hq/split_test_values.csv",
            date_time_file="data/hq/split_test_datetimes.csv",
            time_covariates=["week_of_year"],
            keep_last_time_cov=True,
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            ts_idx=ts,
        )

        # --- Define Model ---
        net = Sequential(
            LSTM(num_features + input_seq_len - 1, 40, 1),
            LSTM(40, 40, 1),
            Linear(40, 2),
            EvenExp(),
        )
        net.set_threads(1)
        out_updater = OutputUpdater(net.device)

        look_back_buffer_val_optim = None
        states_optim = None
        lstm_optim_states = None

        # --- Training ---
        pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in pbar:
            mu_preds = []
            std_preds = []
            train_obs = []

            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

            look_back_buffer = None

            for x, y in batch_iter:

                # replace nans in x with zeros
                x = np.nan_to_num(x, nan=0.0)

                if look_back_buffer is None:
                    look_back_buffer = x[:input_seq_len]
                else:
                    look_back_buffer[:-1] = look_back_buffer[1:]  # shift left
                    look_back_buffer[-1] = float(
                        np.ravel(m_pred)[-1]
                    )  # append most recent pred
                    x[:input_seq_len] = look_back_buffer  # update input sequence
                    x = x.astype(np.float32)

                # Feed forward
                m_pred, v_pred = net(x)

                flat_m = np.ravel(m_pred)
                flat_v = np.ravel(v_pred)

                m_pred = flat_m[::2]  # even indices
                v_pred = flat_v[::2]  # even indices
                var_obs = flat_m[1::2]  # odd indices var_v

                # Update output layer
                out_updater.update_heteros(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    delta_states=net.input_delta_z_buffer,
                )

                # Feed backward
                net.backward()
                net.step()

                mu_preds.extend(m_pred)  # stores prior value
                std_preds.extend(np.sqrt(v_pred + var_obs))  # stores full uncertainty
                train_obs.extend(y)

                m_pred, std_pred = calculate_gaussian_posterior(
                    m_pred, v_pred, y, var_obs
                )

            mu_preds = np.array(mu_preds)
            std_preds = np.array(std_preds)
            train_obs = np.array(train_obs)

            mu_preds = normalizer.unstandardize(
                mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            std_preds = normalizer.unstandardize_std(
                std_preds, train_dtl.x_std[output_col]
            )
            train_obs = normalizer.unstandardize(
                train_obs, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            train_mse = metric.mse(mu_preds, train_obs)

            # Validation
            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds_val = []
            std_preds_val = []
            val_obs = []

            # define the lookback buffer for recursive prediction
            look_back_buffer_val = copy.copy(look_back_buffer)

            # One-step recursive prediction over the validation stream
            for x, y in val_batch_iter:

                if look_back_buffer_val is None:
                    look_back_buffer_val = x[:input_seq_len]
                else:
                    look_back_buffer_val[:-1] = look_back_buffer_val[1:]  # shift left
                    look_back_buffer_val[-1] = float(
                        np.ravel(m_pred)[-1]
                    )  # append most recent pred
                    x[:input_seq_len] = look_back_buffer_val  # update input sequence
                    x = x.astype(np.float32)

                # Predicion
                m_pred, v_pred = net(x)

                # get even positions corresponding to Z_out
                m_pred = np.ravel(m_pred)[::2]
                v_pred = np.ravel(v_pred)[::2]
                var_obs = flat_m[1::2]  # odd indices var_v

                mu_preds_val.extend(m_pred)
                std_preds_val.extend(np.sqrt(v_pred + var_obs))
                val_obs.extend(y)

                m_pred, std_pred = calculate_gaussian_posterior(
                    m_pred, v_pred, y, var_obs
                )

            mu_preds_val = np.array(mu_preds_val)
            std_preds_val = np.array(std_preds_val)
            val_obs = np.array(val_obs)

            mu_preds_val = normalizer.unstandardize(
                mu_preds_val, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            std_preds_val = normalizer.unstandardize_std(
                std_preds_val, train_dtl.x_std[output_col]
            )

            val_obs = normalizer.unstandardize(
                val_obs, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )

            # Compute log-likelihood for validation set
            mse_val = metric.mse(mu_preds_val, val_obs)
            log_lik_val = metric.log_likelihood(
                prediction=mu_preds_val, observation=val_obs, std=std_preds_val
            )

            # Progress bar
            pbar.set_description(
                f"Ts #{ts+1}/{nb_ts} | Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| mse_val: {mse_val:>7.4f} | log_lik_val: {log_lik_val:>7.4f}",
                refresh=True,
            )

            # early-stopping
            if early_stopping_criteria == "mse":
                if float(mse_val) < float(mse_optim):
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        mu_preds_val,
                        std_preds_val,
                    )
                    look_back_buffer_val_optim = copy.copy(look_back_buffer_val)
                    lstm_optim_states = net.get_lstm_states()
            elif early_stopping_criteria == "log_lik":
                if float(log_lik_val) > float(log_lik_optim):
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        mu_preds_val,
                        std_preds_val,
                    )
                    look_back_buffer_val_optim = copy.copy(look_back_buffer_val)
                    lstm_optim_states = net.get_lstm_states()
            if (epoch + 1) >= min_epochs and int(epoch) - int(epoch_optim) > patience:
                break

        # -- Testing --
        if net_optim:
            net.load_state_dict(net_optim)
        net.save(out_dir + "/param/model_{}.pth".format(str(ts)))
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        # load optimal lstm states
        net.set_lstm_states(lstm_optim_states)

        # unpack optimal predictions
        mu_preds, std_preds, mu_preds_val, std_preds_val = states_optim

        mu_preds_test = []
        var_preds_test = []
        test_obs = []

        # define the lookback buffer for recursive prediction
        look_back_buffer_test = copy.copy(look_back_buffer_val_optim)

        # One-step recursive prediction over the validation stream
        for x, y in test_batch_iter:

            if look_back_buffer_test is None:
                look_back_buffer_test = x[:input_seq_len]
            else:
                look_back_buffer_test[:-1] = look_back_buffer_test[1:]  # shift left
                look_back_buffer_test[-1] = float(
                    np.ravel(m_pred)[-1]
                )  # append most recent pred
                x[:input_seq_len] = look_back_buffer_test  # update input sequence
                x = x.astype(np.float32)

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            # Predicion
            m_pred, v_pred = net(x)

            # get even positions corresponding to Z_out
            m_pred = np.ravel(m_pred)[::2]
            v_pred = np.ravel(v_pred)[::2]
            var_obs = flat_m[1::2]  # odd indices var_v

            mu_preds_test.extend(m_pred)
            var_preds_test.extend(np.sqrt(v_pred + var_obs))
            test_obs.extend(y)

        mu_preds_test = np.array(mu_preds_test)
        std_preds_test = np.array(var_preds_test)
        test_obs = np.array(test_obs)

        mu_preds_test = normalizer.unstandardize(
            mu_preds_test, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
        )
        std_preds_test = normalizer.unstandardize_std(
            std_preds_test, train_dtl.x_std[output_col]
        )
        test_obs = normalizer.unstandardize(
            test_obs, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
        )

        # concatenate all predictions
        mu_preds = np.concatenate((mu_preds, mu_preds_val, mu_preds_test), axis=0)
        std_preds = np.concatenate((std_preds, std_preds_val, std_preds_test), axis=0)
        y_true = np.concatenate((train_obs, val_obs, test_obs), axis=0)
        val_start_indices[ts] = int(len(train_obs))
        test_start_indices[ts] = int(val_start_indices[ts] + len(val_obs))

        # save test predicitons for each time series
        mu_preds_padded = np.pad(
            mu_preds.flatten(), (0, horizon_cap - len(mu_preds)), constant_values=np.nan
        )
        std_preds_padded = np.pad(
            std_preds.flatten(),
            (0, horizon_cap - len(std_preds)),
            constant_values=np.nan,
        )
        y_true_padded = np.pad(
            y_true.flatten(), (0, horizon_cap - len(y_true)), constant_values=np.nan
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_true_padded

    np.savetxt(out_dir + "/yPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/yTr.csv", ytestTr, delimiter=",")
    split_indices = np.column_stack((val_start_indices, test_start_indices)).astype(
        np.int32
    )
    np.savetxt(out_dir + "/split_indices.csv", split_indices, fmt="%d", delimiter=",")


def global_model_run(nb_ts, num_epochs, batch_size, seed, early_stopping_criteria):
    """
    Run a single global model across ALL time series using the interleaved dataloader.
    Training/validation: all series at once (round-robin windows).
    Testing: per-series loop preserved (using ts_indices filter) to keep your 1-step recursive logic.
    """

    print("Running global model...")

    # Config
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    net_optim = []
    patience = 10

    # Seed
    manual_seed(seed)

    # --- Output Directory ---
    out_dir = "out/experiment01_global"
    os.makedirs(out_dir, exist_ok=True)

    # Pre-allocate final CSVs
    horizon_cap = 2000
    # create placeholders
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_train_values.csv",
        date_time_file="data/hq/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        scale_covariates=True,
        order_mode="by_series",
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    val_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_val_values.csv",
        date_time_file="data/hq/split_val_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_series",
    )

    # -----------------------
    # Network
    net = Sequential(
        LSTM(input_seq_len + num_features - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )
    if batch_size > 1:
        net.set_threads(8)
    else:
        net.set_threads(1)
    out_updater = OutputUpdater(net.device)

    # Training
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    states_optim = None
    look_back_buffer_optim = None
    train_mses = []

    for epoch in pbar:
        mu_preds = [[] for _ in range(nb_ts)]
        std_preds = [[] for _ in range(nb_ts)]
        train_obs = [[] for _ in range(nb_ts)]

        batch_iter = train_dtl.create_data_loader(
            batch_size=batch_size,
            shuffle=False,
            include_ids=True,
            shuffle_series_blocks=(
                True if train_dtl.order_mode == "by_series" else False
            ),
        )

        # define the lookback buffer for recursive prediction
        look_back_buffer = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        lstm_states = [None] * nb_ts  # placeholder for LSTM states per series
        ts_i = -1  # current time series id

        for x, y, ts_id, _ in batch_iter:

            # reset LSTM states
            # if train_dtl.order_mode == "by_window" or ts_i != ts_id:
            #     reset_lstm_states(net)

            ts_i = copy.copy(ts_id)

            y = np.concatenate(y, axis=0).astype(np.float32)

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))
            B = int(len(ts_idx))  # current batch size

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            if np.isnan(look_back_buffer[ts_idx]).all():
                look_back_buffer[ts_idx] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer[ts_idx]  # update input sequence
                x = x.astype(np.float32)

            flat_x = np.concatenate(x, axis=0)

            # Forward
            m_pred, v_pred = net(flat_x)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # store lstm states
            lstm_states[int(ts_id[0])] = net.get_lstm_states()

            train_mses.append(metric.mse(m_pred, y))

            # Update output layer
            out_updater.update_heteros(
                output_states=net.output_z_buffer,
                mu_obs=y.flatten(),
                delta_states=net.input_delta_z_buffer,
            )

            # Backward + step
            net.backward()
            net.step()

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            m_pred, std_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # Unstadardize
            mu_s = np.asarray(global_mean)[ts_idx].reshape(B, 1)
            sd_s = np.asarray(global_std)[ts_idx].reshape(B, 1)
            scaled_m = normalizer.unstandardize(m_prior.reshape(B, -1), mu_s, sd_s)
            scaled_std = normalizer.unstandardize_std(std_prior.reshape(B, -1), sd_s)
            scaled_y = normalizer.unstandardize(y.reshape(B, -1), mu_s, sd_s)

            # extend the correct series
            for b in range(B):
                s = ts_idx[b]
                mu_preds[s].extend(np.asarray(scaled_m[b]).ravel().tolist())
                std_preds[s].extend(np.asarray(scaled_std[b]).ravel().tolist())
                train_obs[s].extend(np.asarray(scaled_y[b]).ravel().tolist())

            look_back_buffer = update_look_back_buffer(
                look_back_buffer,
                ts_idx=ts_id,
                m_pred=m_pred,
            )

        train_mse = np.nanmean(train_mses)

        # Validating
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(
            batch_size, shuffle=False, include_ids=True
        )

        val_mu_preds = [[] for _ in range(nb_ts)]
        val_std_preds = [[] for _ in range(nb_ts)]
        val_obs = [[] for _ in range(nb_ts)]

        # define the lookback buffer for recursive prediction
        look_back_buffer_val = copy.copy(look_back_buffer)
        ts_i = -1  # current time series id
        val_mses = []
        val_log_liks = []

        # One-step recursive prediction over the validation stream
        for x, y, ts_id, _ in val_batch_iter:

            # set LSTM states
            net.set_lstm_states(lstm_states[int(ts_id[0])])

            # if val_dtl.order_mode == "by_window" or ts_i != ts_id:
            #     reset_lstm_states(net)

            ts_i = copy.copy(ts_id)

            y = np.concatenate(y, axis=0).astype(np.float32)

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))  # shape: (B,)
            B = int(len(ts_idx))

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            if np.isnan(look_back_buffer_val[ts_idx]).all():
                look_back_buffer_val[ts_idx] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer_val[
                    ts_idx
                ]  # update input sequence
                x = x.astype(np.float32)

            flat_x = np.concatenate(x, axis=0)

            # Predicion
            m_pred, v_pred = net(flat_x)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            lstm_states[int(ts_id[0])] = net.get_lstm_states()

            val_mses.append(metric.mse(m_pred, y))
            val_log_liks.append(
                metric.log_likelihood(prediction=m_prior, observation=y, std=std_prior)
            )

            m_pred, std_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # Unstadardize
            mu_s = np.asarray(global_mean)[ts_idx].reshape(B, 1)
            sd_s = np.asarray(global_std)[ts_idx].reshape(B, 1)
            scaled_m = normalizer.unstandardize(m_prior.reshape(B, -1), mu_s, sd_s)
            scaled_std = normalizer.unstandardize_std(std_prior.reshape(B, -1), sd_s)
            scaled_y = normalizer.unstandardize(y.reshape(B, -1), mu_s, sd_s)

            # extend the correct series
            for b in range(B):
                s = ts_idx[b]
                val_mu_preds[s].extend(np.asarray(scaled_m[b]).ravel().tolist())
                val_std_preds[s].extend(np.asarray(scaled_std[b]).ravel().tolist())
                val_obs[s].extend(np.asarray(scaled_y[b]).ravel().tolist())

            look_back_buffer_val = update_look_back_buffer(
                look_back_buffer_val,
                ts_idx=ts_id,
                m_pred=m_pred,
            )

        # Compute log-likelihood for validation set
        mse_val = np.nanmean(val_mses)
        log_lik_val = np.nanmean(val_log_liks)

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| mse_val: {mse_val:>7.4f} | log_lik_val: {log_lik_val:>7.4f}",
            refresh=True,
        )

        # early-stopping
        if early_stopping_criteria == "mse":
            if float(mse_val) < float(mse_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    copy.deepcopy(mu_preds),
                    copy.deepcopy(std_preds),
                    copy.deepcopy(val_mu_preds),
                    copy.deepcopy(val_std_preds),
                )
                look_back_buffer_optim = copy.deepcopy(look_back_buffer_val)
                lstm_optim_states = copy.deepcopy(lstm_states)
        elif early_stopping_criteria == "log_lik":
            if float(log_lik_val) > float(log_lik_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    copy.deepcopy(mu_preds),
                    copy.deepcopy(std_preds),
                    copy.deepcopy(val_mu_preds),
                    copy.deepcopy(val_std_preds),
                )
                look_back_buffer_optim = copy.deepcopy(look_back_buffer_val)
                lstm_optim_states = copy.deepcopy(lstm_states)
        if int(epoch) - int(epoch_optim) > patience:
            break

    if states_optim is None:
        states_optim = (
            copy.deepcopy(mu_preds),
            copy.deepcopy(std_preds),
            copy.deepcopy(val_mu_preds),
            copy.deepcopy(val_std_preds),
        )
    else:
        mu_preds, std_preds, val_mu_preds, val_std_preds = states_optim

    # Load optimal model
    if net_optim:
        net.load_state_dict(net_optim)

    # Save model
    net.save(os.path.join(out_dir, "param/model.pth"))

    # Testing
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_test_values.csv",
        date_time_file="data/hq/split_test_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
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
    if look_back_buffer_optim is not None:
        look_back_buffer_test = copy.copy(look_back_buffer_optim)
    else:
        look_back_buffer_test = copy.copy(look_back_buffer_val)
    ts_i = -1  # current time series id

    for x, y, ts_id, _ in test_batch_iter:

        # if test_dtl.order_mode == "by_window" or ts_i != ts_id:
        #     reset_lstm_states(net)

        ts_i = copy.copy(ts_id)

        ts_idx_arr = np.atleast_1d(np.asarray(ts_id, dtype=int))
        ts_idx = int(ts_idx_arr[0])

        net.set_lstm_states(lstm_optim_states[ts_idx])

        # replace nans in x with zeros
        x = np.nan_to_num(x, nan=0.0).squeeze(0)

        if np.isnan(look_back_buffer_test[ts_idx]).all():
            look_back_buffer_test[ts_idx] = x[:input_seq_len]
        else:
            x[:input_seq_len] = look_back_buffer_test[ts_idx]
            x = x.astype(np.float32)

        # Prediction
        m_pred, v_pred = net(x)

        flat_m = np.ravel(m_pred)
        flat_v = np.ravel(v_pred)

        m_pred = flat_m[::2]  # even indices
        v_pred = flat_v[::2]  # even indices
        var_obs = flat_m[1::2]  # odd indices var_v

        lstm_optim_states[ts_idx] = net.get_lstm_states()

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
        look_back_buffer_test[ts_idx][:-1] = look_back_buffer_test[ts_idx][1:]
        look_back_buffer_test[ts_idx][-1] = float(np.ravel(m_pred)[-1])

        # look_back_buffer_test = update_look_back_buffer(
        #     look_back_buffer_test,
        #     ts_idx=ts_id,
        #     m_pred=m_pred,
        # )

    # concatenate all predictions over train/val/test for each series
    for ts in range(nb_ts):
        mu_preds_ts = np.concatenate(
            (mu_preds[ts], val_mu_preds[ts], test_mu_preds[ts]), axis=0
        )
        std_preds_ts = np.concatenate(
            (std_preds[ts], val_std_preds[ts], test_std_preds[ts]), axis=0
        )
        y_true_ts = np.concatenate((train_obs[ts], val_obs[ts], test_obs[ts]), axis=0)

        val_start_index = int(len(train_obs[ts]))
        test_start_index = int(val_start_index + len(val_obs[ts]))

        # save test predicitons for each time series
        mu_preds_padded = np.pad(
            mu_preds_ts.flatten(),
            (0, horizon_cap - len(mu_preds_ts)),
            constant_values=np.nan,
        )
        std_preds_padded = np.pad(
            std_preds_ts.flatten(),
            (0, horizon_cap - len(std_preds_ts)),
            constant_values=np.nan,
        )
        y_true_padded = np.pad(
            y_true_ts.flatten(),
            (0, horizon_cap - len(y_true_ts)),
            constant_values=np.nan,
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_true_padded
        val_start_indices[ts] = val_start_index
        test_start_indices[ts] = test_start_index

    np.savetxt(out_dir + "/yPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/yTr.csv", ytestTr, delimiter=",")
    split_indices = np.column_stack((val_start_indices, test_start_indices)).astype(
        np.int32
    )
    np.savetxt(out_dir + "/split_indices.csv", split_indices, fmt="%d", delimiter=",")


def embed_model_run(nb_ts, num_epochs, batch_size, seed, early_stopping_criteria):
    """
    Run a single global model across ALL time series using the interleaved dataloader.
    Training/validation: all series at once (round-robin windows).
    Testing: per-series loop preserved (using ts_indices filter) to keep your 1-step recursive logic.
    """

    print("Running embedding model...")

    # Config
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    net_optim = []
    patience = 10

    # Seed
    manual_seed(seed)

    # Initialize embeddings
    embedding_dim = 10
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim),
        encoding_type="normal",
        seed=seed,
    )

    # --- Output Directory ---
    out_dir = "out/experiment01_embed"
    os.makedirs(out_dir, exist_ok=True)

    # save embeddings at beginning
    if not os.path.exists(out_dir + "/embeddings"):
        os.makedirs(out_dir + "/embeddings", exist_ok=True)
    embeddings.save(os.path.join(out_dir, "embeddings/embeddings_start.npz"))

    # Pre-allocate final CSVs
    horizon_cap = 2000
    # create placeholders
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_train_values.csv",
        date_time_file="data/hq/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        scale_covariates=True,
        order_mode="by_series",
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    val_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_val_values.csv",
        date_time_file="data/hq/split_val_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_series",
    )

    # -----------------------
    # Network
    net = Sequential(
        LSTM(input_seq_len + embedding_dim + num_features - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )
    if batch_size > 1:
        net.set_threads(8)
    else:
        net.set_threads(1)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # Training
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    states_optim = None
    look_back_buffer_optim = None
    train_mses = []

    for epoch in pbar:
        mu_preds = [[] for _ in range(nb_ts)]
        std_preds = [[] for _ in range(nb_ts)]
        train_obs = [[] for _ in range(nb_ts)]

        batch_iter = train_dtl.create_data_loader(
            batch_size=batch_size,
            shuffle=False,
            include_ids=True,
            shuffle_series_blocks=(
                True if train_dtl.order_mode == "by_series" else False
            ),
        )

        # define the lookback buffer for recursive prediction
        look_back_buffer = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        lstm_states = [None] * nb_ts  # placeholder for LSTM states per series
        ts_i = -1  # current time series id

        for x, y, ts_id, _ in batch_iter:

            # reset LSTM states
            # if train_dtl.order_mode == "by_window" or ts_i != ts_id:
            #     reset_lstm_states(net)

            ts_i = copy.copy(ts_id)

            y = np.concatenate(y, axis=0).astype(np.float32)

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))
            B = int(len(ts_idx))  # current batch size

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            if np.isnan(look_back_buffer[ts_idx]).all():
                look_back_buffer[ts_idx] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer[ts_idx]  # update input sequence
                x = x.astype(np.float32)

            # append embeddings to each input in the batch
            embed_mu, embed_var = embeddings(ts_idx)  # shape: (B, embedding_dim)
            x_var = np.zeros_like(x)
            x = np.concatenate(
                (x, embed_mu), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
            x_var = np.concatenate(
                (x_var, embed_var), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_x_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Forward
            m_pred, v_pred = net(flat_x, flat_x_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # store lstm states
            lstm_states[int(ts_id[0])] = net.get_lstm_states()

            train_mses.append(metric.mse(m_pred, y))

            # Update output layer
            out_updater.update_heteros(
                output_states=net.output_z_buffer,
                mu_obs=y.flatten(),
                delta_states=net.input_delta_z_buffer,
            )

            # Backward + step
            net.backward()
            net.step()

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            m_pred, std_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # Unstadardize
            mu_s = np.asarray(global_mean)[ts_idx].reshape(B, 1)
            sd_s = np.asarray(global_std)[ts_idx].reshape(B, 1)
            scaled_m = normalizer.unstandardize(m_prior.reshape(B, -1), mu_s, sd_s)
            scaled_std = normalizer.unstandardize_std(std_prior.reshape(B, -1), sd_s)
            scaled_y = normalizer.unstandardize(y.reshape(B, -1), mu_s, sd_s)

            # extend the correct series
            for b in range(B):
                s = ts_idx[b]
                mu_preds[s].extend(np.asarray(scaled_m[b]).ravel().tolist())
                std_preds[s].extend(np.asarray(scaled_std[b]).ravel().tolist())
                train_obs[s].extend(np.asarray(scaled_y[b]).ravel().tolist())

            look_back_buffer = update_look_back_buffer(
                look_back_buffer,
                ts_idx=ts_id,
                m_pred=m_pred,
            )

            # get updates for embeddings
            mu_delta, var_delta = net.get_input_states()
            mu_delta = mu_delta.reshape(B, -1)
            var_delta = var_delta.reshape(B, -1)

            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var

            embeddings.update(
                ts_idx, x_update[:, -embedding_dim:], var_update[:, -embedding_dim:]
            )

        train_mse = np.nanmean(train_mses)

        # Validating
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(
            batch_size, shuffle=False, include_ids=True
        )

        val_mu_preds = [[] for _ in range(nb_ts)]
        val_std_preds = [[] for _ in range(nb_ts)]
        val_obs = [[] for _ in range(nb_ts)]

        # define the lookback buffer for recursive prediction
        look_back_buffer_val = copy.copy(look_back_buffer)
        ts_i = -1  # current time series id
        val_mses = []
        val_log_liks = []

        # One-step recursive prediction over the validation stream
        for x, y, ts_id, _ in val_batch_iter:

            # set LSTM states
            net.set_lstm_states(lstm_states[int(ts_id[0])])

            # if val_dtl.order_mode == "by_window" or ts_i != ts_id:
            #     reset_lstm_states(net)

            ts_i = copy.copy(ts_id)

            y = np.concatenate(y, axis=0).astype(np.float32)

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))  # shape: (B,)
            B = int(len(ts_idx))

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            if np.isnan(look_back_buffer_val[ts_idx]).all():
                look_back_buffer_val[ts_idx] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer_val[
                    ts_idx
                ]  # update input sequence
                x = x.astype(np.float32)

            embed_mu, embed_var = embeddings(ts_idx)  # shape: (B, embedding_dim)
            x_var = np.zeros_like(x)
            x = np.concatenate(
                (x, embed_mu), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
            x_var = np.concatenate(
                (x_var, embed_var), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_x_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Predicion
            m_pred, v_pred = net(flat_x, flat_x_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            lstm_states[int(ts_id[0])] = net.get_lstm_states()

            val_mses.append(metric.mse(m_pred, y))
            val_log_liks.append(
                metric.log_likelihood(prediction=m_prior, observation=y, std=std_prior)
            )

            m_pred, std_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # Unstadardize
            mu_s = np.asarray(global_mean)[ts_idx].reshape(B, 1)
            sd_s = np.asarray(global_std)[ts_idx].reshape(B, 1)
            scaled_m = normalizer.unstandardize(m_prior.reshape(B, -1), mu_s, sd_s)
            scaled_std = normalizer.unstandardize_std(std_prior.reshape(B, -1), sd_s)
            scaled_y = normalizer.unstandardize(y.reshape(B, -1), mu_s, sd_s)

            # extend the correct series
            for b in range(B):
                s = ts_idx[b]
                val_mu_preds[s].extend(np.asarray(scaled_m[b]).ravel().tolist())
                val_std_preds[s].extend(np.asarray(scaled_std[b]).ravel().tolist())
                val_obs[s].extend(np.asarray(scaled_y[b]).ravel().tolist())

            look_back_buffer_val = update_look_back_buffer(
                look_back_buffer_val,
                ts_idx=ts_id,
                m_pred=m_pred,
            )

        # Compute log-likelihood for validation set
        mse_val = np.nanmean(val_mses)
        log_lik_val = np.nanmean(val_log_liks)

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| mse_val: {mse_val:>7.4f} | log_lik_val: {log_lik_val:>7.4f}",
            refresh=True,
        )

        # early-stopping
        if early_stopping_criteria == "mse":
            if float(mse_val) < float(mse_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    copy.deepcopy(mu_preds),
                    copy.deepcopy(std_preds),
                    copy.deepcopy(val_mu_preds),
                    copy.deepcopy(val_std_preds),
                )
                look_back_buffer_optim = copy.deepcopy(look_back_buffer_val)
                lstm_optim_states = copy.deepcopy(lstm_states)
                embeddings_optim = copy.deepcopy(embeddings)
        elif early_stopping_criteria == "log_lik":
            if float(log_lik_val) > float(log_lik_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    copy.deepcopy(mu_preds),
                    copy.deepcopy(std_preds),
                    copy.deepcopy(val_mu_preds),
                    copy.deepcopy(val_std_preds),
                )
                look_back_buffer_optim = copy.deepcopy(look_back_buffer_val)
                lstm_optim_states = copy.deepcopy(lstm_states)
                embeddings_optim = copy.deepcopy(embeddings)
        if int(epoch) - int(epoch_optim) > patience:
            break

    if states_optim is None:
        states_optim = (
            copy.deepcopy(mu_preds),
            copy.deepcopy(std_preds),
            copy.deepcopy(val_mu_preds),
            copy.deepcopy(val_std_preds),
        )
    else:
        mu_preds, std_preds, val_mu_preds, val_std_preds = states_optim

    # Load optimal model
    if net_optim:
        net.load_state_dict(net_optim)

    # load optimal embeddings
    if embeddings_optim:
        embeddings = embeddings_optim
    embeddings.save(os.path.join(out_dir, "embeddings/embeddings_final.npz"))

    # Save model
    net.save(os.path.join(out_dir, "param/model.pth"))

    # Testing
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_test_values.csv",
        date_time_file="data/hq/split_test_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
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
    if look_back_buffer_optim is not None:
        look_back_buffer_test = copy.copy(look_back_buffer_optim)
    else:
        look_back_buffer_test = copy.copy(look_back_buffer_val)
    ts_i = -1  # current time series id

    for x, y, ts_id, _ in test_batch_iter:

        # if test_dtl.order_mode == "by_window" or ts_i != ts_id:
        #     reset_lstm_states(net)

        ts_i = copy.copy(ts_id)

        ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))
        ts_idx = int(ts_idx[0])  # shape: (B,)

        net.set_lstm_states(lstm_optim_states[ts_idx])

        # replace nans in x with zeros
        x = np.nan_to_num(x, nan=0.0).squeeze(0)

        if np.isnan(look_back_buffer_test[ts_idx]).all():
            look_back_buffer_test[ts_idx] = x[:input_seq_len]
        else:
            x[:input_seq_len] = look_back_buffer_test[ts_idx]
            x = x.astype(np.float32)

        # append embeddings to input
        embed_mu, embed_var = embeddings(ts_idx)  # shape: (1, embedding_dim)
        x_var = np.zeros_like(x)
        x = np.concatenate(
            (x, embed_mu), axis=0
        )  # shape: (input_seq_len + embedding_dim + num_features - 1,)
        x_var = np.concatenate(
            (x_var, embed_var), axis=0
        )  # shape: (input_seq_len + embedding_dim + num_features - 1,)
        x = x.astype(np.float32)
        x_var = x_var.astype(np.float32)

        # Prediction
        m_pred, v_pred = net(x, x_var)

        flat_m = np.ravel(m_pred)
        flat_v = np.ravel(v_pred)

        m_pred = flat_m[::2]  # even indices
        v_pred = flat_v[::2]  # even indices
        var_obs = flat_m[1::2]  # odd indices var_v

        lstm_optim_states[ts_idx] = net.get_lstm_states()

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
        look_back_buffer_test[ts_idx][:-1] = look_back_buffer_test[ts_idx][1:]
        look_back_buffer_test[ts_idx][-1] = float(np.ravel(m_pred)[-1])

    # concatenate all predictions over train/val/test for each series
    for ts in range(nb_ts):
        mu_preds_ts = np.concatenate(
            (mu_preds[ts], val_mu_preds[ts], test_mu_preds[ts]), axis=0
        )
        std_preds_ts = np.concatenate(
            (std_preds[ts], val_std_preds[ts], test_std_preds[ts]), axis=0
        )
        y_true_ts = np.concatenate((train_obs[ts], val_obs[ts], test_obs[ts]), axis=0)

        val_start_index = int(len(train_obs[ts]))
        test_start_index = int(val_start_index + len(val_obs[ts]))

        # save test predicitons for each time series
        mu_preds_padded = np.pad(
            mu_preds_ts.flatten(),
            (0, horizon_cap - len(mu_preds_ts)),
            constant_values=np.nan,
        )
        std_preds_padded = np.pad(
            std_preds_ts.flatten(),
            (0, horizon_cap - len(std_preds_ts)),
            constant_values=np.nan,
        )
        y_true_padded = np.pad(
            y_true_ts.flatten(),
            (0, horizon_cap - len(y_true_ts)),
            constant_values=np.nan,
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_true_padded
        val_start_indices[ts] = val_start_index
        test_start_indices[ts] = test_start_index

    np.savetxt(out_dir + "/yPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/yTr.csv", ytestTr, delimiter=",")
    split_indices = np.column_stack((val_start_indices, test_start_indices)).astype(
        np.int32
    )
    np.savetxt(out_dir + "/split_indices.csv", split_indices, fmt="%d", delimiter=",")


def shared_model_run(nb_ts, num_epochs, batch_size, seed, early_stopping_criteria):
    """
    Run a single global model across ALL time series using the interleaved dataloader.
    Training/validation: all series at once (round-robin windows).
    Testing: per-series loop preserved (using ts_indices filter) to keep your 1-step recursive logic.
    """

    # Config
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    net_optim = []
    patience = 10

    # Seed
    manual_seed(seed)

    # Initialize embeddings
    sub_embedding_dim = 3
    # Dam_embedding
    nb_dams = 6
    dam_embeddings = TimeSeriesEmbeddings(
        (nb_dams, sub_embedding_dim),
        encoding_type="normal",
        seed=seed,
    )
    # Dam_type_embedding
    nb_dam_types = 2
    dam_type_embeddings = TimeSeriesEmbeddings(
        (nb_dam_types, sub_embedding_dim),
        encoding_type="normal",
        seed=seed,
    )
    # Sensor_type_embedding
    nb_sensor_types = 3
    sensor_type_embeddings = TimeSeriesEmbeddings(
        (nb_sensor_types, sub_embedding_dim),
        encoding_type="normal",
        seed=seed,
    )
    # Direction_embedding
    nb_directions = 4
    direction_embeddings = TimeSeriesEmbeddings(
        (nb_directions, sub_embedding_dim),
        encoding_type="normal",
        seed=seed,
    )
    # Sensor_embedding
    nb_sensors = nb_ts
    sensor_embeddings = TimeSeriesEmbeddings(
        (nb_sensors, sub_embedding_dim),
        encoding_type="normal",
        seed=seed,
    )

    embedding_dim = (
        sub_embedding_dim * 5
    )  # total embedding dimension after concatenation

    # read embeddings mappings
    embedding_map = pd.read_csv("data/hq/ts_embedding_map.csv")
    ts_ids = embedding_map["ts_id"].values
    dam_ids = embedding_map["dam_id"].values
    dam_type_ids = embedding_map["dam_type_id"].values
    sensor_type_ids = embedding_map["sensor_type_id"].values
    direction_ids = embedding_map["direction_id"].values
    sensor_ids = embedding_map["sensor_id"].values

    # Build a single mapping dict where each ts_id maps to all embedding ids
    embedding_id_map = dict(
        zip(
            ts_ids,
            zip(
                dam_ids,
                dam_type_ids,
                sensor_type_ids,
                direction_ids,
                sensor_ids,
            ),
        )
    )

    # --- Output Directory ---
    out_dir = "out/experiment01_shared"
    os.makedirs(out_dir, exist_ok=True)

    # save embeddings at beginning
    if not os.path.exists(out_dir + "/embeddings"):
        os.makedirs(out_dir + "/embeddings", exist_ok=True)
    dam_embeddings.save(os.path.join(out_dir, "embeddings/dam_embeddings_start.npz"))
    dam_type_embeddings.save(
        os.path.join(out_dir, "embeddings/dam_type_embeddings_start.npz")
    )
    sensor_type_embeddings.save(
        os.path.join(out_dir, "embeddings/sensor_type_embeddings_start.npz")
    )
    direction_embeddings.save(
        os.path.join(out_dir, "embeddings/direction_embeddings_start.npz")
    )
    sensor_embeddings.save(
        os.path.join(out_dir, "embeddings/sensor_embeddings_start.npz")
    )

    # Pre-allocate final CSVs
    horizon_cap = 2000
    # create placeholders
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_train_values.csv",
        date_time_file="data/hq/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        scale_covariates=True,
        order_mode="by_series",
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    val_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_val_values.csv",
        date_time_file="data/hq/split_val_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_series",
    )

    # -----------------------
    # Network
    net = Sequential(
        LSTM(input_seq_len + embedding_dim + num_features - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )
    if batch_size > 1:
        net.set_threads(8)
    else:
        net.set_threads(1)
    out_updater = OutputUpdater(net.device)
    net.input_state_update = True

    # Training
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    states_optim = None
    look_back_buffer_optim = None
    train_mses = []

    for epoch in pbar:
        mu_preds = [[] for _ in range(nb_ts)]
        std_preds = [[] for _ in range(nb_ts)]
        train_obs = [[] for _ in range(nb_ts)]

        batch_iter = train_dtl.create_data_loader(
            batch_size=batch_size,
            shuffle=False,
            include_ids=True,
            shuffle_series_blocks=(
                True if train_dtl.order_mode == "by_series" else False
            ),
        )

        # define the lookback buffer for recursive prediction
        look_back_buffer = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        lstm_states = [None] * nb_ts  # placeholder for LSTM states per series
        ts_i = -1  # current time series id

        for x, y, ts_id, _ in batch_iter:

            # reset LSTM states
            # if train_dtl.order_mode == "by_window" or ts_i != ts_id:
            #     reset_lstm_states(net)

            ts_i = copy.copy(ts_id)

            y = np.concatenate(y, axis=0).astype(np.float32)

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))
            B = int(len(ts_idx))  # current batch size

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            if np.isnan(look_back_buffer[ts_idx]).all():
                look_back_buffer[ts_idx] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer[ts_idx]  # update input sequence
                x = x.astype(np.float32)

            # build embedding
            embed_mu, embed_var = get_combined_embeddings(
                ts_idx,
                embedding_id_map,
                dam_embeddings,
                dam_type_embeddings,
                sensor_type_embeddings,
                direction_embeddings,
                sensor_embeddings,
            )  # shape: (B, embedding_dim)
            x_var = np.zeros_like(x)
            x = np.concatenate(
                (x, embed_mu), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
            x_var = np.concatenate(
                (x_var, embed_var), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_x_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Forward
            m_pred, v_pred = net(flat_x, flat_x_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # store lstm states
            lstm_states[int(ts_id[0])] = net.get_lstm_states()

            train_mses.append(metric.mse(m_pred, y))

            # Update output layer
            out_updater.update_heteros(
                output_states=net.output_z_buffer,
                mu_obs=y.flatten(),
                delta_states=net.input_delta_z_buffer,
            )

            # Backward + step
            net.backward()
            net.step()

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            m_pred, std_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # Unstadardize
            mu_s = np.asarray(global_mean)[ts_idx].reshape(B, 1)
            sd_s = np.asarray(global_std)[ts_idx].reshape(B, 1)
            scaled_m = normalizer.unstandardize(m_prior.reshape(B, -1), mu_s, sd_s)
            scaled_std = normalizer.unstandardize_std(std_prior.reshape(B, -1), sd_s)
            scaled_y = normalizer.unstandardize(y.reshape(B, -1), mu_s, sd_s)

            # extend the correct series
            for b in range(B):
                s = ts_idx[b]
                mu_preds[s].extend(np.asarray(scaled_m[b]).ravel().tolist())
                std_preds[s].extend(np.asarray(scaled_std[b]).ravel().tolist())
                train_obs[s].extend(np.asarray(scaled_y[b]).ravel().tolist())

            look_back_buffer = update_look_back_buffer(
                look_back_buffer,
                ts_idx=ts_id,
                m_pred=m_pred,
            )

            # get updates for embeddings
            mu_delta, var_delta = net.get_input_states()
            mu_delta = mu_delta.reshape(B, -1)
            var_delta = var_delta.reshape(B, -1)

            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var

            update_combined_embeddings(
                ts_idx,
                x_update[:, -embedding_dim:],
                var_update[:, -embedding_dim:],
                embedding_id_map,
                dam_embeddings,
                dam_type_embeddings,
                sensor_type_embeddings,
                direction_embeddings,
                sensor_embeddings,
            )

        train_mse = np.nanmean(train_mses)

        # Validating
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(
            batch_size, shuffle=False, include_ids=True
        )

        val_mu_preds = [[] for _ in range(nb_ts)]
        val_std_preds = [[] for _ in range(nb_ts)]
        val_obs = [[] for _ in range(nb_ts)]

        # define the lookback buffer for recursive prediction
        look_back_buffer_val = copy.copy(look_back_buffer)
        ts_i = -1  # current time series id
        val_mses = []
        val_log_liks = []

        # One-step recursive prediction over the validation stream
        for x, y, ts_id, _ in val_batch_iter:

            # set LSTM states
            net.set_lstm_states(lstm_states[int(ts_id[0])])

            # if val_dtl.order_mode == "by_window" or ts_i != ts_id:
            #     reset_lstm_states(net)

            ts_i = copy.copy(ts_id)

            y = np.concatenate(y, axis=0).astype(np.float32)

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))  # shape: (B,)
            B = int(len(ts_idx))

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            if np.isnan(look_back_buffer_val[ts_idx]).all():
                look_back_buffer_val[ts_idx] = x[:, :input_seq_len]
            else:
                x[:, :input_seq_len] = look_back_buffer_val[
                    ts_idx
                ]  # update input sequence
                x = x.astype(np.float32)

            embed_mu, embed_var = get_combined_embeddings(
                ts_idx,
                embedding_id_map,
                dam_embeddings,
                dam_type_embeddings,
                sensor_type_embeddings,
                direction_embeddings,
                sensor_embeddings,
            )  # shape: (B, embedding_dim)
            x_var = np.zeros_like(x)
            x = np.concatenate(
                (x, embed_mu), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
            x_var = np.concatenate(
                (x_var, embed_var), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_x_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Predicion
            m_pred, v_pred = net(flat_x, flat_x_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            lstm_states[int(ts_id[0])] = net.get_lstm_states()

            val_mses.append(metric.mse(m_pred, y))
            val_log_liks.append(
                metric.log_likelihood(prediction=m_prior, observation=y, std=std_prior)
            )

            m_pred, std_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # Unstadardize
            mu_s = np.asarray(global_mean)[ts_idx].reshape(B, 1)
            sd_s = np.asarray(global_std)[ts_idx].reshape(B, 1)
            scaled_m = normalizer.unstandardize(m_prior.reshape(B, -1), mu_s, sd_s)
            scaled_std = normalizer.unstandardize_std(std_prior.reshape(B, -1), sd_s)
            scaled_y = normalizer.unstandardize(y.reshape(B, -1), mu_s, sd_s)

            # extend the correct series
            for b in range(B):
                s = ts_idx[b]
                val_mu_preds[s].extend(np.asarray(scaled_m[b]).ravel().tolist())
                val_std_preds[s].extend(np.asarray(scaled_std[b]).ravel().tolist())
                val_obs[s].extend(np.asarray(scaled_y[b]).ravel().tolist())

            look_back_buffer_val = update_look_back_buffer(
                look_back_buffer_val,
                ts_idx=ts_id,
                m_pred=m_pred,
            )

        # Compute log-likelihood for validation set
        mse_val = np.nanmean(val_mses)
        log_lik_val = np.nanmean(val_log_liks)

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| mse_val: {mse_val:>7.4f} | log_lik_val: {log_lik_val:>7.4f}",
            refresh=True,
        )

        # early-stopping
        if early_stopping_criteria == "mse":
            if float(mse_val) < float(mse_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    copy.deepcopy(mu_preds),
                    copy.deepcopy(std_preds),
                    copy.deepcopy(val_mu_preds),
                    copy.deepcopy(val_std_preds),
                )
                look_back_buffer_optim = copy.deepcopy(look_back_buffer_val)
                lstm_optim_states = copy.deepcopy(lstm_states)
                # save a copy of embeddings
                embeddings_optim = copy.deepcopy(
                    (
                        dam_embeddings,
                        dam_type_embeddings,
                        sensor_type_embeddings,
                        direction_embeddings,
                        sensor_embeddings,
                    )
                )
        elif early_stopping_criteria == "log_lik":
            if float(log_lik_val) > float(log_lik_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    copy.deepcopy(mu_preds),
                    copy.deepcopy(std_preds),
                    copy.deepcopy(val_mu_preds),
                    copy.deepcopy(val_std_preds),
                )
                look_back_buffer_optim = copy.deepcopy(look_back_buffer_val)
                lstm_optim_states = copy.deepcopy(lstm_states)
                embeddings_optim = copy.deepcopy(
                    (
                        dam_embeddings,
                        dam_type_embeddings,
                        sensor_type_embeddings,
                        direction_embeddings,
                        sensor_embeddings,
                    )
                )
        if int(epoch) - int(epoch_optim) > patience:
            break

    if states_optim is None:
        states_optim = (
            copy.deepcopy(mu_preds),
            copy.deepcopy(std_preds),
            copy.deepcopy(val_mu_preds),
            copy.deepcopy(val_std_preds),
        )
    else:
        mu_preds, std_preds, val_mu_preds, val_std_preds = states_optim

    # Load optimal model
    if net_optim:
        net.load_state_dict(net_optim)

    # load optimal embeddings
    if embeddings_optim:
        (
            dam_embeddings,
            dam_type_embeddings,
            sensor_type_embeddings,
            direction_embeddings,
            sensor_embeddings,
        ) = embeddings_optim
    dam_embeddings.save(os.path.join(out_dir, "embeddings/dam_embeddings_final.npz"))
    dam_type_embeddings.save(
        os.path.join(out_dir, "embeddings/dam_type_embeddings_final.npz")
    )
    sensor_type_embeddings.save(
        os.path.join(out_dir, "embeddings/sensor_type_embeddings_final.npz")
    )
    direction_embeddings.save(
        os.path.join(out_dir, "embeddings/direction_embeddings_final.npz")
    )
    sensor_embeddings.save(
        os.path.join(out_dir, "embeddings/sensor_embeddings_final.npz")
    )

    # Save model
    net.save(os.path.join(out_dir, "param/model.pth"))

    # Testing
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloaderV2(
        x_file="data/hq/split_test_values.csv",
        date_time_file="data/hq/split_test_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
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
    if look_back_buffer_optim is not None:
        look_back_buffer_test = copy.copy(look_back_buffer_optim)
    else:
        look_back_buffer_test = copy.copy(look_back_buffer_val)
    ts_i = -1  # current time series id

    for x, y, ts_id, _ in test_batch_iter:

        # if test_dtl.order_mode == "by_window" or ts_i != ts_id:
        #     reset_lstm_states(net)

        ts_i = copy.copy(ts_id)

        ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))
        ts_idx = int(ts_idx[0])  # shape: (B,)

        net.set_lstm_states(lstm_optim_states[ts_idx])

        # replace nans in x with zeros
        x = np.nan_to_num(x, nan=0.0).squeeze(0)

        if np.isnan(look_back_buffer_test[ts_idx]).all():
            look_back_buffer_test[ts_idx] = x[:input_seq_len]
        else:
            x[:input_seq_len] = look_back_buffer_test[ts_idx]
            x = x.astype(np.float32)

        # append embeddings to input
        embed_mu, embed_var = get_combined_embeddings(
            [ts_idx],
            embedding_id_map,
            dam_embeddings,
            dam_type_embeddings,
            sensor_type_embeddings,
            direction_embeddings,
            sensor_embeddings,
        )  # shape: (1, embedding_dim)
        embed_mu = embed_mu[0]
        embed_var = embed_var[0]
        x_var = np.zeros_like(x)
        x = np.concatenate(
            (x, embed_mu), axis=0
        )  # shape: (input_seq_len + embedding_dim + num_features - 1,)
        x_var = np.concatenate(
            (x_var, embed_var), axis=0
        )  # shape: (input_seq_len + embedding_dim + num_features - 1,)
        x = x.astype(np.float32)
        x_var = x_var.astype(np.float32)

        # Prediction
        m_pred, v_pred = net(x, x_var)

        flat_m = np.ravel(m_pred)
        flat_v = np.ravel(v_pred)

        m_pred = flat_m[::2]  # even indices
        v_pred = flat_v[::2]  # even indices
        var_obs = flat_m[1::2]  # odd indices var_v

        lstm_optim_states[ts_idx] = net.get_lstm_states()

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
        look_back_buffer_test[ts_idx][:-1] = look_back_buffer_test[ts_idx][1:]
        look_back_buffer_test[ts_idx][-1] = float(np.ravel(m_pred)[-1])

    # concatenate all predictions over train/val/test for each series
    for ts in range(nb_ts):
        mu_preds_ts = np.concatenate(
            (mu_preds[ts], val_mu_preds[ts], test_mu_preds[ts]), axis=0
        )
        std_preds_ts = np.concatenate(
            (std_preds[ts], val_std_preds[ts], test_std_preds[ts]), axis=0
        )
        y_true_ts = np.concatenate((train_obs[ts], val_obs[ts], test_obs[ts]), axis=0)

        val_start_index = int(len(train_obs[ts]))
        test_start_index = int(val_start_index + len(val_obs[ts]))

        # save test predicitons for each time series
        mu_preds_padded = np.pad(
            mu_preds_ts.flatten(),
            (0, horizon_cap - len(mu_preds_ts)),
            constant_values=np.nan,
        )
        std_preds_padded = np.pad(
            std_preds_ts.flatten(),
            (0, horizon_cap - len(std_preds_ts)),
            constant_values=np.nan,
        )
        y_true_padded = np.pad(
            y_true_ts.flatten(),
            (0, horizon_cap - len(y_true_ts)),
            constant_values=np.nan,
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_true_padded
        val_start_indices[ts] = val_start_index
        test_start_indices[ts] = test_start_index

    np.savetxt(out_dir + "/yPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/yTr.csv", ytestTr, delimiter=",")
    split_indices = np.column_stack((val_start_indices, test_start_indices)).astype(
        np.int32
    )
    np.savetxt(out_dir + "/split_indices.csv", split_indices, fmt="%d", delimiter=",")


def main(
    nb_ts=127,
    num_epochs=100,
    batch_size=1,
    seed=1,
    early_stopping_criteria="log_lik",
    experiments=["shared"],
):
    """
    Main function to run all experiments on time series
    """
    available_experiments = ("local", "global", "embed", "shared")

    if experiments is None:
        parser = argparse.ArgumentParser(
            description="Run HQ experiment 01 variants.",
            allow_abbrev=False,
        )
        parser.add_argument(
            "--experiments",
            "-e",
            nargs="+",
            choices=(*available_experiments, "all"),
            default=["local"],
            help="Experiments to run; use 'all' to run every available variant.",
        )
        args, _ = parser.parse_known_args()
        experiments = args.experiments
    else:
        experiments = [exp.lower() for exp in experiments]

    selected_experiments = set()
    for exp in experiments:
        if exp == "all":
            selected_experiments.update(available_experiments)
        else:
            selected_experiments.add(exp)

    unknown_experiments = selected_experiments.difference(available_experiments)
    if unknown_experiments:
        raise ValueError(
            "Unknown experiments requested: " + ", ".join(sorted(unknown_experiments))
        )

    if not selected_experiments:
        selected_experiments.add("local")

    # Run1 --> local model
    if "local" in selected_experiments:
        try:
            local_model_run(
                nb_ts,
                num_epochs,
                seed=seed,
                early_stopping_criteria=early_stopping_criteria,
                batch_size=1,  # local model does not need batching
            )
        except Exception as e:
            print(f"Local model run failed: {e}")

    # Run2 --> global model
    if "global" in selected_experiments:
        try:
            global_model_run(
                nb_ts,
                num_epochs,
                batch_size,
                seed,
                early_stopping_criteria,
            )
        except Exception as e:
            print(f"Global model run failed: {e}")

    # Run3 --> global model with embeddings
    if "embed" in selected_experiments:
        try:
            embed_model_run(
                nb_ts,
                num_epochs,
                batch_size,
                seed,
                early_stopping_criteria,
            )
        except Exception as e:
            print(f"Embed model run failed: {e}")

    # Run4 --> global model with shared sub-embeddings
    if "shared" in selected_experiments:
        try:
            shared_model_run(
                nb_ts,
                num_epochs,
                batch_size,
                seed,
                early_stopping_criteria,
            )
        except Exception as e:
            print(f"Shared model run failed: {e}")


if __name__ == "__main__":
    main()

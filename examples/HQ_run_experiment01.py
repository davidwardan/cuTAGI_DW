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
)
from pytagi import manual_seed
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


# TODO: this needs to handle batches update where y is not nan for some samples and for other return pred
def calculate_gaussian_posterior(m_pred, v_pred, y, var_obs):

    # get the indices where y is nan
    nan_indices = np.isnan(y)
    y = np.where(nan_indices, m_pred, y)
    var_obs = np.where(nan_indices, 0.0, var_obs)

    # kalman update
    K = v_pred / (v_pred + var_obs)  # Kalman gain
    m_post = m_pred + K * (y - m_pred)  # posterior mean
    v_post = (1.0 - K) * v_pred  # posterior variance

    return m_post.astype(np.float32), v_post.astype(np.float32)


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
    # TODO: not the best way to reset states, but works for now
    try:
        net.reset_lstm_states()
    except:
        pass
    ts_indices = [int(sid) for sid in ts_ids]
    base_states = net.get_lstm_states()

    if isinstance(lstm_states, dict):
        cached_states = {idx: lstm_states.get(idx) for idx in ts_indices}
    else:
        cached_states = {}
        lstm_len = len(lstm_states) if hasattr(lstm_states, "__len__") else 0
        for idx in ts_indices:
            cached_states[idx] = lstm_states[idx] if 0 <= idx < lstm_len else None

    batch_size = len(ts_indices)
    merged_states: dict[
        int, tuple[list[float], list[float], list[float], list[float]]
    ] = {}

    for layer_idx, layer_state in base_states.items():
        base_mu_h, base_var_h, base_mu_c, base_var_c = (
            list(arr) for arr in layer_state
        )
        total = len(base_mu_h)
        if total == 0:
            merged_states[layer_idx] = (
                base_mu_h,
                base_var_h,
                base_mu_c,
                base_var_c,
            )
            continue

        for batch_pos, series_idx in enumerate(ts_indices):
            stored_state = cached_states.get(series_idx)
            if not stored_state:
                continue
            if isinstance(stored_state, dict):
                series_layer_state = stored_state.get(layer_idx)
            else:
                try:
                    series_layer_state = stored_state[layer_idx]
                except (IndexError, KeyError, TypeError):
                    series_layer_state = None
            if not series_layer_state:
                continue

            mu_h, var_h, mu_c, var_c = series_layer_state
            state_len = len(mu_h)
            if batch_size > 1 and state_len and state_len * batch_size == total:
                start = batch_pos * state_len
                end = start + state_len
                base_mu_h[start:end] = mu_h
                base_var_h[start:end] = var_h
                base_mu_c[start:end] = mu_c
                base_var_c[start:end] = var_c
            else:
                base_mu_h = list(mu_h)
                base_var_h = list(var_h)
                base_mu_c = list(mu_c)
                base_var_c = list(var_c)
                break

        merged_states[layer_idx] = (
            base_mu_h,
            base_var_h,
            base_mu_c,
            base_var_c,
        )

    if merged_states:
        net.set_lstm_states(merged_states)


def local_model_run(
    nb_ts, num_epochs, batch_size, seed, early_stopping_criteria, train_size
):
    """ "
    Runs a seperate local model for each time series in the dataset
    """

    print("Running local models...")
    ts_idx = np.arange(0, nb_ts)

    # --- Output Directory ---
    out_dir = "out/experiment01_local"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # create placeholders
    horizon_cap = 2000  # manually set
    ytestPd = np.full((horizon_cap, nb_ts), np.nan)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # --- Load Data ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    for ts in pbar:

        # Config
        output_col = [0]
        num_features = 2
        input_seq_len = 52
        output_seq_len = 1
        seq_stride = 1

        # early-stopping trackers
        log_lik_optim = -1e100
        mse_optim = 1e100
        epoch_optim = 0
        net_optim = []
        patience = 10  # epochs to wait for improvement before early stopping
        min_epochs = 0  # minimum number of epochs before early stopping
        have_best = False

        train_dtl = TimeSeriesDataloader(
            x_file=f"data/hq/train_{train_size}/split_train_values.csv",
            date_time_file=f"data/hq/train_{train_size}/split_train_datetimes.csv",
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
        manual_seed(seed)
        net = Sequential(
            LSTM(num_features + input_seq_len - 1, 40, 1),
            LSTM(40, 40, 1),
            Linear(40, 2),
            EvenExp(),
        )
        net.to_device("cuda")
        # net.set_threads(1)  # faster for batch size 1
        out_updater = OutputUpdater(net.device)

        # define placeholders for optimal states
        look_back_buffer_val_mu_optim = None
        look_back_buffer_val_var_optim = None
        states_optim = None
        lstm_optim_states = None

        # --- Training ---
        pbar = tqdm(range(num_epochs), desc="Training Progress")

        # save for plotting
        train_mses = []
        train_log_liks = []
        val_mses = []
        val_log_liks = []

        for epoch in pbar:

            # define placeholders for predictions and observations
            mu_preds = []
            std_preds = []
            train_obs = []

            # define the batch iterator
            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

            # define the lookback buffer for recursive prediction
            look_back_buffer_mu = np.full(input_seq_len, np.nan, dtype=np.float32)
            look_back_buffer_var = np.full(input_seq_len, 0.0, dtype=np.float32)

            # Always start with zeroed LSTM states for each epoch
            if (
                epoch != 0
            ):  # TODO: this results in error if reset is called before any forward pass
                net.reset_lstm_states()

            for x, y in batch_iter:

                x = np.nan_to_num(x, nan=0.0)  # clean input from nans
                x_var = np.zeros_like(x)  # make sure covariates have zero variance

                # intialize look back buffer if first step
                if np.isnan(look_back_buffer_mu).all():
                    look_back_buffer_mu = x[:input_seq_len]
                else:
                    x[:input_seq_len] = look_back_buffer_mu  # update input sequence
                    x = x.astype(np.float32)

                # no nans in var buffer, insteatd uses 1.0
                x_var[:input_seq_len] = look_back_buffer_var
                x_var = x_var.astype(np.float32)

                # Feed forward
                m_pred, v_pred = net(x, x_var)

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

                mu_preds.extend(m_pred)
                std_preds.extend(np.sqrt(v_pred + var_obs))  # epistemic + aleatoric
                train_obs.extend(y)

                # get posterior states for look back buffer
                m_pred, v_pred = calculate_gaussian_posterior(
                    m_pred, v_pred, y, var_obs
                )

                # update look back
                look_back_buffer_mu[:-1] = look_back_buffer_mu[1:]  # shift left
                look_back_buffer_var[:-1] = look_back_buffer_var[1:]
                look_back_buffer_mu[-1] = float(np.ravel(m_pred)[-1])
                look_back_buffer_var[-1] = float(np.ravel(v_pred + var_obs)[-1])

            mu_preds = np.array(mu_preds)
            std_preds = np.array(std_preds)
            train_obs = np.array(train_obs)

            # get train metrics
            train_mse = metric.mse(mu_preds, train_obs)
            train_log_lik = metric.log_likelihood(
                prediction=mu_preds, observation=train_obs, std=std_preds
            )
            train_mses.append(train_mse)
            train_log_liks.append(train_log_lik)

            # Unstandardize
            mu_preds = normalizer.unstandardize(
                mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            std_preds = normalizer.unstandardize_std(
                std_preds, train_dtl.x_std[output_col]
            )
            train_obs = normalizer.unstandardize(
                train_obs, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )

            # --- Validation ---
            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds_val = []
            std_preds_val = []
            val_obs = []

            # use posterior states from training as lookback buffer
            look_back_buffer_val_mu = np.copy(look_back_buffer_mu)
            look_back_buffer_val_var = np.copy(look_back_buffer_var)

            # One-step recursive prediction over the validation stream
            for x, y in val_batch_iter:

                x_var = np.zeros_like(x)  # make sure covariates have zero variance

                # insert values from lookback_buffer
                x[:input_seq_len] = look_back_buffer_val_mu  # update input sequence
                x_var[:input_seq_len] = look_back_buffer_val_var
                x = x.astype(np.float32)
                x_var = x_var.astype(np.float32)

                # Predicion
                m_pred, v_pred = net(x, x_var)

                flat_m = np.ravel(m_pred)
                flat_v = np.ravel(v_pred)

                # get even positions corresponding to Z_out
                m_pred = flat_m[::2]
                v_pred = flat_v[::2]
                var_obs = flat_m[1::2]  # odd indices var_v

                mu_preds_val.extend(m_pred)
                std_preds_val.extend(np.sqrt(v_pred + var_obs))  # epistemic + aleatoric
                val_obs.extend(y)

                # get posterior states for look back buffer
                m_pred, v_pred = calculate_gaussian_posterior(
                    m_pred, v_pred, y, var_obs
                )

                # update look back
                look_back_buffer_val_mu[:-1] = look_back_buffer_val_mu[1:]  # shift left
                look_back_buffer_val_var[:-1] = look_back_buffer_val_var[1:]
                look_back_buffer_val_mu[-1] = float(np.ravel(m_pred)[-1])
                look_back_buffer_val_var[-1] = float(np.ravel(v_pred + var_obs)[-1])

            mu_preds_val = np.array(mu_preds_val)
            std_preds_val = np.array(std_preds_val)
            val_obs = np.array(val_obs)

            # get train metrics
            val_mse = metric.mse(mu_preds_val, val_obs)
            val_log_lik = metric.log_likelihood(
                prediction=mu_preds_val, observation=val_obs, std=std_preds_val
            )
            val_mses.append(val_mse)
            val_log_liks.append(val_log_lik)

            # Unstandardize
            mu_preds_val = normalizer.unstandardize(
                mu_preds_val, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            std_preds_val = normalizer.unstandardize_std(
                std_preds_val, train_dtl.x_std[output_col]
            )

            val_obs = normalizer.unstandardize(
                val_obs, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )

            # Progress bar
            pbar.set_description(
                f"Ts #{ts+1}/{nb_ts} | Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| lg_lik: {train_log_lik:>7.4f}| mse_val: {val_mse:>7.4f} | log_lk_val: {val_log_lik:>7.4f}",
                refresh=True,
            )

            # check if warmup period is done
            warmup_done = (epoch + 1) >= min_epochs

            # early-stopping
            if early_stopping_criteria == "mse":
                if warmup_done and float(val_mse) < float(mse_optim):
                    have_best = True
                    mse_optim = val_mse
                    log_lik_optim = val_log_lik
                    epoch_optim = epoch
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        mu_preds_val,
                        std_preds_val,
                    )
                    look_back_buffer_val_mu_optim = np.copy(look_back_buffer_val_mu)
                    look_back_buffer_val_var_optim = np.copy(look_back_buffer_val_var)
                    lstm_optim_states = copy.deepcopy(net.get_lstm_states())
            elif early_stopping_criteria == "log_lik":
                if warmup_done and float(val_log_lik) > float(log_lik_optim):
                    have_best = True
                    mse_optim = val_mse
                    log_lik_optim = val_log_lik
                    epoch_optim = epoch
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        mu_preds_val,
                        std_preds_val,
                    )
                    look_back_buffer_val_mu_optim = np.copy(look_back_buffer_val_mu)
                    look_back_buffer_val_var_optim = np.copy(look_back_buffer_val_var)
                    lstm_optim_states = copy.deepcopy(net.get_lstm_states())

            if warmup_done:
                last_improvement = epoch_optim if have_best else (min_epochs - 1)
                if epoch - last_improvement >= patience:
                    if not have_best:
                        net_optim = net.state_dict()
                        states_optim = (
                            mu_preds,
                            std_preds,
                            mu_preds_val,
                            std_preds_val,
                        )
                        lstm_optim_states = copy.deepcopy(net.get_lstm_states())
                        look_back_buffer_val_mu_optim = np.copy(look_back_buffer_val_mu)
                        look_back_buffer_val_var_optim = np.copy(
                            look_back_buffer_val_var
                        )
                    break

        # reset to optimal model
        if net_optim:
            net.load_state_dict(net_optim)
        net.save(out_dir + "/param/model_{}.pth".format(str(ts)))

        # save training/validation metrics
        metrics = {
            "train_mse": np.array(train_mses),
            "train_log_lik": np.array(train_log_liks),
            "val_mse": np.array(val_mses),
            "val_log_lik": np.array(val_log_liks),
            "epoch_optim": epoch_optim + 1,
        }
        if not os.path.exists(out_dir + "/train_metrics"):
            os.makedirs(out_dir + "/train_metrics")
        np.savez(out_dir + "/train_metrics/metrics_{}.npz".format(str(ts)), **metrics)

        # load optimal lstm states
        net.set_lstm_states(lstm_optim_states)

        # unpack optimal predictions
        mu_preds, std_preds, mu_preds_val, std_preds_val = states_optim

        # -- Testing --
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds_test = []
        var_preds_test = []
        test_obs = []

        # use posterior states from validation as lookback buffer
        look_back_buffer_test_mu = np.copy(look_back_buffer_val_mu_optim)
        look_back_buffer_test_var = np.copy(look_back_buffer_val_var_optim)

        # One-step recursive prediction
        for x, y in test_batch_iter:

            x_var = np.zeros_like(x)  # make sure covariates have zero variance

            # insert values from lookback_buffer
            x[:input_seq_len] = look_back_buffer_test_mu
            x_var[:input_seq_len] = look_back_buffer_test_var
            x = x.astype(np.float32)
            x_var = x_var.astype(np.float32)

            # Predicion
            m_pred, v_pred = net(x, x_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            # get even positions corresponding to Z_out
            m_pred = flat_m[::2]
            v_pred = flat_v[::2]
            var_obs = flat_m[1::2]  # odd indices var_v

            mu_preds_test.extend(m_pred)
            var_preds_test.extend(np.sqrt(v_pred + var_obs))
            test_obs.extend(y)

            # update look back buffer
            look_back_buffer_test_mu[:-1] = look_back_buffer_test_mu[1:]
            look_back_buffer_test_mu[-1] = float(np.ravel(m_pred)[-1])
            look_back_buffer_test_var[:-1] = look_back_buffer_test_var[1:]
            look_back_buffer_test_var[-1] = float(np.ravel(v_pred + var_obs)[-1])

        mu_preds_test = np.array(mu_preds_test)
        std_preds_test = np.array(var_preds_test)
        test_obs = np.array(test_obs)

        # Unstandardize
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


def global_model_run(
    nb_ts, num_epochs, batch_size, seed, early_stopping_criteria, train_size
):
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

    # early-stopping trackers
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    net_optim = []
    patience = 10  # epochs to wait for improvement before early stopping
    min_epochs = 15  # minimum number of epochs before early stopping
    have_best = False

    # --- Output Directory ---
    out_dir = "out/experiment01_global"
    os.makedirs(out_dir, exist_ok=True)

    # Pre-allocate final CSVs
    horizon_cap = 2000  # manually set
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalTimeSeriesDataloader(
        x_file=f"data/hq/train_{train_size}/split_train_values.csv",
        date_time_file=f"data/hq/train_{train_size}/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        order_mode="by_window",
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    val_dtl = GlobalTimeSeriesDataloader(
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
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_window",
    )

    # --- Define Model ---
    manual_seed(seed)
    net = Sequential(
        LSTM(input_seq_len + num_features - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )

    # net.set_threads(8)
    net.to_device("cuda")
    out_updater = OutputUpdater(net.device)

    # optimal states placeholders
    states_optim = None
    look_back_buffer_mu_optim = None
    look_back_buffer_var_optim = None

    # save for plotting
    train_mses = []
    train_log_liks = []
    val_mses = []
    val_log_liks = []

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

            B = int(len(ts_id))  # current batch size
            ts_ids = np.asarray(ts_id, dtype=np.int64).reshape(-1)

            # set/reset LSTM states for the current batch
            batch_set_lstm_states(ts_ids, net, lstm_states)

            x = np.nan_to_num(x, nan=0.0)  # clean input from nans
            x_var = np.zeros_like(x)  # takes care of covariates
            y = np.concatenate(y, axis=0).astype(np.float32)  # shape (B,)

            # initialize look back buffer if first step for any series in batch
            ts_buffer = look_back_buffer_mu[ts_ids]
            needs_init = np.isnan(ts_buffer).all(axis=1)
            if np.any(needs_init):
                look_back_buffer_mu[ts_ids[needs_init]] = x[needs_init, :input_seq_len]
            if np.any(~needs_init):
                x[~needs_init, :input_seq_len] = look_back_buffer_mu[
                    ts_ids[~needs_init]
                ]  # update input sequence

            # always update the var buffer (no nans)
            x_var[:, :input_seq_len] = look_back_buffer_var[
                ts_id
            ]  # update input sequence

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
                m_pred=v_pred,
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

        # --- Validation ---
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(
            batch_size, shuffle=False, include_ids=True
        )

        # create placeholders for predictions and observations
        val_mu_preds = [[] for _ in range(nb_ts)]
        val_std_preds = [[] for _ in range(nb_ts)]
        val_obs = [[] for _ in range(nb_ts)]

        # define the lookback buffer for recursive prediction
        look_back_buffer_val_mu = np.copy(look_back_buffer_mu)
        look_back_buffer_val_var = np.copy(look_back_buffer_var)

        # One-step recursive prediction over the validation stream
        for x, y, ts_id, _ in val_batch_iter:

            B = int(len(ts_id))  # current batch size
            ts_ids = np.asarray(ts_id, dtype=np.int64).reshape(-1)

            # set LSTM states for the current batch
            batch_set_lstm_states(ts_ids, net, lstm_states)

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)
            x_var = np.zeros_like(x)  # takes care of covariates
            y = np.concatenate(y, axis=0).astype(np.float32)

            # insert values from lookback_buffer
            x[:, :input_seq_len] = look_back_buffer_val_mu[
                ts_id
            ]  # update input sequence
            x_var[:, :input_seq_len] = look_back_buffer_val_var[
                ts_id
            ]  # update input sequence

            # flatten input for pytagi
            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Predicion
            m_pred, v_pred = net(flat_x, flat_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # save prior states for plotting and metrics
            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            # store lstm states
            lstm_states = batch_get_lstm_states(ts_ids, net, lstm_states)

            # get the posterior states
            m_pred, v_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            for b in range(B):
                sid = int(ts_id[b])

                # append to per-series lists
                val_mu_preds[sid].append(float(m_prior[b]))
                val_std_preds[sid].append(float(std_prior[b]))
                val_obs[sid].append(float(y[b]))

            look_back_buffer_val_mu = update_look_back_buffer(
                look_back_buffer_val_mu,
                ts_idx=ts_id,
                m_pred=m_pred,
            )
            look_back_buffer_val_var = update_look_back_buffer(
                look_back_buffer_val_var,
                ts_idx=ts_id,
                m_pred=v_pred,
            )

        # get validation metrics
        mses = []
        log_liks = []
        for s in range(nb_ts):
            pred = np.asarray(val_mu_preds[s])
            std = np.asarray(val_std_preds[s])
            obs = np.asarray(val_obs[s])
            val_mse = metric.mse(pred, obs)
            val_log_lik = metric.log_likelihood(
                prediction=pred, observation=obs, std=std
            )
            mses.append(val_mse)
            log_liks.append(val_log_lik)

        val_mse = np.nanmean(mses)
        val_log_lik = np.nanmean(log_liks)
        val_mses.append(val_mse)
        val_log_liks.append(val_log_lik)

        # unstandardize
        for s in range(nb_ts):
            val_mu_preds[s] = np.array(val_mu_preds[s])
            val_std_preds[s] = np.array(val_std_preds[s])
            val_obs[s] = np.array(val_obs[s])
            val_mu_preds[s] = normalizer.unstandardize(
                val_mu_preds[s], global_mean[s], global_std[s]
            )
            val_std_preds[s] = normalizer.unstandardize_std(
                val_std_preds[s], global_std[s]
            )
            val_obs[s] = normalizer.unstandardize(
                val_obs[s], global_mean[s], global_std[s]
            )

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| lg_lik: {train_log_lik:>7.4f}| mse_val: {val_mse:>7.4f} | log_lk_val: {val_log_lik:>7.4f}",
            refresh=True,
        )

        # check if warmup period is done
        warmup_done = (epoch + 1) >= min_epochs

        # early-stopping
        if early_stopping_criteria == "mse":
            if (
                warmup_done
                and float(val_mse) < float(mse_optim)
                and val_mse is not np.nan
            ):
                have_best = True
                mse_optim = val_mse
                log_lik_optim = val_log_lik
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    mu_preds,
                    std_preds,
                    val_mu_preds,
                    val_std_preds,
                )
                look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                lstm_optim_states = copy.deepcopy(lstm_states)
        elif early_stopping_criteria == "log_lik":
            if (
                warmup_done
                and float(val_log_lik) > float(log_lik_optim)
                and val_log_lik is not np.nan
            ):
                have_best = True
                mse_optim = val_mse
                log_lik_optim = val_log_lik
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    mu_preds,
                    std_preds,
                    val_mu_preds,
                    val_std_preds,
                )
                look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                lstm_optim_states = copy.deepcopy(lstm_states)
        if warmup_done:
            last_improvement = epoch_optim if have_best else (min_epochs - 1)
            if epoch - last_improvement >= patience:
                if not have_best:
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        val_mu_preds,
                        val_std_preds,
                    )
                    lstm_optim_states = copy.deepcopy(lstm_states)
                    look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                    look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                break
        if (
            np.isnan(train_mse)
            or np.isnan(train_log_lik)
            or np.isnan(val_mse)
            or np.isnan(val_log_lik)
        ):
            print("Warning: NaN detected in training/validation metrics. Stopping...")
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
    net.save(os.path.join(out_dir, "param/model.pth"))

    # save training/validation metrics
    metrics = {
        "train_mse": np.array(train_mses),
        "train_log_lik": np.array(train_log_liks),
        "val_mse": np.array(val_mses),
        "val_log_lik": np.array(val_log_liks),
        "epoch_optim": epoch_optim + 1,
    }
    if not os.path.exists(out_dir + "/train_metrics"):
        os.makedirs(out_dir + "/train_metrics")
    np.savez(out_dir + "/train_metrics/metrics_0.npz", **metrics)

    # --- Testing ---
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloader(
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
    look_back_buffer_test_mu = np.copy(look_back_buffer_mu_optim)
    look_back_buffer_test_var = np.copy(look_back_buffer_var_optim)

    for x, y, ts_id, _ in test_batch_iter:

        ts_idx = ts_id.item()  # get the integer index of the time series

        # set LSTM states
        net.set_lstm_states(lstm_optim_states[ts_idx])

        x_var = np.zeros_like(x)  # takes care of covariates

        # insert values from lookback_buffer
        x[:, :input_seq_len] = look_back_buffer_test_mu[ts_id]  # update input sequence
        x_var[:, :input_seq_len] = look_back_buffer_test_var[
            ts_id
        ]  # update input sequence

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
        look_back_buffer_test_mu[ts_idx][:-1] = look_back_buffer_test_mu[ts_idx][1:]
        look_back_buffer_test_mu[ts_idx][-1] = float(np.ravel(m_pred)[-1])

        look_back_buffer_test_var[ts_idx][:-1] = look_back_buffer_test_var[ts_idx][1:]
        look_back_buffer_test_var[ts_idx][-1] = float(np.ravel(v_pred + var_obs)[-1])

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


def embed_model_run(
    nb_ts, num_epochs, batch_size, seed, early_stopping_criteria, train_size
):
    """
    Run a single global model across All time series and learn an embedding for each series.
    """

    print("Running global+embed model...")

    # Config
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1

    # early-stopping trackers
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    net_optim = []
    patience = 10  # epochs to wait for improvement before early stopping
    min_epochs = 10  # minimum number of epochs before early stopping
    have_best = False

    # --- Output Directory ---
    out_dir = "out/experiment01_embed"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize embeddings
    embedding_dim = 10
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embedding_dim),
        encoding_type="normal",
        seed=seed,
    )

    # save embeddings at beginning
    if not os.path.exists(out_dir + "/embeddings"):
        os.makedirs(out_dir + "/embeddings", exist_ok=True)
    embeddings.save(os.path.join(out_dir, "embeddings/embeddings_start.npz"))

    # Pre-allocate final CSVs
    horizon_cap = 2000  # manually set
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalTimeSeriesDataloader(
        x_file=f"data/hq/train_{train_size}/split_train_values.csv",
        date_time_file=f"data/hq/train_{train_size}/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        order_mode="by_window",
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    val_dtl = GlobalTimeSeriesDataloader(
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
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_window",
    )

    # --- Define Model ---
    manual_seed(seed)
    net = Sequential(
        LSTM(input_seq_len + num_features + embedding_dim - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )
    # net.set_threads(8)
    net.to_device("cuda")
    out_updater = OutputUpdater(net.device)

    net.input_state_update = True  # enable input state updates for embeddings

    # optimal states placeholders
    states_optim = None
    look_back_buffer_mu_optim = None
    look_back_buffer_var_optim = None

    # save for plotting
    train_mses = []
    train_log_liks = []
    val_mses = []
    val_log_liks = []

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

            B = int(len(ts_id))  # current batch size
            ts_ids = np.asarray(ts_id, dtype=np.int64).reshape(-1)

            # set/reset LSTM states for the current batch
            batch_set_lstm_states(ts_ids, net, lstm_states)

            x = np.nan_to_num(x, nan=0.0)  # clean input from nans
            x_var = np.zeros_like(x)  # takes care of covariates
            y = np.concatenate(y, axis=0).astype(np.float32)  # shape (B,)

            # initialize look back buffer if first step for any series in batch
            ts_buffer = look_back_buffer_mu[ts_ids]
            needs_init = np.isnan(ts_buffer).all(axis=1)
            if np.any(needs_init):
                look_back_buffer_mu[ts_ids[needs_init]] = x[needs_init, :input_seq_len]
            if np.any(~needs_init):
                x[~needs_init, :input_seq_len] = look_back_buffer_mu[
                    ts_ids[~needs_init]
                ]  # update input sequence

            # always update the var buffer (no nans)
            x_var[:, :input_seq_len] = look_back_buffer_var[
                ts_id
            ]  # update input sequence

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

            # store posterior lstm states
            lstm_states = batch_get_lstm_states(ts_ids, net, lstm_states)

            # store prior states
            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            # get the posterior states
            m_pred, v_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)
            #TODO: get posterior aleatoric uncertainty


            # get updates for embeddings
            mu_delta, var_delta = net.get_input_states()
            mu_delta = mu_delta.reshape(B, -1)
            var_delta = var_delta.reshape(B, -1)

            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var

            embeddings.update(
                ts_id, x_update[:, -embedding_dim:], var_update[:, -embedding_dim:]
            )

            # store for plotting
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
                m_pred=v_pred,
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

        # --- Validation ---
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(
            batch_size, shuffle=False, include_ids=True
        )

        # create placeholders for predictions and observations
        val_mu_preds = [[] for _ in range(nb_ts)]
        val_std_preds = [[] for _ in range(nb_ts)]
        val_obs = [[] for _ in range(nb_ts)]

        # define the lookback buffer for recursive prediction
        look_back_buffer_val_mu = np.copy(look_back_buffer_mu)
        look_back_buffer_val_var = np.copy(look_back_buffer_var)

        # One-step recursive prediction over the validation stream
        for x, y, ts_id, _ in val_batch_iter:

            B = int(len(ts_id))  # current batch size
            ts_ids = np.asarray(ts_id, dtype=np.int64).reshape(-1)

            # set LSTM states for the current batch
            batch_set_lstm_states(ts_ids, net, lstm_states)

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)
            x_var = np.zeros_like(x)  # takes care of covariates
            y = np.concatenate(y, axis=0).astype(np.float32)

            # insert values from lookback_buffer
            x[:, :input_seq_len] = look_back_buffer_val_mu[
                ts_id
            ]  # update input sequence
            x_var[:, :input_seq_len] = look_back_buffer_val_var[
                ts_id
            ]  # update input sequence

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

            # Predicion
            m_pred, v_pred = net(flat_x, flat_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # save prior states for plotting and metrics
            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            # store lstm states
            lstm_states = batch_get_lstm_states(ts_ids, net, lstm_states)

            # get the posterior states
            m_pred, v_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)
            var_obs, _ = calculate_gaussian_posterior(var_obs, flat_v[1::2], y, var_obs)

            for b in range(B):
                sid = int(ts_id[b])

                # append to per-series lists
                val_mu_preds[sid].append(float(m_prior[b]))
                val_std_preds[sid].append(float(std_prior[b]))
                val_obs[sid].append(float(y[b]))

            look_back_buffer_val_mu = update_look_back_buffer(
                look_back_buffer_val_mu,
                ts_idx=ts_id,
                m_pred=m_pred,
            )
            look_back_buffer_val_var = update_look_back_buffer(
                look_back_buffer_val_var,
                ts_idx=ts_id,
                m_pred=v_pred,
            )

        # get validation metrics
        mses = []
        log_liks = []
        for s in range(nb_ts):
            pred = np.asarray(val_mu_preds[s])
            std = np.asarray(val_std_preds[s])
            obs = np.asarray(val_obs[s])
            val_mse = metric.mse(pred, obs)
            val_log_lik = metric.log_likelihood(
                prediction=pred, observation=obs, std=std
            )
            mses.append(val_mse)
            log_liks.append(val_log_lik)

        val_mse = np.nanmean(mses)
        val_log_lik = np.nanmean(log_liks)
        val_mses.append(val_mse)
        val_log_liks.append(val_log_lik)

        # unstandardize
        for s in range(nb_ts):
            val_mu_preds[s] = np.array(val_mu_preds[s])
            val_std_preds[s] = np.array(val_std_preds[s])
            val_obs[s] = np.array(val_obs[s])
            val_mu_preds[s] = normalizer.unstandardize(
                val_mu_preds[s], global_mean[s], global_std[s]
            )
            val_std_preds[s] = normalizer.unstandardize_std(
                val_std_preds[s], global_std[s]
            )
            val_obs[s] = normalizer.unstandardize(
                val_obs[s], global_mean[s], global_std[s]
            )

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| lg_lik: {train_log_lik:>7.4f}| mse_val: {val_mse:>7.4f} | log_lk_val: {val_log_lik:>7.4f}",
            refresh=True,
        )

        # check if warmup period is done
        warmup_done = (epoch + 1) >= min_epochs

        # early-stopping
        if early_stopping_criteria == "mse":
            if (
                warmup_done
                and float(val_mse) < float(mse_optim)
                and val_mse is not np.nan
            ):
                have_best = True
                mse_optim = val_mse
                log_lik_optim = val_log_lik
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    mu_preds,
                    std_preds,
                    val_mu_preds,
                    val_std_preds,
                )
                look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                lstm_optim_states = copy.deepcopy(lstm_states)
                embeddings_optim = copy.deepcopy(embeddings)
        elif early_stopping_criteria == "log_lik":
            if (
                warmup_done
                and float(val_log_lik) > float(log_lik_optim)
                and val_log_lik is not np.nan
            ):
                have_best = True
                mse_optim = val_mse
                log_lik_optim = val_log_lik
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    mu_preds,
                    std_preds,
                    val_mu_preds,
                    val_std_preds,
                )
                look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                lstm_optim_states = copy.deepcopy(lstm_states)
                embeddings_optim = copy.deepcopy(embeddings)
        if warmup_done:
            last_improvement = epoch_optim if have_best else (min_epochs - 1)
            if epoch - last_improvement >= patience:
                if not have_best:
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        val_mu_preds,
                        val_std_preds,
                    )
                    lstm_optim_states = copy.deepcopy(lstm_states)
                    look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                    look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                    embeddings_optim = copy.deepcopy(embeddings)
                break
        if (
            np.isnan(train_mse)
            or np.isnan(train_log_lik)
            or np.isnan(val_mse)
            or np.isnan(val_log_lik)
        ):
            print("Warning: NaN detected in training/validation metrics. Stopping...")
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
    net.save(os.path.join(out_dir, "param/model.pth"))

    # load optimal embeddings
    if embeddings_optim:
        embeddings = embeddings_optim
    embeddings.save(os.path.join(out_dir, "embeddings/embeddings_final.npz"))

    # save training/validation metrics
    metrics = {
        "train_mse": np.array(train_mses),
        "train_log_lik": np.array(train_log_liks),
        "val_mse": np.array(val_mses),
        "val_log_lik": np.array(val_log_liks),
        "epoch_optim": epoch_optim + 1,
    }
    if not os.path.exists(out_dir + "/train_metrics"):
        os.makedirs(out_dir + "/train_metrics")
    np.savez(out_dir + "/train_metrics/metrics_0.npz", **metrics)

    # --- Testing ---
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloader(
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
    look_back_buffer_test_mu = np.copy(look_back_buffer_mu_optim)
    look_back_buffer_test_var = np.copy(look_back_buffer_var_optim)

    for x, y, ts_id, _ in test_batch_iter:

        ts_idx = ts_id.item()  # get the integer index of the time series

        # set LSTM states
        net.set_lstm_states(lstm_optim_states[ts_idx])

        x_var = np.zeros_like(x)  # takes care of covariates

        # insert values from lookback_buffer
        x[:, :input_seq_len] = look_back_buffer_test_mu[ts_id]  # update input sequence
        x_var[:, :input_seq_len] = look_back_buffer_test_var[
            ts_id
        ]  # update input sequence

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
        look_back_buffer_test_mu[ts_idx][:-1] = look_back_buffer_test_mu[ts_idx][1:]
        look_back_buffer_test_mu[ts_idx][-1] = float(np.ravel(m_pred)[-1])

        look_back_buffer_test_var[ts_idx][:-1] = look_back_buffer_test_var[ts_idx][1:]
        look_back_buffer_test_var[ts_idx][-1] = float(np.ravel(v_pred)[-1])

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


def shared_model_run(
    nb_ts, num_epochs, batch_size, seed, early_stopping_criteria, train_size
):
    """
    Run a single global model across All time series and learn an embedding for each series.
    """

    print("Running global+shared-embed model...")

    # Config
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1

    # early-stopping trackers
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    net_optim = []
    patience = 10  # epochs to wait for improvement before early stopping
    min_epochs = 10  # minimum number of epochs before early stopping
    have_best = False

    # --- Output Directory ---
    out_dir = "out/experiment01_shared"
    os.makedirs(out_dir, exist_ok=True)

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
    horizon_cap = 2000  # manually set
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    val_start_indices = np.full(nb_ts, -1, dtype=np.int32)
    test_start_indices = np.full(nb_ts, -1, dtype=np.int32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalTimeSeriesDataloader(
        x_file=f"data/hq/train_{train_size}/split_train_values.csv",
        date_time_file=f"data/hq/train_{train_size}/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        order_mode="by_window",
        random_seed=seed,  # defined for reproducibility of series shuffling
    )

    # Use the same scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std
    covariate_means = train_dtl.covariate_means
    covariate_stds = train_dtl.covariate_stds

    val_dtl = GlobalTimeSeriesDataloader(
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
        covariate_means=covariate_means,
        covariate_stds=covariate_stds,
        order_mode="by_series",
    )

    # --- Define Model ---
    manual_seed(seed)
    net = Sequential(
        LSTM(input_seq_len + num_features + embedding_dim - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 2),
        EvenExp(),
    )
    if batch_size > 1:
        net.set_threads(8)
    else:
        net.set_threads(1)
    out_updater = OutputUpdater(net.device)

    net.input_state_update = True  # enable input state updates for embeddings

    # optimal states placeholders
    states_optim = None
    look_back_buffer_mu_optim = None
    look_back_buffer_var_optim = None

    # save for plotting
    train_mses = []
    train_log_liks = []
    val_mses = []
    val_log_liks = []

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
            shuffle_series_blocks=True,  # ordered shuffle
        )

        # define the lookback buffer for recursive prediction
        look_back_buffer_mu = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        look_back_buffer_var = np.full((nb_ts, input_seq_len), 0.0, dtype=np.float32)
        lstm_states = [None] * nb_ts  # placeholder for LSTM states per series
        ts_i = -1  # used to track changes in series for by_series mode

        for x, y, ts_id, _ in batch_iter:

            ts_idx = ts_id.item()
            B = int(len(ts_id))  # current batch size

            # what to do with LSTM states depends on order_mode and batch size
            # TODO: need to update this to handle batches
            if train_dtl.order_mode == "by_series":
                if B == 1 and ts_idx != ts_i and epoch != 0:
                    net.reset_lstm_states()
                ts_i = ts_idx
            elif train_dtl.order_mode == "by_window":
                if lstm_states[ts_idx] is None and epoch != 0:
                    net.reset_lstm_states()
                elif lstm_states[ts_idx] is not None:
                    net.set_lstm_states(lstm_states[ts_idx])

            else:
                net.reset_lstm_states()
                print("Warning: LSTM states reset for each batch.")

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
                x = x.astype(np.float32)

            # always update the var buffer (no nans)
            x_var[:, :input_seq_len] = look_back_buffer_var[
                ts_id
            ]  # update input sequence
            x_var = x_var.astype(np.float32)

            # append embeddings to each input in the batch
            embed_mu, embed_var = get_combined_embeddings(
                ts_id,
                embedding_id_map,
                dam_embeddings,
                dam_type_embeddings,
                sensor_type_embeddings,
                direction_embeddings,
                sensor_embeddings,
            )  # shape: (B, embedding_dim)
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

            # store posterior lstm states
            # TODO: update to handle batches
            lstm_states[ts_idx] = net.get_lstm_states()

            # store prior states
            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            # get the posterior states
            m_pred, v_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            # get updates for embeddings
            mu_delta, var_delta = net.get_input_states()
            mu_delta = mu_delta.reshape(B, -1)
            var_delta = var_delta.reshape(B, -1)

            x_update = mu_delta * x_var
            var_update = x_var * var_delta * x_var

            update_combined_embeddings(
                ts_id,
                x_update[:, -embedding_dim:],
                var_update[:, -embedding_dim:],
                embedding_id_map,
                dam_embeddings,
                dam_type_embeddings,
                sensor_type_embeddings,
                direction_embeddings,
                sensor_embeddings,
            )

            # store for plotting
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
                m_pred=v_pred,
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

        # --- Validation ---
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(
            batch_size, shuffle=False, include_ids=True
        )

        # create placeholders for predictions and observations
        val_mu_preds = [[] for _ in range(nb_ts)]
        val_std_preds = [[] for _ in range(nb_ts)]
        val_obs = [[] for _ in range(nb_ts)]

        # define the lookback buffer for recursive prediction
        look_back_buffer_val_mu = np.copy(look_back_buffer_mu)
        look_back_buffer_val_var = np.copy(look_back_buffer_var)

        # One-step recursive prediction over the validation stream
        for x, y, ts_id, _ in val_batch_iter:

            # set LSTM states
            net.set_lstm_states(lstm_states[ts_id.item()])

            ts_idx = np.atleast_1d(np.asarray(ts_id, dtype=int))  # shape: (B,)
            B = int(len(ts_idx))  # current batch size

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)
            x_var = np.zeros_like(x)  # takes care of covariates
            y = np.concatenate(y, axis=0).astype(np.float32)

            # insert values from lookback_buffer
            x[:, :input_seq_len] = look_back_buffer_val_mu[
                ts_id
            ]  # update input sequence
            x = x.astype(np.float32)
            x_var[:, :input_seq_len] = look_back_buffer_val_var[
                ts_id
            ]  # update input sequence
            x_var = x_var.astype(np.float32)

            # append embeddings to each input in the batch
            embed_mu, embed_var = get_combined_embeddings(
                ts_id,
                embedding_id_map,
                dam_embeddings,
                dam_type_embeddings,
                sensor_type_embeddings,
                direction_embeddings,
                sensor_embeddings,
            )  # shape: (B, embedding_dim)
            x = np.concatenate(
                (x, embed_mu), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
            x_var = np.concatenate(
                (x_var, embed_var), axis=1
            )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

            # flatten input for model
            flat_x = np.concatenate(x, axis=0, dtype=np.float32)
            flat_var = np.concatenate(x_var, axis=0, dtype=np.float32)

            # Predicion
            m_pred, v_pred = net(flat_x, flat_var)

            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_obs = flat_m[1::2]  # odd indices var_v

            # save prior states for plotting and metrics
            m_prior = m_pred.copy()
            std_prior = np.sqrt(v_pred + var_obs)

            # store lstm states
            lstm_states[ts_id.item()] = net.get_lstm_states()

            # get the posterior states
            m_pred, v_pred = calculate_gaussian_posterior(m_pred, v_pred, y, var_obs)

            for b in range(B):
                sid = int(ts_idx[b])

                # append to per-series lists
                val_mu_preds[sid].append(float(m_prior[b]))
                val_std_preds[sid].append(float(std_prior[b]))
                val_obs[sid].append(float(y[b]))

            look_back_buffer_val_mu = update_look_back_buffer(
                look_back_buffer_val_mu,
                ts_idx=ts_id,
                m_pred=m_pred,
            )
            look_back_buffer_val_var = update_look_back_buffer(
                look_back_buffer_val_var,
                ts_idx=ts_id,
                m_pred=v_pred,
            )

        # get validation metrics
        mses = []
        log_liks = []
        for s in range(nb_ts):
            pred = np.asarray(val_mu_preds[s])
            std = np.asarray(val_std_preds[s])
            obs = np.asarray(val_obs[s])
            val_mse = metric.mse(pred, obs)
            val_log_lik = metric.log_likelihood(
                prediction=pred, observation=obs, std=std
            )
            mses.append(val_mse)
            log_liks.append(val_log_lik)

        val_mse = np.nanmean(mses)
        val_log_lik = np.nanmean(log_liks)
        val_mses.append(val_mse)
        val_log_liks.append(val_log_lik)

        # unstandardize
        for s in range(nb_ts):
            val_mu_preds[s] = np.array(val_mu_preds[s])
            val_std_preds[s] = np.array(val_std_preds[s])
            val_obs[s] = np.array(val_obs[s])
            val_mu_preds[s] = normalizer.unstandardize(
                val_mu_preds[s], global_mean[s], global_std[s]
            )
            val_std_preds[s] = normalizer.unstandardize_std(
                val_std_preds[s], global_std[s]
            )
            val_obs[s] = normalizer.unstandardize(
                val_obs[s], global_mean[s], global_std[s]
            )

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| lg_lik: {train_log_lik:>7.4f}| mse_val: {val_mse:>7.4f} | log_lk_val: {val_log_lik:>7.4f}",
            refresh=True,
        )

        # check if warmup period is done
        warmup_done = (epoch + 1) >= min_epochs

        # early-stopping
        if early_stopping_criteria == "mse":
            if (
                warmup_done
                and float(val_mse) < float(mse_optim)
                and val_mse is not np.nan
            ):
                have_best = True
                mse_optim = val_mse
                log_lik_optim = val_log_lik
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    mu_preds,
                    std_preds,
                    val_mu_preds,
                    val_std_preds,
                )
                look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
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
        elif early_stopping_criteria == "log_lik":
            if (
                warmup_done
                and float(val_log_lik) > float(log_lik_optim)
                and val_log_lik is not np.nan
            ):
                have_best = True
                mse_optim = val_mse
                log_lik_optim = val_log_lik
                epoch_optim = epoch
                net_optim = net.state_dict()
                states_optim = (
                    mu_preds,
                    std_preds,
                    val_mu_preds,
                    val_std_preds,
                )
                look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
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
        if warmup_done:
            last_improvement = epoch_optim if have_best else (min_epochs - 1)
            if epoch - last_improvement >= patience:
                if not have_best:
                    net_optim = net.state_dict()
                    states_optim = (
                        mu_preds,
                        std_preds,
                        val_mu_preds,
                        val_std_preds,
                    )
                    lstm_optim_states = copy.deepcopy(lstm_states)
                    look_back_buffer_mu_optim = np.copy(look_back_buffer_val_mu)
                    look_back_buffer_var_optim = np.copy(look_back_buffer_val_var)
                    embeddings_optim = copy.deepcopy(
                        (
                            dam_embeddings,
                            dam_type_embeddings,
                            sensor_type_embeddings,
                            direction_embeddings,
                            sensor_embeddings,
                        )
                    )
                break
        if (
            np.isnan(train_mse)
            or np.isnan(train_log_lik)
            or np.isnan(val_mse)
            or np.isnan(val_log_lik)
        ):
            print("Warning: NaN detected in training/validation metrics. Stopping...")
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
    net.save(os.path.join(out_dir, "param/model.pth"))

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

    # save training/validation metrics
    metrics = {
        "train_mse": np.array(train_mses),
        "train_log_lik": np.array(train_log_liks),
        "val_mse": np.array(val_mses),
        "val_log_lik": np.array(val_log_liks),
        "epoch_optim": epoch_optim + 1,
    }
    if not os.path.exists(out_dir + "/train_metrics"):
        os.makedirs(out_dir + "/train_metrics")
    np.savez(out_dir + "/train_metrics/metrics_0.npz", **metrics)

    # --- Testing ---
    print("Testing...")
    test_dtl = GlobalTimeSeriesDataloader(
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
    look_back_buffer_test_mu = np.copy(look_back_buffer_mu_optim)
    look_back_buffer_test_var = np.copy(look_back_buffer_var_optim)
    for x, y, ts_id, _ in test_batch_iter:

        ts_idx = ts_id.item()  # get the integer index of the time series

        # set LSTM states
        net.set_lstm_states(lstm_optim_states[ts_idx])

        x_var = np.zeros_like(x)  # takes care of covariates

        # insert values from lookback_buffer
        x[:, :input_seq_len] = look_back_buffer_test_mu[ts_id]  # update input sequence
        x = x.astype(np.float32)
        x_var[:, :input_seq_len] = look_back_buffer_test_var[
            ts_id
        ]  # update input sequence
        x_var = x_var.astype(np.float32)

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
        x = np.concatenate(
            (x, embed_mu), axis=1
        )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)
        x_var = np.concatenate(
            (x_var, embed_var), axis=1
        )  # shape: (B, input_seq_len + embedding_dim + num_features - 1)

        # flatten input for model
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
        look_back_buffer_test_mu[ts_idx][:-1] = look_back_buffer_test_mu[ts_idx][1:]
        look_back_buffer_test_mu[ts_idx][-1] = float(np.ravel(m_pred)[-1])

        look_back_buffer_test_var[ts_idx][:-1] = look_back_buffer_test_var[ts_idx][1:]
        look_back_buffer_test_var[ts_idx][-1] = float(np.ravel(v_pred)[-1])

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
    batch_size=127,
    seed=1,
    early_stopping_criteria="log_lik",
    train_size="1.0",  # proportion of training data to use: "0.3", "0.4", "0.6", "0.8", "1.0"
    experiments=None,
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
            default=None,
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
                train_size=train_size,
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
                train_size,
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
                train_size,
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
                train_size,
            )
        except Exception as e:
            print(f"Shared model run failed: {e}")


if __name__ == "__main__":
    main()

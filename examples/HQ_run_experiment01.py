import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import math

from examples.embedding_loader import (
    TimeSeriesEmbeddings,
)
from examples.data_loader import (
    GlobalTimeSeriesDataloader,
    GlobalInterleavedTimeSeriesDataloader,
)
from pytagi import exponential_scheduler, manual_seed
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential

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


def concat_ts_sample(data, data_add):
    """Concatenates two time series datasets."""
    x_combined = np.concatenate(
        (data.dataset["value"][0], data_add.dataset["value"][0]), axis=0
    )
    y_combined = np.concatenate(
        (data.dataset["value"][1], data_add.dataset["value"][1]), axis=0
    )
    # time_combined = np.concatenate(
    #     (data.dataset["date_time"], data_add.dataset["date_time"])
    # )
    data.dataset["value"] = (x_combined, y_combined)
    # data.dataset["date_time"] = time_combined
    return data


# --- Helper functions for embedding batched input ---
def prepare_batch_embeddings(x, input_seq_len, num_features, embed_dim, embeddings):
    """Fill the embedding tail for each sample block in a *flat* batched x.
    Returns (x_filled, x_var, ts_ids, embed_slices).
    Layout per sample block: [input_seq, ts_idx repeated embed_dim times].
    """
    x = np.asarray(x, dtype=np.float32)
    block_len = input_seq_len * num_features + embed_dim
    assert (
        x.size % block_len == 0
    ), f"Unexpected x.size={x.size}, not multiple of block_len={block_len}"
    bs_eff = x.size // block_len

    x_var = np.zeros_like(x, dtype=np.float32)
    ts_ids = []
    embed_slices = []

    for b in range(bs_eff):
        start = b * block_len
        e_start = start + (block_len - embed_dim)
        e_end = start + block_len

        # recover ts id stored as repeated ints in the tail (robust to float dtype)
        ts_b = int(round(float(np.mean(x[e_start:e_end]))))
        ts_ids.append(ts_b)
        embed_slices.append(slice(e_start, e_end))

        # fetch embedding for this ts and write into x/x_var
        embed_mu, embed_var = embeddings.get_embedding(ts_b)
        x[e_start:e_end] = embed_mu
        x_var[e_start:e_end] = embed_var

    return x.astype(np.float32), x_var, ts_ids, embed_slices


def apply_embedding_updates(
    mu_delta, var_delta, x_var, ts_ids, embed_slices, embeddings
):
    """Aggregate input-state updates over the embedding tails and write back to the store."""
    # elementwise updates
    x_update = mu_delta * x_var
    var_update = x_var * var_delta * x_var

    # accumulate per time series (in case a ts appears multiple times in a batch)
    acc_mu = {}
    acc_var = {}
    for ts_b, sl in zip(ts_ids, embed_slices):
        mu_u = x_update[sl]
        var_u = var_update[sl]
        if ts_b in acc_mu:
            acc_mu[ts_b] += mu_u
            acc_var[ts_b] += var_u
        else:
            acc_mu[ts_b] = mu_u.copy()
            acc_var[ts_b] = var_u.copy()

    for ts_b in acc_mu:
        embeddings.update(ts_b, acc_mu[ts_b], acc_var[ts_b])


def local_model_run(nb_ts, num_epochs, batch_size, sigma_v, seed):
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
    ytestPd = np.full((272, nb_ts), np.nan)
    SytestPd = np.full((272, nb_ts), np.nan)
    ytestTr = np.full((272, nb_ts), np.nan)

    # save sigma_v
    sigma_v_max = copy.copy(sigma_v)

    # --- Load Data ---
    pbar = tqdm(ts_idx, desc="Loading Data Progress")
    for ts in pbar:

        output_col = [0]
        num_features = 2
        input_seq_len = 52
        output_seq_len = 1
        seq_stride = 1
        early_stopping_criteria = "mse"
        log_lik_optim = -1e100
        mse_optim = 1e100
        epoch_optim = 0
        net_optim = []
        patience = 10
        sigma_v = sigma_v_max

        # set random seed for reproducibility
        manual_seed(seed)

        train_dtl = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_train_values.csv",
            date_time_file="data/hq/split_train_datetimes.csv",
            time_covariates=["week_of_year"],
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            global_scale="standard",
            scale_covariates=True,
        )

        val_dtl = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_val_values.csv",
            date_time_file="data/hq/split_val_datetimes.csv",
            time_covariates=["week_of_year"],
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            global_scale="standard",
            scale_covariates=True,
            covariate_means=train_dtl.covariate_means,
            covariate_stds=train_dtl.covariate_stds,
        )

        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_test_values.csv",
            date_time_file="data/hq/split_test_datetimes.csv",
            time_covariates=["week_of_year"],
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            global_scale="standard",
            scale_covariates=True,
            covariate_means=train_dtl.covariate_means,
            covariate_stds=train_dtl.covariate_stds,
        )

        # --- Define Model ---
        net = Sequential(
            LSTM(num_features + input_seq_len - 1, 40, 1),
            LSTM(40, 40, 1),
            Linear(40, 1),
        )
        net.set_threads(1)
        out_updater = OutputUpdater(net.device)

        # --- Training ---
        pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in pbar:
            mu_preds = []
            train_obs = []

            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

            # Decaying observation's variance
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * len(output_col),), sigma_v**2, dtype=np.float32
            )

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

                mu_preds.extend(m_pred)
                train_obs.extend(y)

                if not np.isnan(y).any():
                    K = v_pred / (v_pred + sigma_v**2)  # Kalman gain
                    m_pred = m_pred + K * (y - m_pred)  # posterior mean
                    v_pred = (1.0 - K) * v_pred  # posterior variance
                    m_pred = m_pred.astype(np.float32)
                    v_pred = v_pred.astype(np.float32)

            mu_preds = np.array(mu_preds)
            train_obs = np.array(train_obs)

            pred = normalizer.unstandardize(mu_preds, train_dtl.x_mean, train_dtl.x_std)
            obs = normalizer.unstandardize(train_obs, train_dtl.x_mean, train_dtl.x_std)
            train_mse = metric.mse(pred, obs)

            # Validation
            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds = []
            var_preds = []
            y_val = []
            x_val = []

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

                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v**2)
                x_val.extend(x)
                y_val.extend(y)

            mu_preds = np.array(mu_preds)
            std_preds = np.array(var_preds) ** 0.5
            y_val = np.array(y_val)

            mu_preds = normalizer.unstandardize(
                mu_preds, train_dtl.x_mean, train_dtl.x_std
            )
            std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std)

            y_val = normalizer.unstandardize(y_val, train_dtl.x_mean, train_dtl.x_std)

            # Compute log-likelihood for validation set
            mse_val = metric.mse(mu_preds, y_val)
            log_lik_val = metric.log_likelihood(
                prediction=mu_preds, observation=y_val, std=std_preds
            )

            # Progress bar
            pbar.set_description(
                f"Ts #{ts+1}/{nb_ts} | Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| mse_val: {mse_val:>7.4f} | log_lik_val: {log_lik_val:>7.4f} | sigma_v: {sigma_v:.4f}",
                refresh=True,
            )

            # early-stopping
            if early_stopping_criteria == "mse":
                if float(mse_val) < float(mse_optim):
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net.state_dict()
            elif early_stopping_criteria == "log_lik":
                if float(log_lik_val) > float(log_lik_optim):
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net.state_dict()
            if int(epoch) - int(epoch_optim) > patience:
                break

        # -- Testing --
        if net_optim:
            net.load_state_dict(net_optim)
        net.save(out_dir + "/param/model_{}.pth".format(str(ts)))
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        # define the lookback buffer for recursive prediction
        look_back_buffer = None

        # One-step recursive prediction over the validation stream
        for x, y in test_batch_iter:

            if look_back_buffer is None:
                look_back_buffer = x[:input_seq_len]
            else:
                look_back_buffer[:-1] = look_back_buffer[1:]  # shift left
                look_back_buffer[-1] = float(
                    np.ravel(m_pred)[-1]
                )  # append most recent pred
                x[:input_seq_len] = look_back_buffer  # update input sequence
                x = x.astype(np.float32)

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            # Predicion
            m_pred, v_pred = net(x)

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)
            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        mu_preds = normalizer.unstandardize(mu_preds, train_dtl.x_mean, train_dtl.x_std)
        std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std)
        y_test = normalizer.unstandardize(y_test, train_dtl.x_mean, train_dtl.x_std)

        # save test predicitons for each time series
        # Pad predictions to ensure length is 272
        mu_preds_padded = np.pad(
            mu_preds.flatten(), (0, 272 - len(mu_preds)), constant_values=np.nan
        )
        std_preds_padded = np.pad(
            std_preds.flatten() ** 2, (0, 272 - len(std_preds)), constant_values=np.nan
        )
        y_test_padded = np.pad(
            y_test.flatten(), (0, 272 - len(y_test)), constant_values=np.nan
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_test_padded

    np.savetxt(out_dir + "/ytestPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SytestPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/ytestTr.csv", ytestTr, delimiter=",")


def global_model_run(nb_ts, num_epochs, batch_size, sigma_v, seed):
    """
    Run a single global model across ALL time series using the interleaved dataloader.
    Training/validation: all series at once (round-robin windows).
    Testing: per-series loop preserved (using ts_indices filter) to keep your 1-step recursive logic.
    """

    # Config
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    early_stopping_criteria = "mse"
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
    horizon_cap = 272
    ytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    SytestPd = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)
    ytestTr = np.full((horizon_cap, nb_ts), np.nan, dtype=np.float32)

    # Build TRAIN loader over ALL series
    train_dtl = GlobalInterleavedTimeSeriesDataloader(
        x_file="data/hq/split_train_values.csv",
        date_time_file="data/hq/split_train_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        global_scale="standard",  # GLOBAL mean/std across all series
        scale_covariates=True,  # set True if you add covariates
        order_mode="by_window",
    )

    # Use the SAME scaling for validation/test
    global_mean = train_dtl.x_mean
    global_std = train_dtl.x_std

    val_dtl = GlobalInterleavedTimeSeriesDataloader(
        x_file="data/hq/split_val_values.csv",
        date_time_file="data/hq/split_val_datetimes.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        global_scale="standard",
        x_mean=global_mean,
        x_std=global_std,
        scale_covariates=True,
        order_mode="by_window",
    )

    # -----------------------
    # Network
    net = Sequential(
        LSTM(input_seq_len + num_features - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 1),
    )
    net.set_threads(1)
    out_updater = OutputUpdater(net.device)

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        mu_preds = []
        train_obs = []

        batch_iter = train_dtl.create_data_loader(
            batch_size=batch_size,
            shuffle=False,
        )

        # how many batches will be produced
        n = train_dtl.dataset["value"][0].shape[0]         # total samples
        num_batches = math.ceil(n / batch_size)         # same logic as your loader

        # observation noise schedule
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        # define the lookback buffer for recursive prediction
        look_back_buffer = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)

        print("Training...")
        for rw_idx, (x, y) in tqdm(
            enumerate(batch_iter),
            total=num_batches,
            desc="Batches",
            unit="batch",
        ):

            # get ts_idx from rw_idx
            rw_idx = rw_idx % nb_ts

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)
            x = x.squeeze(0)

            if np.isnan(look_back_buffer[rw_idx]).all():
                look_back_buffer[rw_idx] = x[:input_seq_len]
            else:
                x[:input_seq_len] = look_back_buffer[rw_idx]  # update input sequence
                x = x.astype(np.float32)

            # Forward
            m_pred, v_pred = net(x)

            # Update output layer
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y.flatten(),
                var_obs=var_y.flatten(),
                delta_states=net.input_delta_z_buffer,
            )

            # Backward + step
            net.backward()
            net.step()

            mu_preds.extend(m_pred)
            train_obs.extend(y)

            if not np.isnan(y).any():
                K = v_pred / (v_pred + sigma_v**2)  # Kalman gain
                m_pred = m_pred + K * (y - m_pred)  # posterior mean
                v_pred = (1.0 - K) * v_pred  # posterior variance
                m_pred = m_pred.astype(np.float32)
                v_pred = v_pred.astype(np.float32)

            if look_back_buffer is not None:
                look_back_buffer[rw_idx][:-1] = look_back_buffer[rw_idx][
                    1:
                ]  # shift left
                look_back_buffer[rw_idx][-1] = float(
                    np.ravel(m_pred)[-1]
                )  # append most recent pred

        mu_preds = np.array(mu_preds)
        train_obs = np.array(train_obs)

        pred = normalizer.unstandardize(mu_preds, train_dtl.x_mean, train_dtl.x_std)
        obs = normalizer.unstandardize(train_obs, train_dtl.x_mean, train_dtl.x_std)
        train_mse = metric.mse(pred, obs)

        # Validation
        print("Validating...")
        val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

        mu_preds = []
        var_preds = []
        y_val = []
        x_val = []

        # define the lookback buffer for recursive prediction
        # look_back_buffer_val = copy.copy(look_back_buffer)
        look_back_buffer_val = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)

        # One-step recursive prediction over the validation stream
        for rw_idx, (x, y) in enumerate(val_batch_iter):

            # get ts_idx from rw_idx
            rw_idx = rw_idx % nb_ts

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)
            x = x.squeeze(0)

            if np.isnan(look_back_buffer_val[rw_idx]).all():
                look_back_buffer_val[rw_idx] = x[:input_seq_len]
            else:
                x[:input_seq_len] = look_back_buffer_val[rw_idx]  # update input sequence
                x = x.astype(np.float32)

            # Predicion
            m_pred, v_pred = net(x)

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)
            x_val.extend(x)
            y_val.extend(y)

            if look_back_buffer_val is not None:
                look_back_buffer_val[rw_idx][:-1] = look_back_buffer_val[rw_idx][
                    1:
                ]  # shift left
                look_back_buffer_val[rw_idx][-1] = float(
                    np.ravel(m_pred)[-1]
                )  # append most recent pred

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_val = np.array(y_val)

        mu_preds = normalizer.unstandardize(mu_preds, train_dtl.x_mean, train_dtl.x_std)
        std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std)

        y_val = normalizer.unstandardize(y_val, train_dtl.x_mean, train_dtl.x_std)

        # Compute log-likelihood for validation set
        mse_val = metric.mse(mu_preds, y_val)
        log_lik_val = metric.log_likelihood(
            prediction=mu_preds, observation=y_val, std=std_preds
        )

        # Progress bar
        pbar.set_description(
            f"Ts #{ts+1}/{nb_ts} | Epoch {epoch + 1}/{num_epochs}| mse: {train_mse:>7.4f}| mse_val: {mse_val:>7.4f} | log_lik_val: {log_lik_val:>7.4f} | sigma_v: {sigma_v:.4f}",
            refresh=True,
        )

        # early-stopping
        if early_stopping_criteria == "mse":
            if float(mse_val) < float(mse_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
        elif early_stopping_criteria == "log_lik":
            if float(log_lik_val) > float(log_lik_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
        if int(epoch) - int(epoch_optim) > patience:
            break

    # Load optimal model
    if net_optim:
        net.load_state_dict(net_optim)

    # Save model
    net.save(os.path.join(out_dir, "param/model.pth"))

    # Testing (per-series with recursive 1-step)
    pbar = tqdm(ts_idx, desc="Testing Progress")

    for ts in pbar:
        test_dtl = GlobalInterleavedTimeSeriesDataloader(
            x_file="data/hq/split_test_values.csv",
            date_time_file="data/hq/split_test_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            time_covariates=["week_of_year"],
            keep_last_time_cov=True,
            global_scale="standard",
            x_mean=global_mean,
            x_std=global_std,
            scale_covariates=True,
            ts_indices=[int(ts)],
        )

        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        # define the lookback buffer for recursive prediction
        look_back_buffer = None

        # One-step recursive prediction over the validation stream
        for x, y in test_batch_iter:

            if look_back_buffer is None:
                look_back_buffer = x[:input_seq_len]
            else:
                look_back_buffer[:-1] = look_back_buffer[1:]  # shift left
                look_back_buffer[-1] = float(
                    np.ravel(m_pred)[-1]
                )  # append most recent pred
                x[:input_seq_len] = look_back_buffer  # update input sequence
                x = x.astype(np.float32)

            # replace nans in x with zeros
            x = np.nan_to_num(x, nan=0.0)

            # Predicion
            m_pred, v_pred = net(x)

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)
            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        mu_preds = normalizer.unstandardize(mu_preds, train_dtl.x_mean, train_dtl.x_std)
        std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std)
        y_test = normalizer.unstandardize(y_test, train_dtl.x_mean, train_dtl.x_std)

        # save test predicitons for each time series
        # Pad predictions to ensure length is 272
        mu_preds_padded = np.pad(
            mu_preds.flatten(), (0, 272 - len(mu_preds)), constant_values=np.nan
        )
        std_preds_padded = np.pad(
            std_preds.flatten() ** 2, (0, 272 - len(std_preds)), constant_values=np.nan
        )
        y_test_padded = np.pad(
            y_test.flatten(), (0, 272 - len(y_test)), constant_values=np.nan
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_test_padded

    np.savetxt(out_dir + "/ytestPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SytestPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/ytestTr.csv", ytestTr, delimiter=",")


def embed_model_run(nb_ts, num_epochs, batch_size, sigma_v, seed):
    """
    Run on global model on all time series
    """

    # Dataset
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 52  # for rolling window predictions in the test set
    embed_dim = 10  # embedding dimension

    # set random seed for reproducibility
    manual_seed(seed)

    # --- Initialize Embeddings ---
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embed_dim),
        encoding_type="normal",
        # encoding_type="sphere",
        seed=seed,
    )

    # --- Output Directory ---
    out_dir = "out/experiment01_embed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ytestPd = np.full((272, nb_ts), np.nan)
    SytestPd = np.full((272, nb_ts), np.nan)
    ytestTr = np.full((272, nb_ts), np.nan)

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    mean_train = [0.0] * nb_ts
    std_train = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_train_values.csv",
            date_time_file="data/hq/split_train_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            global_scale="standard",
            embedding_dim=embed_dim,
            embed_at_end=True,
        )

        # Store scaling factors
        mean_train[ts] = train_dtl_.x_mean
        std_train[ts] = train_dtl_.x_std

        val_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_val_values.csv",
            date_time_file="data/hq/split_val_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=mean_train[ts],
            x_std=std_train[ts],
            global_scale="standard",
            embedding_dim=embed_dim,
            embed_at_end=True,
        )

        if ts == 0:
            train_dtl = train_dtl_
            val_dtl = val_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)
            val_dtl = concat_ts_sample(val_dtl, val_dtl_)

    # Network
    net = Sequential(
        LSTM(num_features + input_seq_len + embed_dim - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # input state update
    net.input_state_update = True

    # --- Training ---
    mses = []
    mses_val = []  # to save mse_val for plotting
    ll_val = []  # to save log likelihood for plotting

    # options for early stopping
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    early_stopping_criteria = "mse"
    patience = 10
    net_optim = []

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:

        batch_iter = train_dtl.create_data_loader(
            batch_size,
            shuffle=True,
        )

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.5, decaying_factor=0.99, curr_iter=epoch
        )

        for x, y in batch_iter:
            # Fill embeddings per sample block and build x_var
            x, x_var, ts_ids, embed_slices = prepare_batch_embeddings(
                x, input_seq_len, num_features, embed_dim, embeddings
            )

            # Forward
            m_pred, _ = net(np.float32(x), np.float32(x_var))

            # Observation variance sized to actual batch
            var_y = np.full((np.size(y),), sigma_v**2, dtype=np.float32)

            # Output update
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Backprop + step
            net.backward()
            net.step()

            # Read back input-state deltas and update embeddings (batched)
            mu_delta, var_delta = net.get_input_states()
            apply_embedding_updates(
                mu_delta, var_delta, x_var, ts_ids, embed_slices, embeddings
            )

            # Metric
            mse = metric.mse(m_pred, y)
            mses.append(mse)

        # Validation
        val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=True)

        mu_preds = []
        var_preds = []
        y_val = []
        x_val = []

        for x, y in val_batch_iter:
            x, x_var, _, _ = prepare_batch_embeddings(
                x, input_seq_len, num_features, embed_dim, embeddings
            )

            # Prediction
            m_pred, v_pred = net(np.float32(x), np.float32(x_var))

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)

            x_val.extend(x)
            y_val.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_val = np.array(y_val)
        x_val = np.array(x_val)

        # Compute log-likelihood for validation set
        mse_val = metric.mse(mu_preds, y_val)
        log_lik_val = metric.log_likelihood(
            prediction=mu_preds, observation=y_val, std=std_preds
        )

        # Save mse_val and log likelihood for plotting
        mses_val.append(mse_val)
        ll_val.append(log_lik_val)

        # Progress bar
        pbar.set_postfix(
            mse=f"{np.mean(mses):.4f}",
            mse_val=f"{mse_val:.4f}",
            log_lik_val=f"{log_lik_val:.4f}",
        )
        # pbar.set_postfix(mse=f"{np.mean(mses):.4f}", sigma_v=f"{sigma_v:.4f}")

        # early-stopping
        if early_stopping_criteria == "mse":
            if float(mse_val) < float(mse_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
        elif early_stopping_criteria == "log_lik":
            if float(log_lik_val) > float(log_lik_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
        if int(epoch) - int(epoch_optim) > patience:
            break

    # load optimal model
    if net_optim:
        net.load_state_dict(net_optim)

    # save the model
    net.save(out_dir + "/param/model.pth")

    # save learned embeddings
    embeddings.save(out_dir)

    # Testing
    pbar = tqdm(ts_idx, desc="Testing Progress")

    for ts in pbar:

        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_test_values.csv",
            date_time_file="data/hq/split_test_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=mean_train[ts],
            x_std=std_train[ts],
            global_scale="standard",
            scale_covariates=True,
            embedding_dim=embed_dim,
            embed_at_end=True,
        )

        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        # Rolling window predictions
        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % rolling_window
            if RW_idx > 0:
                # Length of the input sequence portion (excluding embedding tail)
                seq_len = input_seq_len * num_features
                # Overwrite the last RW_idx time steps of the target feature with prior predictions
                x[seq_len - RW_idx * num_features : seq_len : num_features] = mu_preds[
                    -RW_idx:
                ]

            # Ensure embedding tail reflects the learned mean/var for this ts
            embed_mu, embed_var = embeddings.get_embedding(ts)
            # If the dataloader already wrote embed_mu, this is idempotent
            x[-embed_dim:] = embed_mu
            x_var = np.zeros_like(x, dtype=np.float32)
            x_var[-embed_dim:] = embed_var

            # Prediction with embedding variance
            m_pred, v_pred = net(np.float32(x), np.float32(x_var))

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)

            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # Unscale the predictions
        mu_preds = normalizer.unstandardize(mu_preds, mean_train[ts], std_train[ts])
        std_preds = normalizer.unstandardize_std(std_preds, std_train[ts])
        y_test = normalizer.unstandardize(y_test, mean_train[ts], std_train[ts])

        # save test predicitons for each time series
        # Pad predictions to ensure length is 272
        mu_preds_padded = np.pad(
            mu_preds.flatten(), (0, 272 - len(mu_preds)), constant_values=np.nan
        )
        std_preds_padded = np.pad(
            std_preds.flatten() ** 2, (0, 272 - len(std_preds)), constant_values=np.nan
        )
        y_test_padded = np.pad(
            y_test.flatten(), (0, 272 - len(y_test)), constant_values=np.nan
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_test_padded

    np.savetxt(out_dir + "/ytestPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SytestPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/ytestTr.csv", ytestTr, delimiter=",")


def shared_model_run(nb_ts, num_epochs, batch_size, sigma_v, seed):
    """
    Run on global model on all time series with shared sub-embeddings
    """

    # Dataset
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 1
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 52  # for rolling window predictions in the test set
    embed_dim = 5 * 4  # embedding dimension

    # map each ts to the sub embeddings

    # set random seed for reproducibility
    manual_seed(seed)

    # --- Initialize Embeddings ---
    embeddings = TimeSeriesEmbeddings(
        (nb_ts, embed_dim),
        encoding_type="normal",
        # encoding_type="sphere",
        seed=seed,
    )

    # --- Output Directory ---
    out_dir = "out/experiment01_embed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ytestPd = np.full((272, nb_ts), np.nan)
    SytestPd = np.full((272, nb_ts), np.nan)
    ytestTr = np.full((272, nb_ts), np.nan)

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    mean_train = [0.0] * nb_ts
    std_train = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_train_values.csv",
            date_time_file="data/hq/split_train_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            global_scale="standard",
            embedding_dim=embed_dim,
            embed_at_end=True,
        )

        # Store scaling factors
        mean_train[ts] = train_dtl_.x_mean
        std_train[ts] = train_dtl_.x_std

        val_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_val_values.csv",
            date_time_file="data/hq/split_val_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=mean_train[ts],
            x_std=std_train[ts],
            global_scale="standard",
            embedding_dim=embed_dim,
            embed_at_end=True,
        )

        if ts == 0:
            train_dtl = train_dtl_
            val_dtl = val_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)
            val_dtl = concat_ts_sample(val_dtl, val_dtl_)

    # Network
    net = Sequential(
        LSTM(num_features + input_seq_len + embed_dim - 1, 40, 1),
        LSTM(40, 40, 1),
        Linear(40, 1),
    )
    net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # input state update
    net.input_state_update = True

    # --- Training ---
    mses = []
    mses_val = []  # to save mse_val for plotting
    ll_val = []  # to save log likelihood for plotting

    # options for early stopping
    log_lik_optim = -1e100
    mse_optim = 1e100
    epoch_optim = 0
    early_stopping_criteria = "mse"
    patience = 10
    net_optim = []

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:

        batch_iter = train_dtl.create_data_loader(
            batch_size,
            shuffle=True,
        )

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
        )

        for x, y in batch_iter:
            # Fill embeddings per sample block and build x_var
            x, x_var, ts_ids, embed_slices = prepare_batch_embeddings(
                x, input_seq_len, num_features, embed_dim, embeddings
            )

            # Forward
            m_pred, _ = net(np.float32(x), np.float32(x_var))

            # Observation variance sized to actual batch
            var_y = np.full((np.size(y),), sigma_v**2, dtype=np.float32)

            # Output update
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )

            # Backprop + step
            net.backward()
            net.step()

            # Read back input-state deltas and update embeddings (batched)
            mu_delta, var_delta = net.get_input_states()
            apply_embedding_updates(
                mu_delta, var_delta, x_var, ts_ids, embed_slices, embeddings
            )

            # Metric
            mse = metric.mse(m_pred, y)
            mses.append(mse)

        # Validation
        val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=True)

        mu_preds = []
        var_preds = []
        y_val = []
        x_val = []

        for x, y in val_batch_iter:
            x, x_var, _, _ = prepare_batch_embeddings(
                x, input_seq_len, num_features, embed_dim, embeddings
            )

            # Prediction
            m_pred, v_pred = net(np.float32(x), np.float32(x_var))

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)

            x_val.extend(x)
            y_val.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_val = np.array(y_val)
        x_val = np.array(x_val)

        # Compute log-likelihood for validation set
        mse_val = metric.mse(mu_preds, y_val)
        log_lik_val = metric.log_likelihood(
            prediction=mu_preds, observation=y_val, std=std_preds
        )

        # Save mse_val and log likelihood for plotting
        mses_val.append(mse_val)
        ll_val.append(log_lik_val)

        # Progress bar
        pbar.set_postfix(
            mse=f"{np.mean(mses):.4f}",
            mse_val=f"{mse_val:.4f}",
            log_lik_val=f"{log_lik_val:.4f}",
        )
        # pbar.set_postfix(mse=f"{np.mean(mses):.4f}", sigma_v=f"{sigma_v:.4f}")

        # early-stopping
        if early_stopping_criteria == "mse":
            if float(mse_val) < float(mse_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
        elif early_stopping_criteria == "log_lik":
            if float(log_lik_val) > float(log_lik_optim):
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net_optim = net.state_dict()
        if int(epoch) - int(epoch_optim) > patience:
            break

    # load optimal model
    if net_optim:
        net.load_state_dict(net_optim)

    # save the model
    net.save(out_dir + "/param/model.pth")

    # save learned embeddings
    embeddings.save(out_dir)

    # Testing
    pbar = tqdm(ts_idx, desc="Testing Progress")

    for ts in pbar:

        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/hq/split_test_values.csv",
            date_time_file="data/hq/split_test_datetimes.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=mean_train[ts],
            x_std=std_train[ts],
            global_scale="standard",
            scale_covariates=True,
            embedding_dim=embed_dim,
            embed_at_end=True,
        )

        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        # Rolling window predictions
        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % rolling_window
            if RW_idx > 0:
                # Length of the input sequence portion (excluding embedding tail)
                seq_len = input_seq_len * num_features
                # Overwrite the last RW_idx time steps of the target feature with prior predictions
                x[seq_len - RW_idx * num_features : seq_len : num_features] = mu_preds[
                    -RW_idx:
                ]

            # Ensure embedding tail reflects the learned mean/var for this ts
            embed_mu, embed_var = embeddings.get_embedding(ts)
            # If the dataloader already wrote embed_mu, this is idempotent
            x[-embed_dim:] = embed_mu
            x_var = np.zeros_like(x, dtype=np.float32)
            x_var[-embed_dim:] = embed_var

            # Prediction with embedding variance
            m_pred, v_pred = net(np.float32(x), np.float32(x_var))

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v**2)

            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # Unscale the predictions
        mu_preds = normalizer.unstandardize(mu_preds, mean_train[ts], std_train[ts])
        std_preds = normalizer.unstandardize_std(std_preds, std_train[ts])
        y_test = normalizer.unstandardize(y_test, mean_train[ts], std_train[ts])

        # save test predicitons for each time series
        # Pad predictions to ensure length is 272
        mu_preds_padded = np.pad(
            mu_preds.flatten(), (0, 272 - len(mu_preds)), constant_values=np.nan
        )
        std_preds_padded = np.pad(
            std_preds.flatten() ** 2, (0, 272 - len(std_preds)), constant_values=np.nan
        )
        y_test_padded = np.pad(
            y_test.flatten(), (0, 272 - len(y_test)), constant_values=np.nan
        )

        ytestPd[:, ts] = mu_preds_padded
        SytestPd[:, ts] = std_preds_padded
        ytestTr[:, ts] = y_test_padded

    np.savetxt(out_dir + "/ytestPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SytestPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/ytestTr.csv", ytestTr, delimiter=",")


def main(nb_ts=135, num_epochs=100, batch_size=1, sigma_v=1, seed=1):
    """
    Main function to run all experiments on 186 time series
    """
    # Run1 --> local model
    # local_model_run(nb_ts, num_epochs, batch_size, sigma_v, seed)

    # Run2 --> global model
    global_model_run(nb_ts, num_epochs, batch_size, sigma_v, seed)

    # Run3 --> global model with embeddings
    # embed_model_run(nb_ts, num_epochs, batch_size, sigma_v=sigma_v, seed=seed)

    # Run4 --> global model with shared sub-embeddings
    # shared_model_run(nb_ts, num_epochs, batch_size, sigma_v=sigma_v, seed=seed)


if __name__ == "__main__":
    main()

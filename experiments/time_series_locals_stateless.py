import os
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from experiments.wandb_helpers import (
    log_model_parameters,
    init_run,
    log_data,
    finish_run,
)
import torch
from pathlib import Path
import pandas as pd

from experiments.config import Config

from examples.data_loader import (
    TimeSeriesDataloader,
)
from pytagi import Normalizer as normalizer
import pytagi.metric as metric

from experiments.utils import (
    build_model,
    plot_series,
    EarlyStopping,
    calculate_updates,
    adjust_params,
)

# Plotting defaults
import matplotlib as mpl

# Update matplotlib parameters in a single dictionary
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,  # Set line width to 1
    }
)


# --- Helper functions ---
# data loaders specific to local models
def prepare_dtls(
    x_file,
    date_file,
    input_seq_len,
    num_features,
    time_covariates,
    ts,
):
    train_dtl = TimeSeriesDataloader(
        x_file=x_file[0],
        date_time_file=date_file[0],
        time_covariates=time_covariates,
        keep_last_time_cov=True,
        output_col=[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        num_features=num_features,
        stride=1,
        ts_idx=ts,
    )

    val_dtl = TimeSeriesDataloader(
        x_file=x_file[1],
        date_time_file=date_file[1],
        time_covariates=time_covariates,
        keep_last_time_cov=True,
        output_col=[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        num_features=num_features,
        stride=1,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        ts_idx=ts,
    )
    test_dtl = TimeSeriesDataloader(
        x_file=x_file[2],
        date_time_file=date_file[2],
        time_covariates=time_covariates,
        keep_last_time_cov=True,
        output_col=[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        num_features=num_features,
        stride=1,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        ts_idx=ts,
    )

    return train_dtl, val_dtl, test_dtl


# prepare input specific to local models
def prepare_input(
    x,
    var_x: Optional[np.ndarray],
    look_back_mu: Optional[np.ndarray],
    look_back_var: Optional[np.ndarray],
    indices,
):
    x = np.nan_to_num(x, nan=0.0)
    if var_x is None:
        var_x = np.zeros_like(x, dtype=np.float32)
    input_seq_len = look_back_mu.shape[1]

    if look_back_mu is not None:
        x[:input_seq_len] = look_back_mu[indices]
    if look_back_var is not None:
        var_x[:input_seq_len] = look_back_var[indices]

    return x, var_x


# Define a class to store the look_back_buffers
# TODO: use the LookBackBuffer in the utils file instead
class LookBackBuffer:
    def __init__(self, input_seq_len, nb_ts):
        self.mu = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        self.var = np.full((nb_ts, input_seq_len), 0.0, dtype=np.float32)
        self.needs_initialization = [True for _ in range(nb_ts)]

    def initialize(self, initial_mu, initial_var, indices):
        if not indices:
            return

        indices = np.asarray(indices, dtype=np.intp)
        mu = np.asarray(initial_mu, dtype=np.float32)
        var = np.asarray(initial_var, dtype=np.float32)

        if mu.ndim == 1:
            mu = np.broadcast_to(mu, (indices.size, mu.shape[0]))
        elif mu.shape[0] != indices.size:
            mu = mu[indices]

        if var.ndim == 1:
            var = np.broadcast_to(var, (indices.size, var.shape[0]))
        elif var.shape[0] != indices.size:
            var = var[indices]

        pending_mask = np.fromiter(
            (self.needs_initialization[idx] for idx in indices),
            dtype=bool,
            count=indices.size,
        )
        if not pending_mask.any():
            return

        pending_indices = indices[pending_mask]
        self.mu[pending_indices] = np.nan_to_num(mu[pending_mask], nan=0.0)
        self.var[pending_indices] = np.nan_to_num(var[pending_mask], nan=0.0)
        for idx in pending_indices:
            self.needs_initialization[idx] = False

    def update(self, new_mu, new_var, indices):
        self.mu[indices] = np.roll(self.mu[indices], -1, axis=1)
        self.var[indices] = np.roll(self.var[indices], -1, axis=1)

        # Update the last column with new values
        self.mu[indices, -1] = new_mu
        self.var[indices, -1] = new_var

    def __call__(self, indices):
        return self.mu[indices], self.var[indices]


# Define class for storing predictions over all time steps
# TODO: use the Sates in the utils file instead
class States:
    def __init__(self, nb_ts, total_time_steps):
        """Initializes storage for mean and variance for time series states."""
        self.mu = np.full((nb_ts, total_time_steps), np.nan, dtype=np.float32)
        self.std = np.full((nb_ts, total_time_steps), np.nan, dtype=np.float32)

    def update(self, new_mu, new_std, indices, time_step):
        """
        Efficiently updates states using vectorized NumPy indexing.

        Args:
            new_mu: Array of new mean values.
            new_std: Array of new std values.
            indices: Array of time series indices to update.
            time_step: Array of time steps to update.
        """
        # This is the optimized part, replacing the slow for-loop.
        self.mu[indices, time_step] = new_mu.flatten()[0]
        self.std[indices, time_step] = new_std.flatten()[0]

    def __getitem__(self, idx):
        """Allows retrieving a full time series' states via my_states[idx]."""
        return self.mu[idx], self.std[idx]

    def __setitem__(self, idx, value):
        """Allows setting a full time series' states via my_states[idx] = (mu_array, var_array)."""
        self.mu[idx], self.std[idx] = value


def train_local_models(config, experiment_name: Optional[str] = None, wandb_run=None):

    # Create output directory
    output_dir = f"out/{experiment_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Display and save configuration
    config.to_yaml(os.path.join(output_dir, "config.yaml"))

    # Initalize states
    cap = 2000
    train_states = States(nb_ts=config.data_loader.nb_ts, total_time_steps=cap)
    val_states = States(nb_ts=config.data_loader.nb_ts, total_time_steps=cap)
    test_states = States(nb_ts=config.data_loader.nb_ts, total_time_steps=cap)

    # Initialize place holder for scaling factors
    x_means = [None] * config.data_loader.nb_ts
    x_stds = [None] * config.data_loader.nb_ts

    for ts in config.ts_to_use:

        # Prepare data loaders
        train_dtl, val_dtl, test_dtl = prepare_dtls(
            config.x_file,
            config.date_file,
            config.data_loader.input_seq_len,
            config.data_loader.num_features,
            config.data_loader.time_covariates,
            ts,
        )
        if config.data_loader.scale_method == "standard":
            x_means[ts] = train_dtl.x_mean[0]
            x_stds[ts] = train_dtl.x_std[0]

        # Build model
        net, output_updater = build_model(
            input_size=config.input_size,
            use_AGVI=config.use_AGVI,
            seed=config.seed,
            device=config.model.device,
            init_params=config.model.init_params,
        )

        # Add plasticity
        if config.model.init_params and config.model.variance_inject != 0.0:
            adjust_params(
                net,
                mode=config.model.variance_action,
                value=config.model.variance_inject,
                threshold=config.model.variance_threshold,
            )

        # Create progress bar
        pbar = tqdm(range(config.training.num_epochs), desc=f"Epochs (TS {ts})")

        # Initialize early stopping
        early_stopping = EarlyStopping(
            criteria=config.training.early_stopping_criteria,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            warmup_epochs=config.training.warmup_epochs,
        )

        # Prepare decaying sigma_v if not using AGVI
        if not config.use_AGVI:
            sigma_start, sigma_end = config.model.Sigma_v_bounds
            if sigma_start is None or sigma_end is None:
                raise ValueError(
                    "Sigma_v_bounds must be defined when AGVI is disabled."
                )
            sigma_start = float(sigma_start)
            sigma_end = float(sigma_end)
            if config.training.num_epochs <= 1:
                decaying_sigma_v = [sigma_start]
            else:
                decay_factor = float(config.model.decaying_factor)
                exponents = decay_factor ** np.arange(
                    config.training.num_epochs, dtype=np.float32
                )
                if np.isclose(exponents[0], exponents[-1]) or decay_factor <= 0.0:
                    weights = np.linspace(
                        1.0, 0.0, config.training.num_epochs, dtype=np.float32
                    )
                else:
                    weights = (exponents - exponents[-1]) / (
                        exponents[0] - exponents[-1]
                    )
                decaying_sigma_v = (
                    sigma_end + (sigma_start - sigma_end) * weights
                ).tolist()

        # --- Training loop ---
        for epoch in pbar:
            net.train()
            train_mse = []
            train_log_lik = []

            train_batch_iter = train_dtl.create_data_loader(
                batch_size=config.data_loader.batch_size,
                shuffle=False,
            )

            # Initialize look-back buffer and LSTM state container
            look_back_buffer = LookBackBuffer(
                input_seq_len=config.data_loader.input_seq_len, nb_ts=1
            )

            # get current sigma_v if not using AGVI
            if not config.use_AGVI:
                sigma_v = decaying_sigma_v[epoch]

            train_time_step = 0
            for x, y in train_batch_iter:

                # get current batch size
                B = config.data_loader.batch_size

                # reset LSTM states
                if epoch != 0:
                    net.reset_lstm_states()

                # prepare obsevation noise matrix
                if not config.use_AGVI:
                    var_y = np.full(
                        (B * len(config.data_loader.output_col),),
                        sigma_v**2,
                        dtype=np.float32,
                    )

                # prepare look_back buffer
                if look_back_buffer.needs_initialization[0]:
                    look_back_buffer.initialize(
                        initial_mu=x[: config.data_loader.input_seq_len],
                        initial_var=np.zeros_like(
                            x[: config.data_loader.input_seq_len], dtype=np.float32
                        ),
                        indices=[0],
                    )

                # prepare input
                x, var_x = prepare_input(
                    x=x,
                    var_x=None,
                    look_back_mu=(
                        look_back_buffer.mu
                        if config.training.use_look_back_predictions
                        else None
                    ),
                    look_back_var=(
                        look_back_buffer.var
                        if config.training.use_look_back_predictions
                        else None
                    ),
                    indices=[0],
                )

                # Feedforward
                m_pred, v_pred = net(x, var_x)

                # Specific to AGVI
                if config.use_AGVI:
                    flat_m = np.ravel(m_pred)
                    flat_v = np.ravel(v_pred)

                    m_pred = flat_m[::2]  # even indices
                    v_pred = flat_v[::2]  # even indices
                    var_y = flat_m[1::2]  # odd indices var_v

                s_pred_total = np.sqrt(v_pred + var_y)

                # Compute metrics
                mask = ~np.isnan(y.flatten())
                y_masked = y.flatten()[mask]
                m_pred_masked = m_pred[mask]
                s_pred_masked = s_pred_total[mask]

                if y_masked.size > 0:
                    batch_mse = metric.rmse(m_pred_masked, y_masked)
                    batch_log_lik = metric.log_likelihood(
                        m_pred_masked, y_masked, s_pred_masked
                    )
                    train_mse.append(batch_mse)
                    train_log_lik.append(batch_log_lik)

                # Store predictions
                train_states.update(
                    new_mu=m_pred,
                    new_std=s_pred_total,
                    indices=[ts],
                    time_step=train_time_step,
                )
                train_time_step += 1

                # Update
                m_post, v_post = calculate_updates(
                    net,
                    output_updater,
                    m_pred,
                    v_pred,
                    y.flatten(),
                    use_AGVI=config.use_AGVI,
                    var_y=var_y,
                )

                # TODO: check if really needed
                v_post = np.clip(v_post, a_min=1e-6, a_max=2.0)

                # Update look_back buffer
                look_back_buffer.update(
                    new_mu=m_post,
                    new_var=v_post,
                    indices=[0],
                )

            # End of epoch
            train_mse = np.mean(train_mse)
            train_log_lik = np.mean(train_log_lik)

            # Validation
            net.eval()
            val_mse = []
            val_log_lik = []

            # reset LSTM states
            net.reset_lstm_states()

            # reset look-back buffer
            look_back_buffer.needs_initialization = [True]

            val_batch_iter = val_dtl.create_data_loader(
                batch_size=config.data_loader.batch_size,
                shuffle=False,
            )

            val_time_step = 0
            for x, y in val_batch_iter:

                # get current batch size
                B = config.data_loader.batch_size

                # reset LSTM states
                net.reset_lstm_states()

                # prepare obsevation noise matrix
                if not config.use_AGVI:
                    var_y = np.full(
                        (B * len(config.data_loader.output_col),),
                        sigma_v**2,
                        dtype=np.float32,
                    )

                look_back_buffer.needs_initialization = [True]

                # prepare look_back buffer
                if look_back_buffer.needs_initialization[0]:
                    look_back_buffer.initialize(
                        initial_mu=x[: config.data_loader.input_seq_len],
                        initial_var=np.zeros_like(
                            x[: config.data_loader.input_seq_len], dtype=np.float32
                        ),
                        indices=[0],
                    )

                # prepare input
                x, var_x = prepare_input(
                    x=x,
                    var_x=None,
                    look_back_mu=(
                        look_back_buffer.mu
                        if config.training.val_predict_recursively
                        else None
                    ),
                    look_back_var=(
                        look_back_buffer.var
                        if config.training.val_predict_recursively
                        else None
                    ),
                    indices=[0],
                )

                # Feedforward
                m_pred, v_pred = net(x, var_x)

                # Specific to AGVI
                if config.use_AGVI:
                    flat_m = np.ravel(m_pred)
                    flat_v = np.ravel(v_pred)

                    m_pred = flat_m[::2]  # even indices
                    v_pred = flat_v[::2]  # even indices
                    var_y = flat_m[1::2]  # odd indices var_v

                s_pred_total = np.sqrt(v_pred + var_y)

                # Compute metrics
                mask = ~np.isnan(y.flatten())
                y_masked = y.flatten()[mask]
                m_pred_masked = m_pred[mask]
                s_pred_masked = s_pred_total[mask]

                if y_masked.size > 0:
                    batch_mse = metric.rmse(m_pred_masked, y_masked)
                    batch_log_lik = metric.log_likelihood(
                        m_pred_masked, y_masked, s_pred_masked
                    )
                    val_mse.append(batch_mse)
                    val_log_lik.append(batch_log_lik)

                # Store predictions
                val_states.update(
                    new_mu=m_pred,
                    new_std=s_pred_total,
                    indices=[ts],
                    time_step=val_time_step,
                )
                val_time_step += 1

                # Update look_back buffer
                look_back_buffer.update(
                    new_mu=m_pred,
                    new_var=v_pred,
                    indices=[0],
                )

            # End of epoch
            val_mse = np.mean(val_mse)
            val_log_lik = np.mean(val_log_lik)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Train RMSE": f"{train_mse:.4f}",
                    "Val RMSE": f"{val_mse:.4f}",
                    "Train LogLik": f"{train_log_lik:.4f}",
                    "Val LogLik": f"{val_log_lik:.4f}",
                    "Sigma_v": f"{sigma_v:.4f}" if not config.use_AGVI else "",
                }
            )

            if wandb_run:
                # Log metrics
                metrics_payload = {
                    "epoch": epoch,
                    "train_rmse": train_mse,
                    "train_log_lik": train_log_lik,
                    "val_rmse": val_mse,
                    "val_log_lik": val_log_lik,
                }
                if log_payload is None:
                    log_payload = metrics_payload
                else:
                    log_payload.update(metrics_payload)
                if not config.use_AGVI:
                    log_payload["sigma_v"] = sigma_v

                # Send all logs for this epoch
                log_data(log_payload, wandb_run=wandb_run)

            # Check for early stopping
            val_score = (
                val_log_lik
                if config.training.early_stopping_criteria == "log_lik"
                else val_mse
            )
            if early_stopping(
                val_score,
                net,
                look_back_buffer,
                lstm_states,
                train_states,
                val_states,
                sigma_v if not config.use_AGVI else None,
                embeddings=None,
            ):
                print(f"Early stopping at epoch {epoch+1}")
                net.load_state_dict(early_stopping.best_state)
                look_back_buffer = early_stopping.best_look_back_buffer
                lstm_states = early_stopping.best_lstm_state_container
                train_states = early_stopping.train_states
                val_states = early_stopping.val_states
                if not config.use_AGVI:
                    sigma_v = early_stopping.best_sigma_v
                break

        else:
            # If loop finished without early stopping, load the best model found
            if early_stopping.best_state is not None:
                print(
                    "Training finished. Loading best model from early stopping tracker."
                )
                net.load_state_dict(early_stopping.best_state)
                look_back_buffer = early_stopping.best_look_back_buffer
                lstm_states = early_stopping.best_lstm_state_container
                train_states = early_stopping.train_states
                val_states = early_stopping.val_states
                if not config.use_AGVI:
                    sigma_v = early_stopping.best_sigma_v

        # Save best model
        net.save(os.path.join(output_dir, f"param/model_{ts}.pth"))

        # --- Testing ---
        net.eval()

        # reset look-back buffer
        look_back_buffer.needs_initialization = [True]

        test_batch_iter = test_dtl.create_data_loader(
            batch_size=config.data_loader.batch_size,
            shuffle=False,
        )

        test_time_step = 0
        for (
            x,
            y,
        ) in test_batch_iter:

            # get current batch size
            B = config.data_loader.batch_size

            # reset LSTM states
            net.reset_lstm_states()

            # prepare obsevation noise matrix
            if not config.use_AGVI:
                var_y = np.full(
                    (B * len(config.data_loader.output_col),),
                    sigma_v**2,
                    dtype=np.float32,
                )

            # rolling window mechanism for traffic and electricity datasets
            if config.data_loader.use_rolling_window:
                if test_time_step % config.data_loader.rolling_window_size == 0:
                    look_back_buffer.needs_initialization = [True]

            # prepare look_back buffer
            if look_back_buffer.needs_initialization[0]:
                look_back_buffer.initialize(
                    initial_mu=x[: config.data_loader.input_seq_len],
                    initial_var=np.zeros_like(
                        x[: config.data_loader.input_seq_len], dtype=np.float32
                    ),
                    indices=[0],
                )

            # prepare input
            x, var_x = prepare_input(
                x=x,
                var_x=None,
                look_back_mu=(
                    look_back_buffer.mu
                    if config.training.test_predict_recursively
                    else None
                ),
                look_back_var=(
                    look_back_buffer.var
                    if config.training.test_predict_recursively
                    else None
                ),
                indices=[0],
            )

            # Feedforward
            m_pred, v_pred = net(x, var_x)

            # Specific to AGVI
            if config.use_AGVI:
                flat_m = np.ravel(m_pred)
                flat_v = np.ravel(v_pred)

                m_pred = flat_m[::2]  # even indices
                v_pred = flat_v[::2]  # even indices
                var_y = flat_m[1::2]  # odd indices var_v

            s_pred_total = np.sqrt(v_pred + var_y)

            # Store predictions
            test_states.update(
                new_mu=m_pred,
                new_std=s_pred_total,
                indices=[ts],
                time_step=test_time_step,
            )
            test_time_step += 1

            # Update look_back buffer
            look_back_buffer.update(
                new_mu=m_pred,
                new_var=v_pred,
                indices=[0],
            )

        # End of epoch
        net.reset_lstm_states()

    # Run over each time series and re_scale it
    if config.data_loader.scale_method == "standard":
        for i in range(config.data_loader.nb_ts):

            # skip if not in ts_to_use
            if i not in config.ts_to_use:
                continue

            # get mean and std
            mean = x_means[i]
            std = x_stds[i]

            # re-scale
            train_states.mu[i] = normalizer.unstandardize(train_states.mu[i], mean, std)
            train_states.std[i] = normalizer.unstandardize_std(train_states.std[i], std)
            val_states.mu[i] = normalizer.unstandardize(val_states.mu[i], mean, std)
            val_states.std[i] = normalizer.unstandardize_std(val_states.std[i], std)
            test_states.mu[i] = normalizer.unstandardize(test_states.mu[i], mean, std)
            test_states.std[i] = normalizer.unstandardize_std(test_states.std[i], std)

    # Save results
    np.savez(
        os.path.join(output_dir, "train_states.npz"),
        mu=train_states.mu,
        std=train_states.std,
    )
    np.savez(
        os.path.join(output_dir, "val_states.npz"),
        mu=val_states.mu,
        std=val_states.std,
    )
    np.savez(
        os.path.join(output_dir, "test_states.npz"),
        mu=test_states.mu,
        std=test_states.std,
    )


def eval_local_models(config, experiment_name: Optional[str] = None):
    """Evaluates forecasts stored in the .npz format."""

    input_dir = Path(f"out/{experiment_name}/")

    train_states = np.load(input_dir / "train_states.npz")
    val_states = np.load(input_dir / "val_states.npz")
    test_states = np.load(input_dir / "test_states.npz")
    true_train = pd.read_csv(
        config.x_file[0],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.ts_to_use,
    ).values
    true_val = pd.read_csv(
        config.x_file[1],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.ts_to_use,
    ).values
    true_test = pd.read_csv(
        config.x_file[2],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.ts_to_use,
    ).values

    def _trim_trailing_nans(x: np.ndarray):
        """Trim padded trailing NaNs in the *target* series, keep the same cut for datetime."""
        x = np.asarray(x)
        if x.ndim > 1:
            x = x.reshape(-1)
        if x.size == 0:
            return x.astype(np.float32)
        valid = ~np.isnan(x)
        if not np.any(valid):
            return np.array([], dtype=np.float32)
        last = np.where(valid)[0][-1]
        x = x[: last + 1]
        return x.astype(np.float32)

    # create placehoders for metrics per series
    test_rmse_list = []
    test_log_lik_list = []
    test_mae_list = []
    test_p50_list = []
    test_p90_list = []

    # create placeholders for global (micro) metrics
    all_stand_y_true = []
    all_stand_y_pred = []
    all_stand_s_pred = []
    all_yt_test = []
    all_ypred_test = []
    all_spred_test = []

    # Iterate over each time series and calculate metrics
    for local_idx, ts_id in tqdm(
        enumerate(config.ts_to_use),
        desc="Evaluating series",
        total=len(config.ts_to_use),
    ):

        # Get true values using the packed local index
        yt_train, yt_val, yt_test = (
            _trim_trailing_nans(
                true_train[config.data_loader.input_seq_len :, local_idx]
            ),
            _trim_trailing_nans(
                true_val[config.data_loader.input_seq_len :, local_idx]
            ),
            _trim_trailing_nans(
                true_test[config.data_loader.input_seq_len :, local_idx]
            ),
        )
        yt_full = np.concatenate([yt_train, yt_val, yt_test])

        # get expected value using the global TS ID
        ypred_train = train_states["mu"][ts_id][: len(yt_train)]
        ypred_val = val_states["mu"][ts_id][: len(yt_val)]
        ypred_test = test_states["mu"][ts_id][: len(yt_test)]
        ypred_full = np.concatenate([ypred_train, ypred_val, ypred_test])

        # get std using the global TS ID
        spred_train = train_states["std"][ts_id][: len(yt_train)]
        spred_val = val_states["std"][ts_id][: len(yt_val)]
        spred_test = test_states["std"][ts_id][: len(yt_test)]
        spred_full = np.concatenate([spred_train, spred_val, spred_test])

        # Store split indices
        val_test_indices = (len(yt_train), len(yt_train) + len(yt_val))

        # --- Plotting ---
        if config.evaluation.eval_plots:
            plot_series(
                ts_idx=ts_id,
                y_true=yt_full,
                y_pred=ypred_full,
                s_pred=spred_full,
                out_dir=input_dir / "figures",
                val_test_indices=val_test_indices,
                std_factor=1,
            )

        # --- Metrics ---
        if config.evaluation.eval_metrics:

            # Standardize test with training mean and std
            if config.data_loader.scale_method == "standard":
                train_mean = np.nanmean(yt_train)
                train_std = np.nanstd(yt_train)
            else:
                # manual standardization
                print("Using manual standardization for metrics...")
                train_mean = 0.0
                train_std = 1.0

            stand_y_true = normalizer.standardize(yt_test, train_mean, train_std)
            stand_y_pred = normalizer.standardize(ypred_test, train_mean, train_std)
            stand_s_pred = normalizer.standardize_std(spred_test, train_std)

            # metrics in standardized space
            test_rmse = metric.rmse(stand_y_pred, stand_y_true)
            test_log_lik = metric.log_likelihood(
                stand_y_pred, stand_y_true, stand_s_pred
            )
            test_mae = metric.mae(stand_y_pred, stand_y_true)

            # metrics in original space (but normalized)
            test_p50 = metric.Np50(yt_test, ypred_test)
            test_p90 = metric.Np90(yt_test, ypred_test, spred_test)

            # Append to lists
            test_rmse_list.append(test_rmse)
            test_log_lik_list.append(test_log_lik)
            test_mae_list.append(test_mae)
            test_p50_list.append(test_p50)
            test_p90_list.append(test_p90)

            # Accumulate for micro-average
            all_stand_y_true.append(stand_y_true)
            all_stand_y_pred.append(stand_y_pred)
            all_stand_s_pred.append(stand_s_pred)
            all_yt_test.append(yt_test)
            all_ypred_test.append(ypred_test)
            all_spred_test.append(spred_test)

    # Calculate overall metrics
    if config.evaluation.eval_metrics:
        # Macro averages
        macro_rmse = np.nanmean(test_rmse_list)
        macro_log_lik = np.nanmean(test_log_lik_list)
        macro_mae = np.nanmean(test_mae_list)
        macro_p50 = np.nanmean(test_p50_list)
        macro_p90 = np.nanmean(test_p90_list)

        # Micro averages
        # Concatenate all arrays
        full_stand_y_true = np.concatenate(all_stand_y_true)
        full_stand_y_pred = np.concatenate(all_stand_y_pred)
        full_stand_s_pred = np.concatenate(all_stand_s_pred)
        full_yt_test = np.concatenate(all_yt_test)
        full_ypred_test = np.concatenate(all_ypred_test)
        full_spred_test = np.concatenate(all_spred_test)

        # Calculate metrics on the full concatenated arrays
        micro_rmse = metric.rmse(full_stand_y_pred, full_stand_y_true)
        micro_log_lik = metric.log_likelihood(
            full_stand_y_pred, full_stand_y_true, full_stand_s_pred
        )
        micro_mae = metric.mae(full_stand_y_pred, full_stand_y_true)
        micro_p50 = metric.Np50(full_yt_test, full_ypred_test)
        micro_p90 = metric.Np90(full_yt_test, full_ypred_test, full_spred_test)

        # save metrics to a table per series and overall
        with open(input_dir / "evaluation_metrics.txt", "w") as f:
            f.write("Series_ID,RMSE,LogLik,MAE,P50,P90\n")
            for i in range(len(config.ts_to_use)):
                f.write(
                    f"{config.ts_to_use[i]},{test_rmse_list[i]:.4f},{test_log_lik_list[i]:.4f},"
                    f"{test_mae_list[i]:.4f},{test_p50_list[i]:.4f},"
                    f"{test_p90_list[i]:.4f}\n"
                )
            f.write(
                f"Macro_Average,{macro_rmse:.4f},{macro_log_lik:.4f},"
                f"{macro_mae:.4f},{macro_p50:.4f},"
                f"{macro_p90:.4f}\n"
            )
            f.write(
                f"Micro_Average,{micro_rmse:.4f},{micro_log_lik:.4f},"
                f"{micro_mae:.4f},{micro_p50:.4f},"
                f"{micro_p90:.4f}\n"
            )


def main(Train=True, Eval=True, log_wandb=False):

    list_of_seeds = [1, 3, 17, 42, 99]
    list_of_experiments = ["train30", "train40", "train60", "train80", "train100"]

    for seed in list_of_seeds:
        for exp in list_of_experiments:
            print(f"Running experiment: {exp} with seed {seed}")

            # Model category
            model_category = "locals"

            # Create folders for storing results
            output_base_dir = f"out/seed{seed}/{exp}"
            if not os.path.exists(output_base_dir):
                os.makedirs(output_base_dir)

            # Define experiment name
            experiment_name = f"seed{seed}/{exp}/experiment01_{model_category}"

            # Create configuration
            config = Config.from_yaml(
                f"experiments/configurations/{model_category}_HQ127.yaml"
            )

            config.seed = seed
            config.model.device = "cpu"
            config.evaluation.eval_plots = False

            # Convert config object to a dictionary for W&B
            config_dict = config.wandb_dict()
            config_dict["model_type"] = model_category

            # Display config
            config.display()

            if log_wandb:
                # Initialize W&B run
                run = init_run(
                    project="Local_Model_Run",
                    group="Time_Series_Local_Models",
                    name=f"{model_category}_{exp}_Seed{seed}",
                    config=config_dict,
                    reinit=True,  # Allows re-initializing in a loop
                    save_code=True,  # Saves the main script
                )
            else:
                run = None

            if Train:
                # Train model
                train_local_models(
                    config, experiment_name=experiment_name, wandb_run=run
                )

            if Eval:
                # Evaluate model
                eval_local_models(config, experiment_name=experiment_name)

            # Finish the W&B run
            if log_wandb:
                finish_run(run)


if __name__ == "__main__":
    main(False, True)

import numpy as np

from sklearn.decomposition import PCA
from typing import Optional
import copy
import networkx as nx

from experiments.data_loader import (
    TimeSeriesDataBuilder,
)

from typing import List

# Plotting defaults
import matplotlib.pyplot as plt

from pytagi import manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential, EvenExp


# --- Buffer Classes ---
class LookBackBuffer:
    def __init__(self, input_seq_len, nb_ts):
        self.mu = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        self.var = np.full((nb_ts, input_seq_len), 0.0, dtype=np.float32)
        self.needs_initialization = [True for _ in range(nb_ts)]

    def initialize(self, initial_mu, initial_var, indices):
        for idx, mu, var in zip(indices, initial_mu, initial_var):
            if self.needs_initialization[idx] and idx >= 0:
                self.mu[idx] = np.nan_to_num(mu, nan=0.0)
                self.var[idx] = np.nan_to_num(var, nan=0.0)
                self.needs_initialization[idx] = False

    def update(self, new_mu, new_var, indices):
        # Check for negative indexes
        active_mask = indices >= 0
        indices = indices[active_mask]
        new_mu = new_mu[active_mask]
        new_var = new_var[active_mask]

        self.mu[indices] = np.roll(self.mu[indices], -1, axis=1)
        self.var[indices] = np.roll(self.var[indices], -1, axis=1)

        # Update the last column with new values
        self.mu[indices, -1] = new_mu.ravel()
        self.var[indices, -1] = new_var.ravel()

    def __call__(self, indices):
        return self.mu[indices], self.var[indices]


# LSTM State Container
class LSTMStateContainer:
    """
    An optimized container for managing LSTM states using pre-allocated NumPy arrays.

    This container stores the hidden and cell states (mean and variance) for all
    time series across all LSTM layers in contiguous NumPy arrays. This avoids the
    overhead of creating and managing dictionaries or lists for each series and
    leverages vectorized operations for updating and retrieving batch states.

    Args:
        num_series (int): The total number of time series.
        layer_state_shapes (dict): A dictionary mapping layer index to the shape
                                   of its hidden state (e.g., {0: 40, 1: 40}).
    """

    def __init__(self, num_series: int, layer_state_shapes: dict):
        self.num_series = num_series
        self.layer_state_shapes = layer_state_shapes
        self.states = {}

        np.random.seed(1)

        # Initialize NumPy arrays for each layer and state component
        for layer_idx, state_dim in layer_state_shapes.items():
            # Shape: (num_series, state_dim)
            self.states[layer_idx] = {
                "mu_h": np.zeros((num_series, state_dim), dtype=np.float32),
                "var_h": np.zeros((num_series, state_dim), dtype=np.float32),
                "mu_c": np.zeros((num_series, state_dim), dtype=np.float32),
                "var_c": np.zeros((num_series, state_dim), dtype=np.float32),
            }

    def _unpack_net_states(self, net_states: dict, batch_size: int):
        """Helper to unpack the flat state arrays from the network into a structured dict."""
        try:
            unpacked = {}
            for layer_idx, (mu_h, var_h, mu_c, var_c) in net_states.items():
                state_dim = self.layer_state_shapes[layer_idx]
                # Reshape from flat array to (batch_size, state_dim)
                unpacked[layer_idx] = {
                    "mu_h": np.asarray(mu_h).reshape(batch_size, state_dim),
                    "var_h": np.asarray(var_h).reshape(batch_size, state_dim),
                    "mu_c": np.asarray(mu_c).reshape(batch_size, state_dim),
                    "var_c": np.asarray(var_c).reshape(batch_size, state_dim),
                }
            return unpacked
        except Exception as e:
            raise ValueError(
                f"Error unpacking network states: {e}. "
                f"Check that the network states match expected shapes."
                f"Layer idx: {layer_idx}, expected state dim: {state_dim}, "
                f"batch size: {batch_size}"
            ) from e

    def _pack_for_net(self, batch_states: dict):
        """Helper to pack a structured dict of states into the flat tuple format for the network."""
        packed = {}
        for layer_idx, states in batch_states.items():
            packed[layer_idx] = (
                states["mu_h"].flatten(),
                states["var_h"].flatten(),
                states["mu_c"].flatten(),
                states["var_c"].flatten(),
            )
        return packed

    def update_states_from_net(self, indices: np.ndarray, net):
        """
        Gets the latest states from the network for the given indices and
        updates the internal storage using vectorized assignment.

        Indices with a value of -1 are skipped.
        """

        valid_mask = indices != -1
        valid_indices_to_update = indices[valid_mask]
        if valid_indices_to_update.size == 0:
            return

        net_states = net.get_lstm_states()
        batch_size = len(indices)

        unpacked_states = self._unpack_net_states(net_states, batch_size)

        # Vectorized update using advanced indexing
        for layer_idx, components in unpacked_states.items():
            self.states[layer_idx]["mu_h"][valid_indices_to_update] = components[
                "mu_h"
            ][valid_mask]
            self.states[layer_idx]["var_h"][valid_indices_to_update] = components[
                "var_h"
            ][valid_mask]
            self.states[layer_idx]["mu_c"][valid_indices_to_update] = components[
                "mu_c"
            ][valid_mask]
            self.states[layer_idx]["var_c"][valid_indices_to_update] = components[
                "var_c"
            ][valid_mask]

    def set_states_on_net(self, indices: np.ndarray, net):
        """
        Retrieves the stored states for the given indices and sets them on the network.

        Indices with a value of -1 are sent as zero-states.
        """

        batch_size = len(indices)
        if batch_size != 1:
            valid_mask = indices != -1
            valid_indices_to_read = indices[valid_mask]
        else:
            valid_mask = np.array([True], dtype=bool)
            valid_indices_to_read = indices

        batch_states = {}
        for layer_idx, components in self.states.items():
            state_dim = self.layer_state_shapes[layer_idx]

            # 1. Create zero-filled arrays for the batch
            batch_mu_h = np.zeros((batch_size, state_dim), dtype=np.float32)
            batch_var_h = np.zeros((batch_size, state_dim), dtype=np.float32)
            batch_mu_c = np.zeros((batch_size, state_dim), dtype=np.float32)
            batch_var_c = np.zeros((batch_size, state_dim), dtype=np.float32)

            # 2. Get the states from storage *only* for valid indices
            source_mu_h = components["mu_h"][valid_indices_to_read]
            source_var_h = components["var_h"][valid_indices_to_read]
            source_mu_c = components["mu_c"][valid_indices_to_read]
            source_var_c = components["var_c"][valid_indices_to_read]

            # 3. Fill the batch arrays at the valid slots
            batch_mu_h[valid_mask] = source_mu_h
            batch_var_h[valid_mask] = source_var_h
            batch_mu_c[valid_mask] = source_mu_c
            batch_var_c[valid_mask] = source_var_c

            # 4. Store the correctly constructed batch
            batch_states[layer_idx] = {
                "mu_h": batch_mu_h,
                "var_h": batch_var_h,
                "mu_c": batch_mu_c,
                "var_c": batch_var_c,
            }

        packed_states = self._pack_for_net(batch_states)
        net.set_lstm_states(packed_states)

    def __call__(self, *args, **kwds):
        pass

    def get_statistics(self):
        stats = {}
        for layer_idx, components in self.states.items():
            stats[layer_idx] = {
                "mu_h_mean": np.mean(components["mu_h"]),
                "var_h_mean": np.mean(components["var_h"]),
                "mu_c_mean": np.mean(components["mu_c"]),
                "var_c_mean": np.mean(components["var_c"]),
            }
        return stats

# --- Early Stopping Class ---
class EarlyStopping:
    def __init__(
        self,
        criteria="log_lik",
        patience=10,
        min_delta=1e-4,
        warmup_epochs=0,
    ):
        self.criteria = criteria
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf if criteria == "log_lik" else np.inf
        self.counter = 0
        self.best_state = None
        self.best_look_back_buffer = None
        self.best_lstm_state_container = None
        self.train_states = None
        self.val_states = None
        self.best_sigma_v = None
        self.best_embeddings = None
        self.warmup_epochs = max(0, warmup_epochs)
        self.epoch_count = 0

    def _as_float(self, score):
        score_array = np.asarray(score)
        if score_array.size != 1:
            raise ValueError(
                f"Expected scalar score for early stopping, got shape {score_array.shape}"
            )
        return float(score_array.item())

    def _is_improvement(self, current_score):
        if not np.isfinite(current_score):
            return False
        if self.criteria == "log_lik":
            return current_score > self.best_score + self.min_delta
        return current_score < self.best_score - self.min_delta

    def _store_checkpoint(
        self,
        current_score,
        model,
        look_back_buffer,
        lstm_state_container,
        train_states,
        val_states,
        sigma_v,
        embeddings,
    ):
        self.best_score = current_score
        self.best_state = copy.deepcopy(model.state_dict())
        self.best_look_back_buffer = copy.deepcopy(look_back_buffer)
        self.best_lstm_state_container = copy.deepcopy(lstm_state_container)
        self.train_states = copy.deepcopy(train_states)
        self.val_states = copy.deepcopy(val_states)
        self.best_sigma_v = sigma_v
        self.best_embeddings = copy.deepcopy(embeddings)

    def __call__(
        self,
        current_score,
        model,
        look_back_buffer,
        lstm_state_container,
        train_states,
        val_states,
        sigma_v,
        embeddings,
    ):
        self.epoch_count += 1
        current_score = self._as_float(current_score)
        in_warmup = self.epoch_count <= self.warmup_epochs

        if in_warmup:
            return False  # Skip checkpointing during warmup

        if self.best_state is None or self._is_improvement(current_score):
            self._store_checkpoint(
                current_score,
                model,
                look_back_buffer,
                lstm_state_container,
                train_states,
                val_states,
                sigma_v,
                embeddings,
            )
            self.counter = 0
            return False  # Not early stopping

        self.counter += 1
        if self.counter >= self.patience:
            return True  # Early stopping
        return False  # Not early stopping


# --- States Class for plotting ---
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
        # Check for negative indices and skip them
        valid_mask = indices >= 0
        indices = indices[valid_mask]
        time_step = time_step[valid_mask]
        new_mu = new_mu[valid_mask]
        new_std = new_std[valid_mask]

        self.mu[indices, time_step] = new_mu.flatten()
        self.std[indices, time_step] = new_std.flatten()

    def __getitem__(self, idx):
        """Allows retrieving a full time series' states via my_states[idx]."""
        return self.mu[idx], self.std[idx]

    def __setitem__(self, idx, value):
        """Allows setting a full time series' states via my_states[idx] = (mu_array, var_array)."""
        self.mu[idx], self.std[idx] = value


# --- Helper functions ---
def prepare_data(
    x_file,
    date_file,
    input_seq_len,
    time_covariates,
    scale_method,
    order_mode,
    ts_to_use,
):
    train_data = TimeSeriesDataBuilder(
        x_file=x_file[0],
        date_time_file=date_file[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        stride=1,
        time_covariates=time_covariates,
        scale_method=scale_method,
        order_mode=order_mode,
        ts_to_use=ts_to_use,
    )

    val_data = TimeSeriesDataBuilder(
        x_file=x_file[1],
        date_time_file=date_file[1],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        stride=1,
        time_covariates=time_covariates,
        scale_method=scale_method,
        x_mean=train_data.x_mean,
        x_std=train_data.x_std,
        order_mode=order_mode,
        ts_to_use=ts_to_use,
    )
    test_data = TimeSeriesDataBuilder(
        x_file=x_file[2],
        date_time_file=date_file[2],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        stride=1,
        time_covariates=time_covariates,
        scale_method=scale_method,
        x_mean=train_data.x_mean,
        x_std=train_data.x_std,
        order_mode=order_mode,
        ts_to_use=ts_to_use,
    )

    return train_data, val_data, test_data


# Define model
def build_model(input_size, use_AGVI, seed, device, init_params=None):
    manual_seed(seed)
    if use_AGVI:
        net = Sequential(
            LSTM(input_size, 40, 1),
            LSTM(40, 40, 1),
            Linear(40, 2),
            EvenExp(),
        )
    else:
        net = Sequential(
            LSTM(input_size, 40, 1),
            LSTM(40, 40, 1),
            Linear(40, 1),
        )

    # hard set mu_b to 1.0 for all LSTM layers
    # state_dict = net.state_dict()
    # for layer_name, (mu_w, var_w, mu_b, var_b) in state_dict.items():
    #     if layer_name.split(".")[0].lower() == "lstm":
    #         b_gate_size = len(mu_b) // 4
    #         gate_size = len(mu_w) // 4

    #         print(f"Adjusting LSTM layer: {layer_name}")
    #         print(f" - Bias gate size: {b_gate_size}, Weight gate size: {gate_size}")

    #         # -- bias of forget gate
    #         # forget is the first quarter
    #         mu_b = mu_b.copy()
    #         mu_b[:b_gate_size] = [1.0] * b_gate_size

    #         # -- weights of forget gate
    #         mu_w = mu_w.copy()
    #         mu_w[:gate_size] = mu_w[:gate_size] + [0.05] * gate_size # shift weights to favor forget gate

    #         state_dict[layer_name] = (mu_w, var_w, mu_b, var_b)
    # net.load_state_dict(state_dict)

    if init_params is not None:
        # reset_param_variance(net, init_params)
        net.load(init_params)
    if device == "cpu":
        net.set_threads(1)
    elif device == "cuda":
        net.to_device("cuda")
    out_updater = OutputUpdater(net.device)
    return net, out_updater


def prepare_input(
    x,
    var_x: Optional[np.ndarray],
    look_back_mu: Optional[np.ndarray],
    look_back_var: Optional[np.ndarray],
    indices: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
):
    if var_x is None:
        var_x = np.zeros_like(x, dtype=np.float32)

    active_mask = indices >= 0
    input_seq_len = look_back_mu.shape[1]

    if look_back_mu is not None:
        x[active_mask, :input_seq_len] = look_back_mu[indices[active_mask]]
    if look_back_var is not None:
        var_x[active_mask, :input_seq_len] = look_back_var[indices[active_mask]]

    if embeddings is not None:
        # TODO: temporary fix for shared setup
        indices[indices == -1] = 0
        embed_mu, embed_var = embeddings(indices)  # shape: (B, E)

        # Zero out embedding rows for inactive entries
        embed_mu[~active_mask] = 0.0
        embed_var[~active_mask] = 0.0

        # Append embeddings
        x = np.concatenate((x, embed_mu), axis=1)
        var_x = np.concatenate((var_x, embed_var), axis=1)

    flat_x = x.astype(np.float32).reshape(-1)
    flat_var = var_x.astype(np.float32).reshape(-1)

    np.nan_to_num(flat_x, copy=False, nan=0.0)
    np.nan_to_num(flat_var, copy=False, nan=0.0)

    return flat_x, flat_var


# Define function to calculate updates
def calculate_updates(net, out_updater, m_pred, v_pred, y, use_AGVI, var_y=None):
    """
    Calculates the posterior mean and variance (Kalman update) for the
    output predictions.
    """
    # calculate updates
    if use_AGVI:
        # Update output layer
        out_updater.update_heteros(
            output_states=net.output_z_buffer,
            mu_obs=y,
            delta_states=net.input_delta_z_buffer,
        )
    elif not use_AGVI and var_y is not None:
        out_updater.update(
            output_states=net.output_z_buffer,
            mu_obs=y,
            var_obs=var_y,
            delta_states=net.input_delta_z_buffer,
        )
    else:
        raise ValueError("var_y must be provided when not using AGVI")

    # Backward + step
    net.backward()
    net.step()

    # --- Kalman Update Section ---

    # Use a safe epsilon for clipping, not machine epsilon.
    eps = 1e-8

    # manually update states on python API
    nan_indices = np.isnan(y)
    has_nan = bool(np.any(nan_indices))
    if has_nan:
        # Make copies to avoid modifying the original arrays
        y = np.array(y, copy=True)
        var_y = np.array(var_y, copy=True)

        # TODO: check this logic
        # Where y is NaN, we treat the prediction as the observation.
        np.copyto(y, m_pred, where=nan_indices)

        # Set variance of missing observations to 0 (or a small eps)
        # This effectively tells the Kalman filter to trust the prediction
        # for these points.
        np.copyto(var_y, 0.0, where=nan_indices)

    # Clean all inputs to the Kalman filter before using them.
    safe_m_pred = np.nan_to_num(m_pred, nan=0.0)
    safe_v_pred = np.nan_to_num(v_pred, nan=0.0)
    safe_y = np.nan_to_num(y, nan=0.0)
    safe_var_y = np.nan_to_num(var_y, nan=0.0)

    # Clip the observation variance away from zero.
    # Use a larger max value (e.g., 5.0) just in case.
    stablized_var_y = np.clip(safe_var_y, eps, 5.0)

    # Clip the *denominator* of the gain calculation.
    kalman_denominator = safe_v_pred + stablized_var_y
    stablized_denominator = np.clip(kalman_denominator, a_min=eps, a_max=None)

    # Kalman gain
    K = safe_v_pred / stablized_denominator

    # Clean the gain just in case 0/0 or inf/inf occurred.
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip gain to [0, 1] range to ensure variance reduction.
    K = np.clip(K, 0.0, 1.0)

    # Posterior mean
    m_post = safe_m_pred + K * (safe_y - safe_m_pred)

    # Posterior variance
    v_post = (1.0 - K) * safe_v_pred

    # Ensure posterior variance is always positive.
    v_post = np.clip(v_post, a_min=eps, a_max=None)

    if has_nan:
        # reset the posterior variance to the prior.
        np.copyto(v_post, safe_v_pred, where=nan_indices)

    # Final check to ensure no NaNs are returned.
    m_post = np.nan_to_num(m_post)
    v_post = np.nan_to_num(v_post)

    return m_post, v_post


def update_aleatoric_uncertainty(
    mu_z0: np.ndarray,
    var_z0: np.ndarray,
    mu_v2bar: np.ndarray,
    var_v2bar: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Updates the aleatoric uncertainty (v2bar) using a Kalman filter
    on the moments of the output distribution.
    """

    eps = 1e-8  # Cap for numerical stability
    max_val = 1e6  # Cap values at 1 million
    max_val_sq = 1e12  # Cap squared values

    # Clean ALL inputs
    mu_z0 = np.nan_to_num(mu_z0, posinf=max_val, neginf=-max_val)
    var_z0 = np.nan_to_num(var_z0, nan=eps, posinf=max_val)  # Use eps for variance
    mu_v2bar = np.nan_to_num(mu_v2bar, posinf=max_val, neginf=-max_val)
    var_v2bar = np.nan_to_num(
        var_v2bar, nan=eps, posinf=max_val
    )  # Use eps for variance

    # y may contain NaNs
    valid_indices = ~np.isnan(y)

    # Initialize posterior arrays as copies of the priors
    mu_v2bar_posterior = mu_v2bar.copy()
    var_v2bar_posterior = var_v2bar.copy()

    # If all y values are NaN, no update is possible.
    if not np.any(valid_indices):
        return (
            mu_v2bar_posterior,
            var_v2bar_posterior,
        )

    # Filter all inputs to only include the data for valid (non-NaN) indices.
    y_valid = y[valid_indices]
    y_valid = np.nan_to_num(y_valid, posinf=max_val, neginf=-max_val)

    mu_z0_valid = mu_z0[valid_indices]
    var_z0_valid = var_z0[valid_indices]
    mu_v2bar_valid = mu_v2bar[valid_indices]
    var_v2bar_valid = var_v2bar[valid_indices]

    # --- Kalman Filter on H ---

    # Define Prior Moments for V, Y, H on valid data
    mu_v = np.zeros_like(mu_v2bar_valid)

    # Clip prior variance to be positive
    var_v = np.clip(
        mu_v2bar_valid, a_min=eps, a_max=None
    )  # Prior aleatoric uncertainty

    mu_y = mu_z0_valid + mu_v

    # Clip prior variance components to be positive
    var_z0_valid_clipped = np.clip(var_z0_valid, a_min=eps, a_max=None)
    var_v_clipped = np.clip(var_v, a_min=eps, a_max=None)
    var_y = var_z0_valid_clipped + var_v_clipped

    mu_h = np.stack([mu_z0_valid, mu_v], axis=1)

    cov_h = np.zeros((len(mu_z0_valid), 2, 2), dtype=np.float64)
    cov_h[:, 0, 0] = var_z0_valid_clipped
    cov_h[:, 1, 1] = var_v_clipped

    cov_hy = np.stack([var_z0_valid_clipped, var_v_clipped], axis=1)

    # Clip denominator for Kalman gain
    stabilized_var_y = np.clip(var_y, eps, 5.0)  # Using 5.0 as a reasonable max

    kalman_gain_h = np.divide(
        cov_hy,
        stabilized_var_y[:, np.newaxis],
        out=np.zeros_like(cov_hy),
        where=stabilized_var_y[:, np.newaxis] > eps,  # Use > eps
    )

    # Clean gain just in case
    kalman_gain_h = np.nan_to_num(kalman_gain_h, nan=0.0, posinf=0.0, neginf=0.0)

    mu_h_posterior = mu_h + kalman_gain_h * (y_valid - mu_y)[:, np.newaxis]
    cov_h_posterior = cov_h - np.einsum("bi,bj->bij", kalman_gain_h, cov_hy)

    # Enforce symmetry
    cov_h_posterior_transposed = np.transpose(cov_h_posterior, (0, 2, 1))
    cov_h_posterior = 0.5 * (cov_h_posterior + cov_h_posterior_transposed)

    # Add diagonal jitter to ensure positive definiteness
    num_hidden = cov_h_posterior.shape[1]
    jitter = 1e-6
    cov_h_posterior = cov_h_posterior + jitter * np.eye(num_hidden)[np.newaxis, :, :]

    # --- Moment Matching for V^2 ---

    mu_v_posterior = mu_h_posterior[:, 1]
    # Clip mu_v_posterior to prevent overflow when squared
    mu_v_posterior = np.clip(mu_v_posterior, -max_val, max_val)

    var_v_posterior = cov_h_posterior[:, 1, 1]

    # Clip posterior variance to be positive and prevent overflow
    var_v_posterior = np.clip(var_v_posterior, eps, max_val_sq)

    # Calculate Posterior Moments for V^2 on valid data
    mu_v2_posterior = mu_v_posterior**2 + var_v_posterior
    var_v2_posterior = 2 * (var_v_posterior**2) + 4 * var_v_posterior * (
        mu_v_posterior**2
    )
    # Clip resulting variance
    var_v2_posterior = np.clip(var_v2_posterior, eps, max_val_sq)

    # Calculate Prior Moments for V^2 and Gain 'k' on valid data
    mu_v2_prior = mu_v2bar_valid
    var_v2_prior_unclipped = 3.0 * var_v2bar_valid + 2.0 * (mu_v2bar_valid**2)

    # Clip prior variance (with max)
    var_v2_prior = np.clip(var_v2_prior_unclipped, eps, max_val_sq)

    # 1. Clean inputs (though they should be clean from the top, this is safer)
    safe_var_v2bar_valid = np.nan_to_num(var_v2bar_valid, nan=eps, posinf=max_val)
    safe_var_v2_prior = np.nan_to_num(var_v2_prior, nan=eps, posinf=max_val_sq)

    # 2. Stabilize denominator
    stabilized_var_v2_prior = np.clip(safe_var_v2_prior, a_min=eps, a_max=None)

    # 3. This division is now safe
    k = safe_var_v2bar_valid / stabilized_var_v2_prior

    # Clean and clip gain k
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    k = np.clip(k, 0.0, 1.0)  # Gain should be between 0 and 1

    # Update V2bar to get its Posterior Moments on valid data
    mu_v2bar_posterior_valid = mu_v2bar_valid + k * (mu_v2_posterior - mu_v2_prior)
    var_v2bar_posterior_valid = var_v2bar_valid + k**2 * (
        var_v2_posterior - var_v2_prior
    )

    # Clip final posterior variance
    var_v2bar_posterior_valid = np.clip(var_v2bar_posterior_valid, eps, max_val_sq)

    # Place the calculated posterior values
    mu_v2bar_posterior[valid_indices] = mu_v2bar_posterior_valid
    var_v2bar_posterior[valid_indices] = var_v2bar_posterior_valid

    # Final check
    mu_v2bar_posterior = np.nan_to_num(
        mu_v2bar_posterior, posinf=max_val, neginf=-max_val
    )
    var_v2bar_posterior = np.nan_to_num(var_v2bar_posterior, nan=eps, posinf=max_val_sq)

    return (
        mu_v2bar_posterior,
        var_v2bar_posterior,
    )


# --- Uncertainty Injection Function ---
def adjust_params(net, mode="add", value=1e-2, threshold=5e-4, which_layer=None):
    """
    Adjusts the variances of weights and biases in the network's state dictionary.

    For each layer (or specified layers), if a variance value is below the given threshold,
    either adds or sets it to the specified value depending on the mode.

    Args:
        net: The neural network whose parameters will be modified.
        mode (str): "add" to increment variances, "set" to assign the value directly.
        value (float): The value to add or set for the variances.
        threshold (float): Variances below this value will be adjusted.
        which_layer (list or None): List of layer names to adjust. If None, all layers are adjusted.

    Returns:
        None. The network's parameters are updated in-place.
    """

    state_dict = net.state_dict()
    for layer_name, (mu_w, var_w, mu_b, var_b) in state_dict.items():
        if which_layer is None or layer_name in which_layer:
            if mode.lower() == "add":
                var_w = [x + value if x < threshold else x for x in var_w]
                var_b = [x + value if x < threshold else x for x in var_b]
            elif mode.lower() == "set":
                var_w = [value if x < threshold else x for x in var_w]
                var_b = [value if x < threshold else x for x in var_b]
            state_dict[layer_name] = (mu_w, var_w, mu_b, var_b)
    net.load_state_dict(state_dict)


def reset_param_variance(net, param_dir):
    """
    Reset the variances on `net` while loading the means from `param_to_load`.
    """
    variance_dict = net.state_dict()
    net.load(param_dir)
    mean_dict = net.state_dict()

    new_state_dict = {}
    for layer_name, (_, var_w, _, var_b) in variance_dict.items():
        if layer_name not in mean_dict:
            raise KeyError(f"Layer '{layer_name}' not found in source state_dict.")
        mu_w, _, mu_b, _ = mean_dict[layer_name]

        new_state_dict[layer_name] = (
            np.array(mu_w, copy=True),
            np.array(var_w, copy=True),
            np.array(mu_b, copy=True),
            np.array(var_b, copy=True),
        )

    net.load_state_dict(new_state_dict)


# Similarity / Distance Matrices
def cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = emb / norms
    return normalized @ normalized.T


def bhattacharyya_distance_matrix(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    if mu.shape != var.shape:
        raise ValueError("mu and var must share the same shape")

    eps = np.float32(1e-12)
    var = np.maximum(var, eps)

    mu_i = mu[:, None, :]
    mu_j = mu[None, :, :]
    var_i = var[:, None, :]
    var_j = var[None, :, :]
    sigma = 0.5 * (var_i + var_j)
    sigma = np.maximum(sigma, eps)

    diff = mu_i - mu_j
    term1 = 0.125 * np.sum(diff * diff / sigma, axis=-1)

    log_sigma = np.log(sigma)
    log_var_i = np.log(var_i)
    log_var_j = np.log(var_j)
    term2 = 0.5 * np.sum(log_sigma - 0.5 * (log_var_i + log_var_j), axis=-1)

    dist = term1 + term2
    np.fill_diagonal(dist, 0.0)
    return dist


# --- Eval Helper Functions ---
def plot_series(
    ts_idx, y_true, y_pred, s_pred, out_dir, val_test_indices, std_factor=1
):
    """Plot truth, prediction, and std_factor band for a single series."""

    out_dir.mkdir(parents=True, exist_ok=True)

    yt = y_true
    yp = y_pred if y_pred is not None else None
    sp = s_pred if s_pred is not None else None
    x = np.arange(len(yt))

    plt.figure(figsize=(10, 3))
    plt.plot(x, yt, label=r"$y_{true}$", color="red")
    if yp is not None:
        plt.plot(x, yp, label=r"$\mathbb{E}[Y']$", color="blue")
    if sp is not None and yp is not None:
        lower = yp - std_factor * sp
        upper = yp + std_factor * sp
        plt.fill_between(
            x,
            lower,
            upper,
            color="blue",
            alpha=0.3,
            label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(std_factor),
        )

    # Shade validation and test regions
    if val_test_indices is not None:
        val_start, test_start = val_test_indices
        end_time = len(yt)
        if val_start < test_start:
            plt.axvspan(
                val_start,
                test_start - 1,
                color="green",
                alpha=0.15,
                label="Validation",
                linewidth=0,
            )
        if test_start < end_time:
            plt.axvspan(
                test_start,
                end_time,
                color="purple",
                alpha=0.15,
                label="Test",
                linewidth=0,
            )

    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False,
    )
    out_path = out_dir / f"series_{ts_idx:03d}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_embeddings(mu_embedding, n_series, in_dir, out_path, labels=None):
    pca = PCA(n_components=2)
    mu_emb_2d = pca.fit_transform(mu_embedding)

    # plot embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(mu_emb_2d[:, 0], mu_emb_2d[:, 1], c="blue", alpha=0.7)
    if labels is not None:
        for i, label in enumerate(labels):
            plt.text(mu_emb_2d[i, 0], mu_emb_2d[i, 1], label)
    else:
        for i in range(n_series):
            plt.text(mu_emb_2d[i, 0], mu_emb_2d[i, 1], str(i))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    emb_plot_path = in_dir / out_path
    plt.savefig(emb_plot_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_similarity(
    sim_matrix: np.ndarray,
    out_path,
    title: str,
    labels: Optional[List[str]] = None,
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("sim_matrix must be a square 2D array")

    # Order rows/cols by aggregate similarity to highlight structure.
    score = np.sum(sim_matrix, axis=1)
    order = np.argsort(-score)
    ordered = sim_matrix[order][:, order]

    if labels is not None:
        if len(labels) != sim_matrix.shape[0]:
            raise ValueError("labels must have the same length as sim_matrix size")
        ordered_labels = [str(labels[idx]) for idx in order]
    else:
        ordered_labels = [str(idx) for idx in order]

    num_series = ordered.shape[0]
    width = max(8.0, min(num_series * 0.4, 24.0))
    height = max(6.0, min(num_series * 0.4, 24.0))
    plt.figure(figsize=(width, height))
    heatmap = plt.imshow(
        ordered,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.title(f"{title} (sorted by similarity)")
    plt.xlabel("Entity/Series Index")  # Generic label
    plt.ylabel("Entity/Series Index")  # Generic label

    if num_series > 0:
        tick_positions = np.arange(num_series, dtype=int)
        rotation = 45 if num_series <= 20 else 90
        fontsize = 8 if num_series <= 30 else max(4, 12 - num_series // 10)
        plt.xticks(
            tick_positions,
            ordered_labels,
            rotation=rotation,
            ha="right",
            fontsize=fontsize,
        )
        plt.yticks(tick_positions, ordered_labels, fontsize=fontsize)

    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


# --- Plotting Functions ---
def plot_similarity(
    sim_matrix: np.ndarray,
    out_path,
    title: str,
    labels: Optional[List[str]] = None,
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    sim_matrix = np.asarray(sim_matrix, dtype=np.float32)
    if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("sim_matrix must be a square 2D array")

    if labels is not None:
        if len(labels) != sim_matrix.shape[0]:
            raise ValueError("labels must have the same length as sim_matrix size")

    num_series = sim_matrix.shape[0]
    width = max(8.0, min(num_series * 0.4, 24.0))
    height = max(6.0, min(num_series * 0.4, 24.0))
    plt.figure(figsize=(width, height))
    heatmap = plt.imshow(
        sim_matrix,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.title(f"{title} (sorted by similarity)")
    plt.xlabel("Entity/Series Index")  # Generic label
    plt.ylabel("Entity/Series Index")  # Generic label

    if num_series > 0:
        tick_positions = np.arange(num_series, dtype=int)
        rotation = 45 if num_series <= 20 else 90
        fontsize = 8 if num_series <= 30 else max(4, 12 - num_series // 10)
        plt.xticks(
            tick_positions,
            labels,
            rotation=rotation,
            ha="right",
            fontsize=fontsize,
        )
        plt.yticks(tick_positions, labels, fontsize=fontsize)

    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_similarity_graph(
    sim_matrix: np.ndarray,
    out_path,
    threshold=0.8,
    title="Time Series Similarity Graph",
):
    sim = np.asarray(sim_matrix, dtype=float)
    n = sim.shape[0]
    assert sim.shape[0] == sim.shape[1], "sim_matrix must be square"

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            val = sim[i, j]
            if np.isfinite(val) and val >= threshold:
                G.add_edge(i, j, weight=float(val))

    if G.number_of_edges() == 0:
        print(f"No edges at threshold {threshold}. Try lowering it.")
        return

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.4)

    # Get edges + widths safely
    edges_with_data = list(G.edges(data=True))
    edgelist = [(u, v) for (u, v, _) in edges_with_data]
    widths = [max(0.5, d.get("weight", 0.0) * 2.0) for (_, _, d) in edges_with_data]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=300, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=14)
    plt.title(f"{title}\n(Edges ≥ {threshold})", fontsize=16)
    plt.axis("off")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()

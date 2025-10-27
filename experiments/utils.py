import numpy as np

from sklearn.decomposition import PCA
from typing import Optional
import copy

from experiments.data_loader import (
    GlobalTimeSeriesDataloader,
)

# Plotting defaults
import matplotlib.pyplot as plt

from pytagi import manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential, EvenExp


# --- Helper functions ---
def prepare_dtls(
    x_file,
    date_file,
    input_seq_len,
    time_covariates,
    scale_method,
    order_mode,
    ts_to_use,
):
    train_dtl = GlobalTimeSeriesDataloader(
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

    val_dtl = GlobalTimeSeriesDataloader(
        x_file=x_file[1],
        date_time_file=date_file[1],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        stride=1,
        time_covariates=time_covariates,
        scale_method=scale_method,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        order_mode=order_mode,
        ts_to_use=ts_to_use,
    )
    test_dtl = GlobalTimeSeriesDataloader(
        x_file=x_file[2],
        date_time_file=date_file[2],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        stride=1,
        time_covariates=time_covariates,
        scale_method=scale_method,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        order_mode=order_mode,
        ts_to_use=ts_to_use,
    )

    return train_dtl, val_dtl, test_dtl


# Define model
def build_model(input_size, use_AGVI, seed=1, device="cpu"):
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

    # manually update states on python API
    nan_indices = np.isnan(y)
    has_nan = bool(np.any(nan_indices))
    if has_nan:
        y = np.array(y, copy=True)
        var_y = np.array(var_y, copy=True)
        np.copyto(y, m_pred, where=nan_indices)
        np.copyto(var_y, 0.0, where=nan_indices)

    # kalman update
    K = v_pred / (v_pred + var_y)  # Kalman gain
    m_post = m_pred + K * (y - m_pred)  # posterior mean
    v_post = (1.0 - K) * v_pred  # posterior variance
    if has_nan:
        np.copyto(v_post, v_pred, where=nan_indices)

    return m_post, v_post


# Define function to update aleatoric uncertainty
def update_aleatoric_uncertainty(
    mu_z0: np.ndarray,
    var_z0: np.ndarray,
    mu_v2bar: np.ndarray,
    var_v2bar: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    # Promote to float64 for numerical stability; remember original dtype for output
    out_dtype = mu_v2bar.dtype
    mu_z0 = np.asarray(mu_z0, dtype=np.float64, order="C")
    var_z0 = np.asarray(var_z0, dtype=np.float64, order="C")
    mu_v2bar = np.asarray(mu_v2bar, dtype=np.float64, order="C")
    var_v2bar = np.asarray(var_v2bar, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64, order="C")

    # Handle NaN values in y
    valid_indices = ~np.isnan(y)

    # Initialize posterior arrays as copies of the priors
    mu_v2bar_posterior = mu_v2bar.copy()
    var_v2bar_posterior = var_v2bar.copy()

    # If all y values are NaN, no update is needed
    if not np.any(valid_indices):
        return (
            mu_v2bar_posterior.astype(out_dtype, copy=False),
            var_v2bar_posterior.astype(out_dtype, copy=False),
        )

    # Filter all inputs to only include the data for valid (non-NaN) indices.
    y_valid = y[valid_indices]
    mu_z0_valid = mu_z0[valid_indices]
    var_z0_valid = var_z0[valid_indices]
    mu_v2bar_valid = mu_v2bar[valid_indices]
    var_v2bar_valid = var_v2bar[valid_indices]

    # Step 1: Define Prior Moments for V, Y, H on valid data
    mu_v = np.zeros_like(mu_v2bar_valid)
    var_v = mu_v2bar_valid  # Prior aleatoric uncertainty

    mu_y = mu_z0_valid + mu_v
    var_y = var_z0_valid + var_v

    mu_h = np.stack([mu_z0_valid, mu_v], axis=1)

    cov_h = np.zeros((len(mu_z0_valid), 2, 2), dtype=np.float64)
    cov_h[:, 0, 0] = var_z0_valid
    cov_h[:, 1, 1] = var_v

    cov_hy = np.stack([var_z0_valid, var_v], axis=1)

    # Step 2: Calculate Posterior Moments for H using y on valid data
    stabilized_var_y = np.clip(var_y, np.finfo(np.float64).eps, None)
    kalman_gain_h = np.divide(
        cov_hy,
        stabilized_var_y[:, np.newaxis],
        out=np.zeros_like(cov_hy),
        where=stabilized_var_y[:, np.newaxis] != 0,
    )
    mu_h_posterior = mu_h + kalman_gain_h * (y_valid - mu_y)[:, np.newaxis]
    cov_h_posterior = cov_h - np.einsum("bi,bj->bij", kalman_gain_h, cov_hy)

    mu_v_posterior = mu_h_posterior[:, 1]
    var_v_posterior = cov_h_posterior[:, 1, 1]
    var_v_posterior = np.clip(var_v_posterior, 0.0, None)

    # Step 3: Calculate Posterior Moments for V^2 on valid data
    mu_v2_posterior = mu_v_posterior**2 + var_v_posterior
    var_v2_posterior = 2 * (var_v_posterior**2) + 4 * var_v_posterior * (
        mu_v_posterior**2
    )
    var_v2_posterior = np.clip(var_v2_posterior, 0.0, None)

    # Step 4: Calculate Prior Moments for V^2 and Gain 'k' on valid data
    mu_v2_prior = mu_v2bar_valid
    var_v2bar_valid = np.nan_to_num(
        var_v2bar_valid,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    mu_v2bar_valid = np.nan_to_num(
        mu_v2bar_valid,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    var_v2_prior = 3.0 * var_v2bar_valid
    var_v2_prior += 2.0 * np.square(mu_v2bar_valid)
    var_v2_prior = np.clip(
        var_v2_prior, np.finfo(np.float64).eps, np.finfo(np.float64).max
    )

    k = np.divide(
        var_v2bar_valid,
        var_v2_prior,
        out=np.zeros_like(var_v2bar_valid),
        where=np.isfinite(var_v2_prior) & (var_v2_prior != 0.0),
    )

    # Step 5: Update V2bar to get its Posterior Moments on valid data
    mu_v2bar_posterior_valid = mu_v2bar_valid + k * (mu_v2_posterior - mu_v2_prior)
    var_v2bar_posterior_valid = var_v2bar_valid + k**2 * (
        var_v2_posterior - var_v2_prior
    )
    var_v2bar_posterior_valid = np.clip(var_v2bar_posterior_valid, 0.0, None)

    # Place the calculated posterior values
    mu_v2bar_posterior[valid_indices] = mu_v2bar_posterior_valid
    var_v2bar_posterior[valid_indices] = var_v2bar_posterior_valid

    return (
        mu_v2bar_posterior.astype(out_dtype, copy=False),
        var_v2bar_posterior.astype(out_dtype, copy=False),
    )


# Define a class to store the look_back_buffers
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


# Define a class to handle the lstm state buffers
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
        valid_mask = indices != -1
        valid_indices_to_read = indices[valid_mask]

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


# Define class for storing predictions over all time steps
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

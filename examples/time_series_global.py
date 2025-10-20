import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import List, Optional
import copy

from examples.embedding_loader import EmbeddingLayer, MappedTimeSeriesEmbeddings
from examples.data_loader import (
    GlobalTimeSeriesDataloader,
)
from pytagi import manual_seed
from pytagi import Normalizer as normalizer
import pytagi.metric as metric
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential, EvenExp

# Plotting defaults
import matplotlib.pyplot as plt
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
def prepare_dtls(
    x_file,
    date_file,
    input_seq_len,
    num_features,
    time_covariates,
    scale_method,
    order_mode,
    seed,
    ts_to_use,  # Make it an argument
):
    train_dtl = GlobalTimeSeriesDataloader(
        x_file=x_file[0],
        date_time_file=date_file[0],
        output_col=[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        num_features=num_features,
        stride=1,
        time_covariates=time_covariates,
        keep_last_time_cov=True,
        scale_method=scale_method,
        order_mode=order_mode,
        seed=seed,
        ts_to_use=ts_to_use,
    )

    val_dtl = GlobalTimeSeriesDataloader(
        x_file=x_file[1],
        date_time_file=date_file[1],
        output_col=[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        num_features=num_features,
        stride=1,
        time_covariates=time_covariates,
        keep_last_time_cov=True,
        scale_method=scale_method,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        covariate_means=train_dtl.covariate_means,
        covariate_stds=train_dtl.covariate_stds,
        order_mode=order_mode,
        seed=seed,
        ts_to_use=ts_to_use,
    )
    test_dtl = GlobalTimeSeriesDataloader(
        x_file=x_file[2],
        date_time_file=date_file[2],
        output_col=[0],
        input_seq_len=input_seq_len,
        output_seq_len=1,
        num_features=num_features,
        stride=1,
        time_covariates=time_covariates,
        keep_last_time_cov=True,
        scale_method=scale_method,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        covariate_means=train_dtl.covariate_means,
        covariate_stds=train_dtl.covariate_stds,
        order_mode=order_mode,
        seed=seed,
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
        net.set_threads(8)
    elif device == "cuda":
        net.to_device("cuda")
    out_updater = OutputUpdater(net.device)
    return net, out_updater


def prepare_input(
    x,
    var_x: Optional[np.ndarray],
    look_back_mu: Optional[np.ndarray],
    look_back_var: Optional[np.ndarray],
    indices,
    embeddings: Optional[np.ndarray] = None,
):
    x = np.nan_to_num(x, nan=0.0)
    if var_x is None:
        var_x = np.zeros_like(x, dtype=np.float32)
    input_seq_len = look_back_mu.shape[1]

    if look_back_mu is not None:
        x[:, :input_seq_len] = look_back_mu[indices]
    if look_back_var is not None:
        var_x[:, :input_seq_len] = look_back_var[indices]

    if embeddings is not None:
        embed_mu, embed_var = embeddings(indices)  # shape: (B, embedding_size)
        x = np.concatenate(
            (x, embed_mu), axis=1
        )  # shape: (B, input_seq_len + embedding_size + num_features - 1)
        var_x = np.concatenate(
            (var_x, embed_var), axis=1
        )  # shape: (B, input_seq_len + embedding_size + num_features - 1)

    flat_x = np.concatenate(x, axis=0, dtype=np.float32)
    flat_var = np.concatenate(var_x, axis=0, dtype=np.float32)

    return flat_x, flat_var


# Define a class to store the look_back_buffers
class LookBackBuffer:
    def __init__(self, input_seq_len, nb_ts):
        self.mu = np.full((nb_ts, input_seq_len), np.nan, dtype=np.float32)
        self.var = np.full((nb_ts, input_seq_len), 0.0, dtype=np.float32)
        self.needs_initialization = [True for _ in range(nb_ts)]

    def initialize(self, initial_mu, initial_var, indices):
        for idx, mu, var in zip(indices, initial_mu, initial_var):
            if self.needs_initialization[idx]:
                self.mu[idx] = np.nan_to_num(mu, nan=0.0)
                self.var[idx] = np.nan_to_num(var, nan=0.0)
                self.needs_initialization[idx] = False

    def update(self, new_mu, new_var, indices):
        self.mu[indices] = np.roll(self.mu[indices], -1, axis=1)
        self.var[indices] = np.roll(self.var[indices], -1, axis=1)

        # Update the last column with new values
        self.mu[indices, -1] = new_mu
        self.var[indices, -1] = new_var

    def __call__(self, indices):
        return self.mu[indices], self.var[indices]


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
    nan_indices = np.isnan(y)  # get the indices where y is nan
    y = np.where(nan_indices, m_pred, y)
    var_y = np.where(nan_indices, 0.0, var_y)

    # kalman update
    K = v_pred / (v_pred + var_y)  # Kalman gain
    m_post = m_pred + K * (y - m_pred)  # posterior mean
    v_post = (1.0 - K) * v_pred  # posterior variance

    return m_post, v_post


# Define function to update aleatoric uncertainty
def update_aleatoric_uncertainty(
    mu_z0: np.ndarray,
    var_z0: np.ndarray,
    mu_v2bar: np.ndarray,
    var_v2bar: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    # Handle NaN values in y
    valid_indices = ~np.isnan(y)

    # Initialize posterior arrays as copies of the priors
    mu_v2bar_posterior = np.copy(mu_v2bar)
    var_v2bar_posterior = np.copy(var_v2bar)

    # If all y values are NaN, no update is needed
    if not np.any(valid_indices):
        return mu_v2bar_posterior, var_v2bar_posterior

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

    cov_h = np.zeros((len(mu_z0_valid), 2, 2))
    cov_h[:, 0, 0] = var_z0_valid
    cov_h[:, 1, 1] = var_v

    cov_hy = np.stack([var_z0_valid, var_v], axis=1)

    # Step 2: Calculate Posterior Moments for H using y on valid data
    kalman_gain_h = cov_hy / var_y[:, np.newaxis]
    mu_h_posterior = mu_h + kalman_gain_h * (y_valid - mu_y)[:, np.newaxis]
    cov_h_posterior = cov_h - np.einsum("bi,bj->bij", kalman_gain_h, cov_hy)

    mu_v_posterior = mu_h_posterior[:, 1]
    var_v_posterior = cov_h_posterior[:, 1, 1]

    # Step 3: Calculate Posterior Moments for V^2 on valid data
    mu_v2_posterior = mu_v_posterior**2 + var_v_posterior
    var_v2_posterior = 2 * (var_v_posterior**2) + 4 * var_v_posterior * (
        mu_v_posterior**2
    )

    # Step 4: Calculate Prior Moments for V^2 and Gain 'k' on valid data
    mu_v2_prior = mu_v2bar_valid
    var_v2_prior = 3 * var_v2bar_valid + 2 * (mu_v2bar_valid**2)

    k = np.divide(
        var_v2bar_valid,
        var_v2_prior,
        out=np.zeros_like(var_v2bar_valid),
        where=var_v2_prior != 0,
    )

    # Step 5: Update V2bar to get its Posterior Moments on valid data
    mu_v2bar_posterior_valid = mu_v2bar_valid + k * (mu_v2_posterior - mu_v2_prior)
    var_v2bar_posterior_valid = var_v2bar_valid + k**2 * (
        var_v2_posterior - var_v2_prior
    )

    # Place the calculated posterior values
    mu_v2bar_posterior[valid_indices] = mu_v2bar_posterior_valid
    var_v2bar_posterior[valid_indices] = var_v2bar_posterior_valid

    return mu_v2bar_posterior, var_v2bar_posterior


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
        """
        net_states = net.get_lstm_states()
        batch_size = len(indices)

        unpacked_states = self._unpack_net_states(net_states, batch_size)

        # Vectorized update using advanced indexing
        for layer_idx, components in unpacked_states.items():
            self.states[layer_idx]["mu_h"][indices] = components["mu_h"]
            self.states[layer_idx]["var_h"][indices] = components["var_h"]
            self.states[layer_idx]["mu_c"][indices] = components["mu_c"]
            self.states[layer_idx]["var_c"][indices] = components["var_c"]

    def set_states_on_net(self, indices: np.ndarray, net):
        """
        Retrieves the stored states for the given indices and sets them on the network.
        """
        batch_states = {}
        # Vectorized retrieval using advanced indexing
        for layer_idx, components in self.states.items():
            batch_states[layer_idx] = {
                "mu_h": components["mu_h"][indices],
                "var_h": components["var_h"][indices],
                "mu_c": components["mu_c"][indices],
                "var_c": components["var_c"][indices],
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
        # This is the optimized part, replacing the slow for-loop.
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


class Config:
    def __init__(self):
        # Seed for reproducibility
        self.seed: int = 1

        # Set data paths
        self.x_train = "data/hq/train_1.0/split_train_values.csv"
        self.dates_train = "data/hq/train_1.0/split_train_datetimes.csv"
        self.x_val = "data/hq/split_val_values.csv"
        self.dates_val = "data/hq/split_val_datetimes.csv"
        self.x_test = "data/hq/split_test_values.csv"
        self.dates_test = "data/hq/split_test_datetimes.csv"

        # Set data_loader parameters
        self.num_features: int = 2
        self.time_covariates: list = ["week_of_year"]
        self.scale_method: str = "standard"
        self.order_mode: str = "by_window"
        self.input_seq_len: int = 52
        self.batch_size: int = 127
        self.output_col: list = [0]
        self.ts_to_use: Optional[List[int]] = [i for i in range(127)]  # Use all series

        # 1. For standard (one-per-series) embeddings:
        self.embedding_size: Optional[int] = None
        self.embedding_initializer: str = "normal"

        # 2. For mapped (shared) embeddings:
        self.embedding_map_dir: Optional[str] = None
        self.embedding_map_sizes = {
            "dam_id": 3,
            "dam_type_id": 3,
            "sensor_type_id": 3,
            "direction_id": 3,
            "sensor_id": 3,
        }
        self.embedding_map_initializer = {
            "dam_id": "normal",
            "dam_type_id": "normal",
            "sensor_type_id": "normal",
            "direction_id": "normal",
            "sensor_id": "normal",
        }
        self.embedding_map_labels = {
            "dam_id": ["DRU", "GOU", "LGA", "LTU", "MAT", "M5"],
            "dam_type_id": ["Run-of-River", "Reservoir"],
            "sensor_type_id": ["PIZ", "EXT", "PEN"],
            "direction_id": ["NA", "X", "Y", "Z"],
            "sensor_id": [f"sensor_{i}" for i in self.ts_to_use],
        }

        # Set model parameters
        self.Sigma_v_bounds: tuple = (None, None)
        self.decaying_factor: float = 0.999
        self.device: str = "cuda"

        # Set training parameters
        self.num_epochs: int = 100
        self.early_stopping_criteria: str = "rmse"
        self.patience: int = 10
        self.min_delta: float = 1e-4
        self.warmup_epochs: int = 0
        self.shuffle_train_windows: bool = False

        # Set evaluation parameters
        self.eval_plots: bool = True
        self.eval_metrics: bool = True
        self.seansonal_period: int = 52

    @property
    def x_file(self) -> list:
        """Dynamically creates the list of x files."""
        return [self.x_train, self.x_val, self.x_test]

    @property
    def date_file(self) -> list:
        """Dynamically creates the list of date files."""
        return [self.dates_train, self.dates_val, self.dates_test]

    @property
    def use_AGVI(self) -> bool:
        """Determines whether to use AGVI based on Sigma_v_bounds."""
        return self.Sigma_v_bounds[0] is None and self.Sigma_v_bounds[1] is None

    @property
    def use_mapped_embeddings(self) -> bool:
        """True if mapped embeddings are configured."""
        return self.embedding_map_dir is not None

    @property
    def use_standard_embeddings(self) -> bool:
        """True if standard (one-per-series) embeddings are configured."""
        return (
            not self.use_mapped_embeddings
            and self.embedding_size is not None
            and self.embedding_size > 0
        )

    @property
    def total_embedding_size(self) -> int:
        """Calculates the total embedding dimension based on configuration."""
        if self.use_mapped_embeddings:
            return sum(self.embedding_map_sizes.values())
        if self.use_standard_embeddings:
            return self.embedding_size
        return 0  # No embeddings

    @property
    def input_size(self) -> int:
        """Calculates the total input size for the model."""
        base_size = self.num_features + self.input_seq_len - 1
        return base_size + self.total_embedding_size

    @property
    def nb_ts(self) -> int:
        """Calculates the number of time series based on ts_to_use."""
        if self.ts_to_use is not None:
            return len(self.ts_to_use)
        # This case should not be hit if ts_to_use is always populated
        # Re-using dataloader's logic might be more robust
        return 1

    @property
    def plot_embeddings(self) -> bool:
        """
        Determines whether to plot embeddings.
        """
        return self.use_standard_embeddings

    def display(self):
        print("\nConfiguration:")
        # Display both regular attributes and properties
        for name in dir(self):
            if not name.startswith("_") and not callable(getattr(self, name)):
                print(f"  {name}: {getattr(self, name)}")
        print("\n")

    def save(self, path):
        with open(path, "w") as f:
            for name in dir(self):
                if not name.startswith("_") and not callable(getattr(self, name)):
                    f.write(f"{name}: {getattr(self, name)}\n")


def train_global_model(config, experiment_name: Optional[str] = None):

    # Create output directory
    output_dir = f"out/{experiment_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Display and save configuration
    config.display()
    config.save(os.path.join(output_dir, "config.txt"))

    # Prepare data loaders
    train_dtl, val_dtl, test_dtl = prepare_dtls(
        x_file=config.x_file,
        date_file=config.date_file,
        input_seq_len=config.input_seq_len,
        num_features=config.num_features,
        time_covariates=config.time_covariates,
        scale_method=config.scale_method,
        order_mode=config.order_mode,
        seed=config.seed,
        ts_to_use=config.ts_to_use,
    )

    # Embeddings
    embeddings = None  # Initialize as None
    embedding_dir = os.path.join(output_dir, "embeddings")

    if config.use_mapped_embeddings:
        print(
            f"Using MappedTimeSeriesEmbeddings. Total embedding size: {config.total_embedding_size}"
        )
        embeddings = MappedTimeSeriesEmbeddings(
            map_file_path=config.embedding_map_dir,
            embedding_sizes=config.embedding_map_sizes,
            encoding_types=config.embedding_map_initializer,
            seed=config.seed,
        )
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)
        # Mapped save uses a file prefix
        embeddings.save(os.path.join(embedding_dir, "embeddings_start"))

    elif config.use_standard_embeddings:
        print(
            f"Using standard EmbeddingLayer. Embedding size: {config.total_embedding_size}"
        )
        embeddings = EmbeddingLayer(
            (config.nb_ts, config.embedding_size),
            encoding_type=config.embedding_initializer,
            seed=config.seed,
        )
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)
        # Standard save uses a full file name
        embeddings.save(os.path.join(embedding_dir, "embeddings_start.npz"))

    else:
        print("No embeddings will be used.")

    # Build model
    net, output_updater = build_model(
        input_size=config.input_size,
        use_AGVI=config.use_AGVI,
        seed=config.seed,
        device=config.device,
    )

    # Enable input state updates only if embeddings are being used
    if embeddings is not None:
        net.input_state_update = True

    # Initalize states
    train_states = States(nb_ts=config.nb_ts, total_time_steps=train_dtl.max_len)
    val_states = States(nb_ts=config.nb_ts, total_time_steps=val_dtl.max_len)
    test_states = States(nb_ts=config.nb_ts, total_time_steps=test_dtl.max_len)

    # Create progress bar
    pbar = tqdm(range(config.num_epochs), desc="Epochs")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        criteria=config.early_stopping_criteria,
        patience=config.patience,
        min_delta=config.min_delta,
        warmup_epochs=config.warmup_epochs,
    )

    # Prepare decaying sigma_v if not using AGVI
    if not config.use_AGVI:
        sigma_start, sigma_end = config.Sigma_v_bounds
        if sigma_start is None or sigma_end is None:
            raise ValueError("Sigma_v_bounds must be defined when AGVI is disabled.")
        sigma_start = float(sigma_start)
        sigma_end = float(sigma_end)
        if config.num_epochs <= 1:
            decaying_sigma_v = [sigma_start]
        else:
            decay_factor = float(config.decaying_factor)
            exponents = decay_factor ** np.arange(config.num_epochs, dtype=np.float32)
            if np.isclose(exponents[0], exponents[-1]) or decay_factor <= 0.0:
                weights = np.linspace(1.0, 0.0, config.num_epochs, dtype=np.float32)
            else:
                weights = (exponents - exponents[-1]) / (exponents[0] - exponents[-1])
            decaying_sigma_v = (
                sigma_end + (sigma_start - sigma_end) * weights
            ).tolist()

    # --- Training loop ---
    for epoch in pbar:
        # net.train()
        train_mse = []
        train_log_lik = []

        train_batch_iter = train_dtl.create_data_loader(
            batch_size=config.batch_size,
            shuffle=False,
            include_ids=True,
            shuffle_series_blocks=config.shuffle_train_windows,
        )

        # Initialize look-back buffer and LSTM state container
        look_back_buffer = LookBackBuffer(
            input_seq_len=config.input_seq_len, nb_ts=config.nb_ts
        )
        lstm_state_container = LSTMStateContainer(
            num_series=config.nb_ts, layer_state_shapes={0: 40, 1: 40}
        )

        # get current sigma_v if not using AGVI
        if not config.use_AGVI:
            sigma_v = decaying_sigma_v[epoch]

        for x, y, ts_id, w_id in train_batch_iter:

            # get current batch size and indices
            B = x.shape[0]
            indices = ts_id
            time_steps = w_id

            # set LSTM states for the current batch
            lstm_state_container.set_states_on_net(indices, net)

            # prepare obsevation noise matrix
            if not config.use_AGVI:
                var_y = np.full(
                    (B * len(config.output_col),), sigma_v**2, dtype=np.float32
                )

            # prepare look_back buffer
            if any(look_back_buffer.needs_initialization[i] for i in indices):
                look_back_buffer.initialize(
                    initial_mu=x[:, : config.input_seq_len],
                    initial_var=np.zeros_like(
                        x[:, : config.input_seq_len], dtype=np.float32
                    ),
                    indices=indices,
                )

            # prepare input
            x, var_x = prepare_input(
                x=x,
                var_x=None,
                look_back_mu=look_back_buffer.mu,
                look_back_var=look_back_buffer.var,
                indices=indices,
                embeddings=embeddings,  # Pass the object directly
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

            s_pred = np.sqrt(v_pred + var_y)

            # Compute metrics
            mask = ~np.isnan(y.flatten())
            y_masked = y.flatten()[mask]
            m_pred_masked = m_pred[mask]
            s_pred_masked = s_pred[mask]

            if y_masked.size > 0:
                batch_mse = metric.rmse(m_pred_masked, y_masked)
                batch_log_lik = metric.log_likelihood(
                    m_pred_masked, y_masked, s_pred_masked
                )
                train_mse.append(batch_mse)
                train_log_lik.append(batch_log_lik)

            # Store predictions
            train_states.update(
                new_mu=m_pred.reshape(B, -1),
                new_std=s_pred.reshape(B, -1),
                indices=indices,
                time_step=time_steps,
            )

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

            # Update embeddings if used
            if embeddings is not None:
                mu_delta, var_delta = net.get_input_states()

                mu_delta = mu_delta * var_x
                var_delta = var_x * var_delta * var_x

                mu_delta = mu_delta.reshape(B, -1)
                var_delta = var_delta.reshape(B, -1)

                # Get the slice corresponding to all embeddings
                total_emb_size = config.total_embedding_size
                mu_delta_slice = mu_delta[:, -total_emb_size:]
                var_delta_slice = var_delta[:, -total_emb_size:]

                # Update embeddings
                embeddings.update(
                    indices,
                    mu_delta_slice,
                    var_delta_slice,
                )

            # update aleatoric uncertainty if using AGVI
            if config.use_AGVI:
                (
                    mu_v2bar_post,
                    _,
                ) = update_aleatoric_uncertainty(
                    mu_z0=m_pred,
                    var_z0=v_pred,
                    mu_v2bar=flat_m[1::2],
                    var_v2bar=flat_v[1::2],
                    y=y.flatten(),
                )
                var_y = mu_v2bar_post  # updated aleatoric uncertainty

            # Update LSTM states for the current batch
            lstm_state_container.update_states_from_net(indices, net)

            # Update look_back buffer
            look_back_buffer.update(
                new_mu=m_post.reshape(B, -1)[:, -1],
                new_var=(v_post + var_y).reshape(B, -1)[:, -1],
                indices=indices,
            )

        # End of epoch
        train_mse = np.mean(train_mse)
        train_log_lik = np.mean(train_log_lik)

        # Validation
        # net.eval()
        val_mse = []
        val_log_lik = []

        # reset LSTM states
        net.reset_lstm_states()

        # reset look-back buffer
        look_back_buffer.needs_initialization = [True for _ in range(config.nb_ts)]

        val_batch_iter = val_dtl.create_data_loader(
            batch_size=config.batch_size,
            shuffle=False,
            include_ids=True,
        )

        for x, y, ts_id, w_id in val_batch_iter:

            # get current batch size and indices
            B = x.shape[0]
            indices = ts_id
            time_steps = w_id

            # set LSTM states for the current batch
            lstm_state_container.set_states_on_net(indices, net)

            # prepare obsevation noise matrix
            if not config.use_AGVI:
                var_y = np.full(
                    (B * len(config.output_col),), sigma_v**2, dtype=np.float32
                )

            # prepare look_back buffer
            if any(look_back_buffer.needs_initialization[i] for i in indices):
                look_back_buffer.initialize(
                    initial_mu=x[:, : config.input_seq_len],
                    initial_var=np.zeros_like(
                        x[:, : config.input_seq_len], dtype=np.float32
                    ),
                    indices=indices,
                )

            # prepare input
            x, var_x = prepare_input(
                x=x,
                var_x=None,
                look_back_mu=look_back_buffer.mu,
                look_back_var=look_back_buffer.var,
                indices=indices,
                embeddings=embeddings,  # Pass the object directly
            )

            # Feedforward
            m_pred, v_pred = net(x, var_x)

            # Update LSTM states for the current batch
            lstm_state_container.update_states_from_net(indices, net)

            # Specific to AGVI
            if config.use_AGVI:
                flat_m = np.ravel(m_pred)
                flat_v = np.ravel(v_pred)

                m_pred = flat_m[::2]  # even indices
                v_pred = flat_v[::2]  # even indices
                var_y = flat_m[1::2]  # odd indices var_v

            v_pred_total = v_pred + var_y
            s_pred = np.sqrt(v_pred_total)

            # Compute metrics
            mask = ~np.isnan(y.flatten())
            y_masked = y.flatten()[mask]
            m_pred_masked = m_pred[mask]
            s_pred_masked = s_pred[mask]

            if y_masked.size > 0:
                batch_mse = metric.rmse(m_pred_masked, y_masked)
                batch_log_lik = metric.log_likelihood(
                    m_pred_masked, y_masked, s_pred_masked
                )
                val_mse.append(batch_mse)
                val_log_lik.append(batch_log_lik)

            # Store predictions
            val_states.update(
                new_mu=m_pred.reshape(B, -1)[:, -1],
                new_std=s_pred.reshape(B, -1)[:, -1],
                indices=indices,
                time_step=time_steps,
            )

            # Update look_back buffer
            look_back_buffer.update(
                new_mu=m_pred.reshape(B, -1)[:, -1],
                new_var=v_pred_total.reshape(B, -1)[:, -1],  # Use total variance
                indices=indices,
            )

        # End of epoch
        val_mse = np.mean(val_mse)
        val_log_lik = np.mean(val_log_lik)

        # Update progress bar
        sigma_v_str = (
            f"{sigma_v:.4f}"
            if not config.use_AGVI and sigma_v is not None
            else "N/A (AGVI)"
        )
        pbar.set_postfix(
            {
                "Train RMSE": f"{train_mse:.4f}",
                "Val RMSE": f"{val_mse:.4f}",
                "Train LogLik": f"{train_log_lik:.4f}",
                "Val LogLik": f"{val_log_lik:.4f}",
                "Sigma_v": sigma_v_str,
            }
        )

        # Check for early stopping
        val_score = (
            val_log_lik if config.early_stopping_criteria == "log_lik" else val_mse
        )
        if early_stopping(
            val_score,
            net,
            look_back_buffer,
            lstm_state_container,
            train_states,
            val_states,
            sigma_v if not config.use_AGVI else None,
            embeddings,  # Pass the object directly
        ):
            print(f"Early stopping at epoch {epoch+1}")
            net.load_state_dict(early_stopping.best_state)
            look_back_buffer = early_stopping.best_look_back_buffer
            lstm_state_container = early_stopping.best_lstm_state_container
            train_states = early_stopping.train_states
            val_states = early_stopping.val_states
            if not config.use_AGVI:
                sigma_v = early_stopping.best_sigma_v

            # Restore best embeddings
            embeddings = early_stopping.best_embeddings
            break

    # Save best model
    net.save(os.path.join(output_dir, "param/model.pth"))

    # Save best embeddings based on type
    if embeddings is not None:
        if config.use_mapped_embeddings:
            embeddings.save(os.path.join(embedding_dir, "embeddings_final"))
        elif config.use_standard_embeddings:
            embeddings.save(os.path.join(embedding_dir, "embeddings_final.npz"))

    # --- Testing ---
    # net.eval()

    # reset LSTM states
    net.reset_lstm_states()

    # reset look-back buffer
    look_back_buffer.needs_initialization = [True for _ in range(config.nb_ts)]

    test_batch_iter = test_dtl.create_data_loader(
        batch_size=config.batch_size,
        shuffle=False,
        include_ids=True,
    )

    for x, y, ts_id, w_id in test_batch_iter:

        # get current batch size and indices
        B = x.shape[0]
        indices = ts_id.tolist()
        time_steps = w_id.tolist()

        # set LSTM states for the current batch
        lstm_state_container.set_states_on_net(indices, net)

        # prepare obsevation noise matrix
        if not config.use_AGVI:
            var_y = np.full((B * len(config.output_col),), sigma_v**2, dtype=np.float32)

        # prepare look_back buffer
        if any(look_back_buffer.needs_initialization[i] for i in indices):
            look_back_buffer.initialize(
                initial_mu=x[:, : config.input_seq_len],
                initial_var=np.zeros_like(
                    x[:, : config.input_seq_len], dtype=np.float32
                ),
                indices=indices,
            )

        # prepare input
        x, var_x = prepare_input(
            x=x,
            var_x=None,
            look_back_mu=look_back_buffer.mu,
            look_back_var=look_back_buffer.var,
            indices=indices,
            embeddings=embeddings,  # Pass the object directly
        )

        # Feedforward
        m_pred, v_pred = net(x, var_x)

        # Update LSTM states for the current batch
        lstm_state_container.update_states_from_net(indices, net)

        # Specific to AGVI
        if config.use_AGVI:
            flat_m = np.ravel(m_pred)
            flat_v = np.ravel(v_pred)

            m_pred = flat_m[::2]  # even indices
            v_pred = flat_v[::2]  # even indices
            var_y = flat_m[1::2]  # odd indices var_v

        v_pred_total = v_pred + var_y
        s_pred = np.sqrt(v_pred_total)

        # Store predictions
        test_states.update(
            new_mu=m_pred.reshape(B, -1)[:, -1],
            new_std=s_pred.reshape(B, -1)[:, -1],
            indices=indices,
            time_step=time_steps,
        )

        # Update look_back buffer
        look_back_buffer.update(
            new_mu=m_pred.reshape(B, -1)[:, -1],
            new_var=v_pred_total.reshape(B, -1)[:, -1],  # Use total variance
            indices=indices,
        )

    # End of epoch
    net.reset_lstm_states()

    # Run over each time series and re_scale it
    for i in range(config.nb_ts):
        # get mean and std
        mean = train_dtl.x_mean[i]
        std = train_dtl.x_std[i]

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


def eval_global_model(config, experiment_name: Optional[str] = None):
    """Evaluates forecasts stored in the .npz format."""

    from pathlib import Path

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
    test_mse_list = []
    test_p50_list = []
    test_p90_list = []
    test_mase_list = []

    # Iterate over each time series and calculate metrics
    for i in tqdm(range(config.nb_ts), desc="Evaluating series"):

        # Get true values
        yt_train, yt_val, yt_test = (
            _trim_trailing_nans(true_train[config.input_seq_len :, i]),
            _trim_trailing_nans(true_val[config.input_seq_len :, i]),
            _trim_trailing_nans(true_test[config.input_seq_len :, i]),
        )
        yt_full = np.concatenate([yt_train, yt_val, yt_test])

        # get expected value
        ypred_train = train_states["mu"][i][: len(yt_train)]
        ypred_val = val_states["mu"][i][: len(yt_val)]
        ypred_test = test_states["mu"][i][: len(yt_test)]
        ypred_full = np.concatenate([ypred_train, ypred_val, ypred_test])

        # get std
        spred_train = train_states["std"][i][: len(yt_train)]
        spred_val = val_states["std"][i][: len(yt_val)]
        spred_test = test_states["std"][i][: len(yt_test)]
        spred_full = np.concatenate([spred_train, spred_val, spred_test])

        # Store split indices
        val_test_indices = (len(yt_train), len(yt_train) + len(yt_val))

        # --- Plotting ---
        if config.eval_plots:
            plot_series(
                ts_idx=i,
                y_true=yt_full,
                y_pred=ypred_full,
                s_pred=spred_full,
                out_dir=input_dir / "figures",
                val_test_indices=val_test_indices,
                std_factor=1,
            )

        # --- Metrics ---
        if config.eval_metrics:
            mask_test = (
                np.isfinite(yt_test) & np.isfinite(ypred_test) & np.isfinite(spred_test)
            )

            if np.any(mask_test):
                y_true = yt_test[mask_test]
                y_pred = ypred_test[mask_test]
                s_pred = np.clip(spred_test[mask_test], 1e-6, None)

                test_rmse = metric.rmse(y_pred, y_true)
                test_log_lik = metric.log_likelihood(y_pred, y_true, s_pred)
                test_mse = metric.mse(y_pred, y_true)

                denom = np.sum(np.abs(y_true))
                if denom == 0:
                    test_p50 = np.nan
                    test_p90 = np.nan
                else:
                    test_p50 = metric.p50(y_true, y_pred)
                    test_p90 = metric.p90(y_true, y_pred, s_pred)

                test_mase = metric.mase(
                    y_true, y_pred, yt_train, config.seansonal_period
                )
            else:
                test_rmse = np.nan
                test_log_lik = np.nan
                test_mse = np.nan
                test_p50 = np.nan
                test_p90 = np.nan
                test_mase = np.nan

            # Append to lists
            test_rmse_list.append(test_rmse)
            test_log_lik_list.append(test_log_lik)
            test_mse_list.append(test_mse)
            test_p50_list.append(test_p50)
            test_p90_list.append(test_p90)
            test_mase_list.append(test_mase)

    # Calculate overall metrics
    if config.eval_metrics:
        overall_rmse = np.nanmean(test_rmse_list)
        overall_log_lik = np.nanmean(test_log_lik_list)
        overall_mse = np.nanmean(test_mse_list)
        overall_p50 = np.nanmean(test_p50_list)
        overall_p90 = np.nanmean(test_p90_list)
        overall_mase = np.nanmean(test_mase_list)

        # save metrics to a table per series and overall
        with open(input_dir / "evaluation_metrics.txt", "w") as f:
            f.write("Series_ID,RMSE,LogLik,MSE,P50,P90,MASE\n")
            for i in range(config.nb_ts):
                f.write(
                    f"{config.ts_to_use[i]},{test_rmse_list[i]:.4f},{test_log_lik_list[i]:.4f},"
                    f"{test_mse_list[i]:.4f},{test_p50_list[i]:.4f},"
                    f"{test_p90_list[i]:.4f},{test_mase_list[i]:.4f}\n"
                )
            f.write(
                f"Overall,{overall_rmse:.4f},{overall_log_lik:.4f},"
                f"{overall_mse:.4f},{overall_p50:.4f},"
                f"{overall_p90:.4f},{overall_mase:.4f}\n"
            )

    # Display and plot embeddings if used
    def _cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = emb / norms
        return normalized @ normalized.T

    def _plot_similarity(
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

    def _bhattacharyya_distance_matrix(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
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

    # Check if *any* embeddings were used
    if config.total_embedding_size > 0:
        embedding_dir = input_dir / "embeddings"

        if config.use_standard_embeddings:
            print("Plotting standard (one-per-series) embeddings...")
            try:
                start_data = np.load(embedding_dir / "embeddings_start.npz")
                final_data = np.load(embedding_dir / "embeddings_final.npz")

                start_embeddings_mu = start_data["mu"]
                start_embeddings_var = start_data["var"]
                final_embeddings_mu = final_data["mu"]
                final_embeddings_var = final_data["var"]

                # Labels for standard embeddings are the time series IDs
                labels = [str(ts_id) for ts_id in config.ts_to_use]

                plot_embeddings(
                    start_embeddings_mu,
                    config.nb_ts,
                    input_dir,
                    "embeddings/embeddings_mu_pca_start.png",
                    labels=labels,
                )
                plot_embeddings(
                    final_embeddings_mu,
                    config.nb_ts,
                    input_dir,
                    "embeddings/embeddings_mu_pca_final.png",
                    labels=labels,
                )

                start_similarity = _cosine_similarity_matrix(start_embeddings_mu)
                final_similarity = _cosine_similarity_matrix(final_embeddings_mu)

                _plot_similarity(
                    start_similarity,
                    embedding_dir / "embeddings_cosine_similarity_start.png",
                    "Cosine Similarity (Start)",
                )
                _plot_similarity(
                    final_similarity,
                    embedding_dir / "embeddings_cosine_similarity_final.png",
                    "Cosine Similarity (Final)",
                )

                start_bhattacharyya = _bhattacharyya_distance_matrix(
                    start_embeddings_mu, start_embeddings_var
                )
                final_bhattacharyya = _bhattacharyya_distance_matrix(
                    final_embeddings_mu, final_embeddings_var
                )

                bhatt_vmax = float(
                    max(
                        np.nanmax(start_bhattacharyya),
                        np.nanmax(final_bhattacharyya),
                        1e-12,
                    )
                )

                _plot_similarity(
                    start_bhattacharyya,
                    embedding_dir / "embeddings_bhattacharyya_distance_start.png",
                    "Bhattacharyya Distance (Start)",
                    vmin=0.0,
                    vmax=bhatt_vmax,
                )
                _plot_similarity(
                    final_bhattacharyya,
                    embedding_dir / "embeddings_bhattacharyya_distance_final.png",
                    "Bhattacharyya Distance (Final)",
                    vmin=0.0,
                    vmax=bhatt_vmax,
                )
            except FileNotFoundError as e:
                print(
                    f"Warning: Could not plot standard embeddings. File not found: {e}"
                )
            except Exception as e:
                print(f"Warning: Failed to plot standard embeddings. Error: {e}")

        elif config.use_mapped_embeddings:
            print("Plotting mapped (categorical) embeddings...")

            # --- 1. Plot per-category embeddings ---
            categories = sorted(list(config.embedding_map_sizes.keys()))

            # Store loaded embeddings to use for stitching
            loaded_start_mus = {}
            loaded_start_vars = {}
            loaded_final_mus = {}
            loaded_final_vars = {}

            for category in categories:
                print(f"  Plotting for category: {category}")
                try:
                    start_data = np.load(
                        embedding_dir / f"embeddings_start_{category}.npz"
                    )
                    final_data = np.load(
                        embedding_dir / f"embeddings_final_{category}.npz"
                    )

                    start_mu = start_data["mu"]
                    start_var = start_data["var"]
                    final_mu = final_data["mu"]
                    final_var = final_data["var"]

                    # Store for stitching
                    loaded_start_mus[category] = start_mu
                    loaded_start_vars[category] = start_var
                    loaded_final_mus[category] = final_mu
                    loaded_final_vars[category] = final_var

                    n_entities = start_mu.shape[0]
                    labels = config.embedding_map_labels[category]

                    # Create sub-directory
                    category_plot_dir = embedding_dir / category
                    category_plot_dir.mkdir(parents=True, exist_ok=True)

                    # Plot PCA
                    plot_embeddings(
                        start_mu,
                        n_entities,
                        input_dir,  # base dir
                        f"embeddings/{category}/pca_start.png",
                        labels=labels,
                    )
                    plot_embeddings(
                        final_mu,
                        n_entities,
                        input_dir,  # base dir
                        f"embeddings/{category}/pca_final.png",
                        labels=labels,
                    )

                    # Plot Cosine Similarity
                    start_similarity = _cosine_similarity_matrix(start_mu)
                    final_similarity = _cosine_similarity_matrix(final_mu)

                    _plot_similarity(
                        start_similarity,
                        category_plot_dir / "cosine_similarity_start.png",
                        f"Cosine Similarity (Start) - {category}",
                        labels=labels,
                    )
                    _plot_similarity(
                        final_similarity,
                        category_plot_dir / "cosine_similarity_final.png",
                        f"Cosine Similarity (Final) - {category}",
                        labels=labels,
                    )

                    # Plot Bhattacharyya Distance
                    start_bhat = _bhattacharyya_distance_matrix(start_mu, start_var)
                    final_bhat = _bhattacharyya_distance_matrix(final_mu, final_var)
                    bhat_vmax = float(
                        max(np.nanmax(start_bhat), np.nanmax(final_bhat), 1e-12)
                    )

                    _plot_similarity(
                        start_bhat,
                        category_plot_dir / "bhattacharyya_distance_start.png",
                        f"Bhattacharyya Distance (Start) - {category}",
                        vmin=0.0,
                        vmax=bhat_vmax,
                        labels=labels,
                    )
                    _plot_similarity(
                        final_bhat,
                        category_plot_dir / "bhattacharyya_distance_final.png",
                        f"Bhattacharyya Distance (Final) - {category}",
                        vmin=0.0,
                        vmax=bhat_vmax,
                        labels=labels,
                    )
                except FileNotFoundError as e:
                    print(
                        f"  Warning: Could not plot category {category}. File not found: {e}"
                    )
                    continue  # Skip to next category
                except Exception as e:
                    print(f"  Warning: Failed to plot category {category}. Error: {e}")
                    continue

            # --- 2. Stitch Embeddings ---
            print("Stitching full time series embeddings for plotting...")
            try:
                # Load and filter map to the series we used, in the correct order
                if not os.path.exists(config.embedding_map_dir):
                    raise FileNotFoundError(
                        f"Map file not found: {config.embedding_map_dir}"
                    )

                map_df = pd.read_csv(config.embedding_map_dir).set_index("ts_id")

                if config.ts_to_use is None:
                    raise ValueError(
                        "config.ts_to_use is None, cannot stitch embeddings."
                    )

                # Re-order map based on ts_to_use
                map_df_ordered = map_df.loc[config.ts_to_use]

                # Initialize stitched matrices
                mu_stitched_start = np.zeros(
                    (config.nb_ts, config.total_embedding_size), dtype=np.float32
                )
                var_stitched_start = np.zeros_like(mu_stitched_start)
                mu_stitched_final = np.zeros_like(mu_stitched_start)
                var_stitched_final = np.zeros_like(mu_stitched_start)

                current_offset = 0
                for category in categories:  # categories is already sorted
                    if category not in loaded_start_mus:
                        print(
                            f"  Skipping category {category} in stitching (was not loaded)."
                        )
                        # Need to advance offset!
                        current_offset += config.embedding_map_sizes[category]
                        continue

                    cat_size = config.embedding_map_sizes[category]

                    # Get indices from the ordered map
                    cat_indices = map_df_ordered[category].values

                    # Pull embeddings using the indices
                    mu_stitched_start[:, current_offset : current_offset + cat_size] = (
                        loaded_start_mus[category][cat_indices]
                    )
                    var_stitched_start[
                        :, current_offset : current_offset + cat_size
                    ] = loaded_start_vars[category][cat_indices]
                    mu_stitched_final[:, current_offset : current_offset + cat_size] = (
                        loaded_final_mus[category][cat_indices]
                    )
                    var_stitched_final[
                        :, current_offset : current_offset + cat_size
                    ] = loaded_final_vars[category][cat_indices]

                    current_offset += cat_size

                # --- 3. Plot Stitched Embeddings ---
                print("Plotting stitched (full) time series embeddings...")
                labels = [str(ts_id) for ts_id in config.ts_to_use]

                # Plot PCA
                plot_embeddings(
                    mu_stitched_start,
                    config.nb_ts,
                    input_dir,
                    "embeddings/embeddings_mu_pca_start_stitched.png",
                    labels=labels,
                )
                plot_embeddings(
                    mu_stitched_final,
                    config.nb_ts,
                    input_dir,
                    "embeddings/embeddings_mu_pca_final_stitched.png",
                    labels=labels,
                )

                # Plot Cosine Similarity
                start_similarity = _cosine_similarity_matrix(mu_stitched_start)
                final_similarity = _cosine_similarity_matrix(mu_stitched_final)

                _plot_similarity(
                    start_similarity,
                    embedding_dir / "embeddings_cosine_similarity_start_stitched.png",
                    "Cosine Similarity (Start) - Stitched",
                )
                _plot_similarity(
                    final_similarity,
                    embedding_dir / "embeddings_cosine_similarity_final_stitched.png",
                    "Cosine Similarity (Final) - Stitched",
                )

                # Plot Bhattacharyya Distance
                start_bhat = _bhattacharyya_distance_matrix(
                    mu_stitched_start, var_stitched_start
                )
                final_bhat = _bhattacharyya_distance_matrix(
                    mu_stitched_final, var_stitched_final
                )
                bhat_vmax = float(
                    max(np.nanmax(start_bhat), np.nanmax(final_bhat), 1e-12)
                )

                _plot_similarity(
                    start_bhat,
                    embedding_dir
                    / "embeddings_bhattacharyya_distance_start_stitched.png",
                    "Bhattacharyya Distance (Start) - Stitched",
                    vmin=0.0,
                    vmax=bhat_vmax,
                )
                _plot_similarity(
                    final_bhat,
                    embedding_dir
                    / "embeddings_bhattacharyya_distance_final_stitched.png",
                    "Bhattacharyya Distance (Final) - Stitched",
                    vmin=0.0,
                    vmax=bhat_vmax,
                )

            except Exception as e:
                print(
                    f"  Warning: Failed to stitch and plot full embeddings. Error: {e}"
                )


def main(Train=True, Eval=True):
    list_of_seeds = [1, 42, 235, 1234, 2024]
    list_of_experiments = ["train30", "train40", "train60", "train80", "train100"]

    for seed in list_of_seeds:
        for exp in list_of_experiments:
            print(f"Running experiment: {exp} with seed {seed}")

            # Define experiment name
            experiment_name = f"seed{seed}/{exp}/experiment01_global_model"

            # Create configuration
            config = Config()
            config.seed = seed
            config.batch_size = 16
            config.x_train = f"data/hq/{exp}/split_train_values.csv"
            config.dates_train = f"data/hq/{exp}/split_train_datetimes.csv"

            if Train:
                # Train model
                train_global_model(config, experiment_name=experiment_name)

            if Eval:
                # Evaluate model
                eval_global_model(config, experiment_name=experiment_name)


if __name__ == "__main__":
    main()

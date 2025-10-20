import os
import numpy as np
from tqdm import tqdm
from typing import List, Optional
import copy

from examples.data_loader import (
    TimeSeriesDataloader,
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
    ):
        self.best_score = current_score
        self.best_state = copy.deepcopy(model.state_dict())
        self.best_look_back_buffer = copy.deepcopy(look_back_buffer)
        self.best_lstm_state_container = copy.deepcopy(lstm_state_container)
        self.train_states = copy.deepcopy(train_states)
        self.val_states = copy.deepcopy(val_states)
        self.best_sigma_v = sigma_v

    def __call__(
        self,
        current_score,
        model,
        look_back_buffer,
        lstm_state_container,
        train_states,
        val_states,
        sigma_v,
    ):
        self.epoch_count += 1
        current_score = self._as_float(current_score)

        if self.best_state is None or self._is_improvement(current_score):
            self._store_checkpoint(
                current_score,
                model,
                look_back_buffer,
                lstm_state_container,
                train_states,
                val_states,
                sigma_v,
            )
            self.counter = 0
            return False  # Not early stopping

        if self.epoch_count <= self.warmup_epochs:
            return False  # Still in warmup period

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
        self.mu[indices, time_step] = new_mu.flatten()[0]
        self.std[indices, time_step] = new_std.flatten()[0]

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


class Config:
    def __init__(self):
        # Seed for reproducibility
        self.seed: int = 1

        # Set data paths
        self.x_train = "data/hq/train_0.3/split_train_values.csv"
        self.dates_train = "data/hq/train_0.3/split_train_datetimes.csv"
        self.x_val = "data/hq/split_val_values.csv"
        self.dates_val = "data/hq/split_val_datetimes.csv"
        self.x_test = "data/hq/split_test_values.csv"
        self.dates_test = "data/hq/split_test_datetimes.csv"

        # Set data_loader parameters
        self.num_features: int = 2
        self.time_covariates: list = ["week_of_year"]
        self.input_seq_len: int = 52
        self.batch_size: int = 1
        self.output_col: list = [0]
        self.ts_to_use: Optional[List[int]] = [i for i in range(127)]  # Use all series

        # Set model parameters
        self.Sigma_v_bounds: tuple = (None, None)
        self.decaying_factor: float = 0.999
        self.device: str = "cpu"

        # Set training parameters
        self.num_epochs: int = 100
        self.early_stopping_criteria: str = "rmse"
        self.patience: int = 10
        self.min_delta: float = 1e-4
        self.warmup_epochs: int = 0

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
    def input_size(self) -> int:
        """Calculates the input size for the model based on other params."""
        input_size = self.num_features + self.input_seq_len - 1
        return input_size

    @property
    def nb_ts(self) -> int:
        """Calculates the number of time series based on ts_to_use."""
        if self.ts_to_use is not None:
            return len(self.ts_to_use)
        return 1  # Default value if ts_to_use is None

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


def train_local_models(config, experiment_name: Optional[str] = None):

    # Create output directory
    output_dir = f"out/{experiment_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Display and save configuration
    config.display()
    config.save(os.path.join(output_dir, "config.txt"))

    # Initalize states
    cap = 2000
    train_states = States(nb_ts=config.nb_ts, total_time_steps=cap)
    val_states = States(nb_ts=config.nb_ts, total_time_steps=cap)
    test_states = States(nb_ts=config.nb_ts, total_time_steps=cap)

    # Initialize place holder for scaling factors
    x_means = []
    x_stds = []

    for ts in np.arange(0, config.nb_ts):

        indices = [ts]

        # Prepare data loaders
        train_dtl, val_dtl, test_dtl = prepare_dtls(
            config.x_file,
            config.date_file,
            config.input_seq_len,
            config.num_features,
            config.time_covariates,
            ts,
        )
        x_means.append(train_dtl.x_mean[0])
        x_stds.append(train_dtl.x_std[0])

        # Build model
        net, output_updater = build_model(
            input_size=config.input_size,
            use_AGVI=config.use_AGVI,
            seed=config.seed,
            device=config.device,
        )

        # Create progress bar
        pbar = tqdm(range(config.num_epochs), desc=f"Epochs (TS {ts+1}/{config.nb_ts})")

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
                raise ValueError(
                    "Sigma_v_bounds must be defined when AGVI is disabled."
                )
            sigma_start = float(sigma_start)
            sigma_end = float(sigma_end)
            if config.num_epochs <= 1:
                decaying_sigma_v = [sigma_start]
            else:
                decay_factor = float(config.decaying_factor)
                exponents = decay_factor ** np.arange(
                    config.num_epochs, dtype=np.float32
                )
                if np.isclose(exponents[0], exponents[-1]) or decay_factor <= 0.0:
                    weights = np.linspace(1.0, 0.0, config.num_epochs, dtype=np.float32)
                else:
                    weights = (exponents - exponents[-1]) / (
                        exponents[0] - exponents[-1]
                    )
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
            )

            # Initialize look-back buffer and LSTM state container
            look_back_buffer = LookBackBuffer(
                input_seq_len=config.input_seq_len, nb_ts=1
            )
            lstm_state_container = LSTMStateContainer(
                num_series=1, layer_state_shapes={0: 40, 1: 40}
            )

            # get current sigma_v if not using AGVI
            if not config.use_AGVI:
                sigma_v = decaying_sigma_v[epoch]

            train_time_step = 0
            for x, y in train_batch_iter:

                # get current batch size
                B = config.batch_size

                # set LSTM states for the current batch
                lstm_state_container.set_states_on_net([0], net)

                # prepare obsevation noise matrix
                if not config.use_AGVI:
                    var_y = np.full(
                        (B * len(config.output_col),), sigma_v**2, dtype=np.float32
                    )

                # prepare look_back buffer
                if look_back_buffer.needs_initialization[0]:
                    look_back_buffer.initialize(
                        initial_mu=x[: config.input_seq_len],
                        initial_var=np.zeros_like(
                            x[: config.input_seq_len], dtype=np.float32
                        ),
                        indices=[0],
                    )

                # prepare input
                x, var_x = prepare_input(
                    x=x,
                    var_x=None,
                    look_back_mu=look_back_buffer.mu,
                    look_back_var=look_back_buffer.var,
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
                    new_mu=m_pred,
                    new_std=s_pred,
                    indices=indices,
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
                lstm_state_container.update_states_from_net([0], net)

                # Update look_back buffer
                look_back_buffer.update(
                    new_mu=m_post,
                    new_var=v_post + var_y,
                    indices=[0],
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
            look_back_buffer.needs_initialization = [True]

            val_batch_iter = val_dtl.create_data_loader(
                batch_size=config.batch_size,
                shuffle=False,
            )

            val_time_step = 0
            for x, y in val_batch_iter:

                # get current batch size
                B = config.batch_size

                # set LSTM states for the current batch
                lstm_state_container.set_states_on_net([0], net)

                # prepare obsevation noise matrix
                if not config.use_AGVI:
                    var_y = np.full(
                        (B * len(config.output_col),), sigma_v**2, dtype=np.float32
                    )

                # prepare look_back buffer
                if look_back_buffer.needs_initialization[0]:
                    look_back_buffer.initialize(
                        initial_mu=x[: config.input_seq_len],
                        initial_var=np.zeros_like(
                            x[: config.input_seq_len], dtype=np.float32
                        ),
                        indices=[0],
                    )

                # prepare input
                x, var_x = prepare_input(
                    x=x,
                    var_x=None,
                    look_back_mu=look_back_buffer.mu,
                    look_back_var=look_back_buffer.var,
                    indices=[0],
                )

                # Feedforward
                m_pred, v_pred = net(x, var_x)

                # Update LSTM states for the current batch
                lstm_state_container.update_states_from_net([0], net)

                # Specific to AGVI
                if config.use_AGVI:
                    flat_m = np.ravel(m_pred)
                    flat_v = np.ravel(v_pred)

                    m_pred = flat_m[::2]  # even indices
                    v_pred = flat_v[::2]  # even indices
                    var_y = flat_m[1::2]  # odd indices var_v

                v_pred += var_y
                s_pred = np.sqrt(v_pred)

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
                    new_mu=m_pred,
                    new_std=s_pred,
                    indices=indices,
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
            ):
                print(f"Early stopping at epoch {epoch+1}")
                net.load_state_dict(early_stopping.best_state)
                look_back_buffer = early_stopping.best_look_back_buffer
                lstm_state_container = early_stopping.best_lstm_state_container
                train_states = early_stopping.train_states
                val_states = early_stopping.val_states
                if not config.use_AGVI:
                    sigma_v = early_stopping.best_sigma_v
                break

        # Save best model
        net.save(os.path.join(output_dir, f"param/model_{ts}.pth"))

        # --- Testing ---
        # net.eval()

        # reset LSTM states
        net.reset_lstm_states()

        # reset look-back buffer
        look_back_buffer.needs_initialization = [True]

        test_batch_iter = test_dtl.create_data_loader(
            batch_size=config.batch_size,
            shuffle=False,
        )

        test_time_step = 0
        for (
            x,
            y,
        ) in test_batch_iter:

            # get current batch size
            B = config.batch_size

            # set LSTM states for the current batch
            lstm_state_container.set_states_on_net([0], net)

            # prepare obsevation noise matrix
            if not config.use_AGVI:
                var_y = np.full(
                    (B * len(config.output_col),), sigma_v**2, dtype=np.float32
                )

            # prepare look_back buffer
            if look_back_buffer.needs_initialization[0]:
                look_back_buffer.initialize(
                    initial_mu=x[: config.input_seq_len],
                    initial_var=np.zeros_like(
                        x[: config.input_seq_len], dtype=np.float32
                    ),
                    indices=[0],
                )

            # prepare input
            x, var_x = prepare_input(
                x=x,
                var_x=None,
                look_back_mu=look_back_buffer.mu,
                look_back_var=look_back_buffer.var,
                indices=[0],
            )

            # Feedforward
            m_pred, v_pred = net(x, var_x)

            # Update LSTM states for the current batch
            lstm_state_container.update_states_from_net([0], net)

            # Specific to AGVI
            if config.use_AGVI:
                flat_m = np.ravel(m_pred)
                flat_v = np.ravel(v_pred)

                m_pred = flat_m[::2]  # even indices
                v_pred = flat_v[::2]  # even indices
                var_y = flat_m[1::2]  # odd indices var_v

            v_pred += var_y
            s_pred = np.sqrt(v_pred)

            # Store predictions
            test_states.update(
                new_mu=m_pred,
                new_std=s_pred,
                indices=ts,
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
    for i in range(config.nb_ts):
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

    from pathlib import Path
    from tqdm import tqdm
    import pandas as pd

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
                    f"{i},{test_rmse_list[i]:.4f},{test_log_lik_list[i]:.4f},"
                    f"{test_mse_list[i]:.4f},{test_p50_list[i]:.4f},"
                    f"{test_p90_list[i]:.4f},{test_mase_list[i]:.4f}\n"
                )
            f.write(
                f"Overall,{overall_rmse:.4f},{overall_log_lik:.4f},"
                f"{overall_mse:.4f},{overall_p50:.4f},"
                f"{overall_p90:.4f},{overall_mase:.4f}\n"
            )


def main(Train=True, Eval=True):
    # Define experiment name
    experiment_name = "local_lstm_hq_warmup5_0.3"

    # Create configuration
    config = Config()
    config.warmup_epochs = 5

    if Train:
        # Train model
        train_local_models(config, experiment_name=experiment_name)

    if Eval:
        # Evaluate model
        eval_local_models(config, experiment_name=experiment_name)


if __name__ == "__main__":
    main()

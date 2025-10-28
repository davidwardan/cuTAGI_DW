import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

from experiments.embedding_loader import EmbeddingLayer, MappedTimeSeriesEmbeddings
from experiments.data_loader import GlobalBatchLoader
from experiments.utils import (
    prepare_dtls,
    build_model,
    prepare_input,
    plot_series,
    plot_embeddings,
    bhattacharyya_distance_matrix,
    cosine_similarity_matrix,
    plot_similarity,
    calculate_updates,
    update_aleatoric_uncertainty,
    States,
    EarlyStopping,
    LookBackBuffer,
    LSTMStateContainer,
)


from pytagi import Normalizer as normalizer
import pytagi.metric as metric

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
        self.batch_size: int = 16
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
        self.decaying_factor: float = 0.99
        self.device: str = "cpu"

        # Set training parameters
        self.num_epochs: int = 100
        self.early_stopping_criteria: str = "rmse"
        self.patience: int = 10
        self.min_delta: float = 1e-4
        self.warmup_epochs: int = 0
        self.shuffle: bool = False

        # Set evaluation parameters
        self.eval_plots: bool = False
        self.eval_metrics: bool = True
        self.seansonal_period: int = 52
        self.embed_plots: bool = False

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
    train_data, val_data, test_data = prepare_dtls(
        x_file=config.x_file,
        date_file=config.date_file,
        input_seq_len=config.input_seq_len,
        time_covariates=config.time_covariates,
        scale_method=config.scale_method,
        order_mode=config.order_mode,
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
            num_embeddings=config.nb_ts,
            embedding_size=config.embedding_size,
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
    train_states = States(nb_ts=config.nb_ts, total_time_steps=train_data.max_len)
    val_states = States(nb_ts=config.nb_ts, total_time_steps=val_data.max_len)
    test_states = States(nb_ts=config.nb_ts, total_time_steps=test_data.max_len)

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

        train_batch_iter = GlobalBatchLoader.create_data_loader(
            dataset=train_data.dataset,
            order_mode=config.order_mode,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
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

        for (x, y), ts_id, w_id in train_batch_iter:

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
                indices=ts_id,
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
                new_var=(v_post + var_y).reshape(B, -1),
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

        val_batch_iter = GlobalBatchLoader.create_data_loader(
            dataset=val_data.dataset,
            order_mode=config.order_mode,
            batch_size=config.batch_size,
            shuffle=False,
        )

        for (x, y), ts_id, w_id in val_batch_iter:

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
                new_mu=m_pred.reshape(B, -1),
                new_std=s_pred.reshape(B, -1),
                indices=indices,
                time_step=time_steps,
            )

            # Update look_back buffer
            look_back_buffer.update(
                new_mu=m_pred.reshape(B, -1)[:, -1],
                new_var=v_pred_total.reshape(B, -1)[:, -1],
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

    test_batch_iter = GlobalBatchLoader.create_data_loader(
        dataset=test_data.dataset,
        order_mode=config.order_mode,
        batch_size=config.batch_size,
        shuffle=False,
    )

    for (x, y), ts_id, w_id in test_batch_iter:

        # get current batch size and indices
        B = x.shape[0]
        indices = ts_id
        time_steps = w_id

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
        lstm_state_container.update_states_from_net(ts_id, net)

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
            new_mu=m_pred.reshape(B, -1),
            new_std=s_pred.reshape(B, -1),
            indices=indices,
            time_step=time_steps,
        )

        # Update look_back buffer
        look_back_buffer.update(
            new_mu=m_pred.reshape(B, -1)[:, -1],
            new_var=v_pred_total.reshape(B, -1)[:, -1],
            indices=indices,
        )

    # End of epoch
    net.reset_lstm_states()

    # Run over each time series and re_scale it
    for i in range(config.nb_ts):
        # get mean and std
        mean = train_data.x_mean[i][0]
        std = train_data.x_std[i][0]

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
    test_mae_list = []
    test_p50_list = []
    test_p90_list = []

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

            # Standardize test with training mean and std
            train_mean = np.nanmean(yt_train)
            train_std = np.nanstd(yt_train)

            stand_yt_test = normalizer.standardize(yt_test, train_mean, train_std)
            stand_ypred_test = normalizer.standardize(ypred_test, train_mean, train_std)
            stand_spred_test = spred_test / (train_std + 1e-6)

            if np.any(mask_test):
                y_true = stand_yt_test[mask_test]
                y_pred = stand_ypred_test[mask_test]
                s_pred = stand_spred_test[mask_test]
                max_std = 3.0 * np.std(y_true - y_pred)
                s_pred = np.where(
                    s_pred > max_std, np.nan, np.clip(s_pred, 1e-6, max_std)
                )

                # normalize data
                test_rmse = metric.rmse(y_pred, y_true)
                test_log_lik = metric.log_likelihood(y_pred, y_true, s_pred)
                test_mae = metric.mae(y_pred, y_true)

                denom = np.sum(np.abs(y_true))
                if denom == 0:
                    test_p50 = np.nan
                    test_p90 = np.nan
                else:
                    test_p50 = metric.p50(y_true, y_pred)
                    test_p90 = metric.p90(y_true, y_pred, s_pred)

            else:
                test_rmse = np.nan
                test_log_lik = np.nan
                test_mae = np.nan
                test_p50 = np.nan
                test_p90 = np.nan

            # Append to lists
            test_rmse_list.append(test_rmse)
            test_log_lik_list.append(test_log_lik)
            test_mae_list.append(test_mae)
            test_p50_list.append(test_p50)
            test_p90_list.append(test_p90)

    # Calculate overall metrics
    if config.eval_metrics:
        overall_rmse = np.nanmean(test_rmse_list)
        overall_log_lik = np.nanmean(test_log_lik_list)
        overall_mae = np.nanmean(test_mae_list)
        overall_p50 = np.nanmean(test_p50_list)
        overall_p90 = np.nanmean(test_p90_list)

        # save metrics to a table per series and overall
        with open(input_dir / "evaluation_metrics.txt", "w") as f:
            f.write("Series_ID,RMSE,LogLik,MAE,P50,P90\n")
            for i in range(config.nb_ts):
                f.write(
                    f"{config.ts_to_use[i]},{test_rmse_list[i]:.4f},{test_log_lik_list[i]:.4f},"
                    f"{test_mae_list[i]:.4f},{test_p50_list[i]:.4f},"
                    f"{test_p90_list[i]:.4f}\n"
                )
            f.write(
                f"Overall,{overall_rmse:.4f},{overall_log_lik:.4f},"
                f"{overall_mae:.4f},{overall_p50:.4f},"
                f"{overall_p90:.4f}\n"
            )

    # Check if *any* embeddings were used
    if config.embed_plots:
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

                    start_similarity = cosine_similarity_matrix(start_embeddings_mu)
                    final_similarity = cosine_similarity_matrix(final_embeddings_mu)

                    plot_similarity(
                        start_similarity,
                        embedding_dir / "embeddings_cosine_similarity_start.png",
                        "Cosine Similarity (Start)",
                    )
                    plot_similarity(
                        final_similarity,
                        embedding_dir / "embeddings_cosine_similarity_final.png",
                        "Cosine Similarity (Final)",
                    )

                    start_bhattacharyya = bhattacharyya_distance_matrix(
                        start_embeddings_mu, start_embeddings_var
                    )
                    final_bhattacharyya = bhattacharyya_distance_matrix(
                        final_embeddings_mu, final_embeddings_var
                    )

                    bhatt_vmax = float(
                        max(
                            np.nanmax(start_bhattacharyya),
                            np.nanmax(final_bhattacharyya),
                            1e-12,
                        )
                    )

                    plot_similarity(
                        start_bhattacharyya,
                        embedding_dir / "embeddings_bhattacharyya_distance_start.png",
                        "Bhattacharyya Distance (Start)",
                        vmin=0.0,
                        vmax=bhatt_vmax,
                    )
                    plot_similarity(
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
                        start_similarity = cosine_similarity_matrix(start_mu)
                        final_similarity = cosine_similarity_matrix(final_mu)

                        plot_similarity(
                            start_similarity,
                            category_plot_dir / "cosine_similarity_start.png",
                            f"Cosine Similarity (Start) - {category}",
                            labels=labels,
                        )
                        plot_similarity(
                            final_similarity,
                            category_plot_dir / "cosine_similarity_final.png",
                            f"Cosine Similarity (Final) - {category}",
                            labels=labels,
                        )

                        # Plot Bhattacharyya Distance
                        start_bhat = bhattacharyya_distance_matrix(start_mu, start_var)
                        final_bhat = bhattacharyya_distance_matrix(final_mu, final_var)
                        bhat_vmax = float(
                            max(np.nanmax(start_bhat), np.nanmax(final_bhat), 1e-12)
                        )

                        plot_similarity(
                            start_bhat,
                            category_plot_dir / "bhattacharyya_distance_start.png",
                            f"Bhattacharyya Distance (Start) - {category}",
                            vmin=0.0,
                            vmax=bhat_vmax,
                            labels=labels,
                        )
                        plot_similarity(
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
                        print(
                            f"  Warning: Failed to plot category {category}. Error: {e}"
                        )
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
                        mu_stitched_start[
                            :, current_offset : current_offset + cat_size
                        ] = loaded_start_mus[category][cat_indices]
                        var_stitched_start[
                            :, current_offset : current_offset + cat_size
                        ] = loaded_start_vars[category][cat_indices]
                        mu_stitched_final[
                            :, current_offset : current_offset + cat_size
                        ] = loaded_final_mus[category][cat_indices]
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
                    start_similarity = cosine_similarity_matrix(mu_stitched_start)
                    final_similarity = cosine_similarity_matrix(mu_stitched_final)

                    plot_similarity(
                        start_similarity,
                        embedding_dir
                        / "embeddings_cosine_similarity_start_stitched.png",
                        "Cosine Similarity (Start) - Stitched",
                    )
                    plot_similarity(
                        final_similarity,
                        embedding_dir
                        / "embeddings_cosine_similarity_final_stitched.png",
                        "Cosine Similarity (Final) - Stitched",
                    )

                    # Plot Bhattacharyya Distance
                    start_bhat = bhattacharyya_distance_matrix(
                        mu_stitched_start, var_stitched_start
                    )
                    final_bhat = bhattacharyya_distance_matrix(
                        mu_stitched_final, var_stitched_final
                    )
                    bhat_vmax = float(
                        max(np.nanmax(start_bhat), np.nanmax(final_bhat), 1e-12)
                    )

                    plot_similarity(
                        start_bhat,
                        embedding_dir
                        / "embeddings_bhattacharyya_distance_start_stitched.png",
                        "Bhattacharyya Distance (Start) - Stitched",
                        vmin=0.0,
                        vmax=bhat_vmax,
                    )
                    plot_similarity(
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
    # list_of_experiments = ["train30", "train40", "train60", "train80", "train100"]
    # list_of_seeds = [1, 42, 235, 1234, 2024]

    list_of_experiments = ["train100"]
    list_of_seeds = [1]

    for seed in list_of_seeds:
        for exp in list_of_experiments:
            print(f"Running experiment: {exp} with seed {seed}")

            # Define experiment name
            experiment_name = f"seed{seed}/{exp}/experiment01_global_model"

            # Create configuration
            config = Config()
            config.seed = seed
            config.x_train = f"data/hq/{exp}/split_train_values.csv"
            config.dates_train = f"data/hq/{exp}/split_train_datetimes.csv"
            # config.embedding_size = 5
            # config.embedding_map_dir = "data/hq/ts_embedding_map.csv"

            if Train:
                # Train model
                train_global_model(config, experiment_name=experiment_name)

            if Eval:
                # Evaluate model
                eval_global_model(config, experiment_name=experiment_name)


if __name__ == "__main__":
    main()

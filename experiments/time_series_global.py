import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, List, Optional

from experiments.wandb_helpers import (
    log_model_parameters,
    init_run,
    log_data,
    finish_run,
)
from experiments.config import Config
import torch

from experiments.embedding_loader import EmbeddingLayer, MappedTimeSeriesEmbeddings
from experiments.data_loader import GlobalBatchLoader
from experiments.utils import (
    prepare_data,
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
from experiments.tracking import EmbeddingUpdateTracker, ParameterTracker


from pytagi import Normalizer as normalizer
import pytagi.metric as metric

# Plotting defaults
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


def train_global_model(config, experiment_name: Optional[str] = None, wandb_run=None):

    # Create output directory
    output_dir = f"out/{experiment_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Display and save configuration
    config.to_yaml(os.path.join(output_dir, "config.yaml"))

    # Prepare data loaders
    train_data, val_data, test_data = prepare_data(
        x_file=config.x_file,
        date_file=config.date_file,
        input_seq_len=config.data_loader.input_seq_len,
        time_covariates=config.data_loader.time_covariates,
        scale_method=config.data_loader.scale_method,
        order_mode=config.data_loader.order_mode,
        ts_to_use=config.data_loader.ts_to_use,
    )

    # Embeddings
    embeddings = None  # Initialize as None
    embedding_dir = os.path.join(output_dir, "embeddings")

    if config.use_mapped_embeddings:
        print(
            f"Using MappedTimeSeriesEmbeddings. Total embedding size: {config.total_embedding_size}"
        )
        embeddings = MappedTimeSeriesEmbeddings(
            map_file_path=config.embeddings.mapped.embedding_map_dir,
            embedding_sizes=config.embeddings.mapped.embedding_map_sizes,
            encoding_types=config.embeddings.mapped.embedding_map_initializer,
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
            embedding_size=config.embeddings.standard.embedding_size,
            encoding_type=config.embeddings.standard.embedding_initializer,
            seed=config.seed,
            init_file=config.embeddings.standard.embedding_init_file,
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
        device=config.model.device,
        hidden_sizes=config.model.lstm_hidden_sizes,
    )

    # Enable input state updates only if embeddings are being used
    if embeddings is not None:
        net.input_state_update = True

    # Initalize states
    train_states = States(nb_ts=config.nb_ts, total_time_steps=train_data.max_len)
    val_states = States(nb_ts=config.nb_ts, total_time_steps=val_data.max_len)
    test_states = States(nb_ts=config.nb_ts, total_time_steps=test_data.max_len)

    # Create progress bar
    pbar = tqdm(range(config.training.num_epochs), desc="Epochs")

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
            raise ValueError("Sigma_v_bounds must be defined when AGVI is disabled.")
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
                weights = (exponents - exponents[-1]) / (exponents[0] - exponents[-1])
            decaying_sigma_v = (
                sigma_end + (sigma_start - sigma_end) * weights
            ).tolist()

    # Storage for 2D embedding coordinates per epoch
    # embedding_coords_history: List[np.ndarray] = []

    # Initialize embedding update tracker
    # update_tracker = EmbeddingUpdateTracker(num_series=config.nb_ts)

    # Initialize parameter tracker
    # param_tracker = ParameterTracker()
    # param_tracker.track_parameter(
    #     layer_name="LSTM.0",
    #     param_type="weight",
    #     indices=list(range(40)),
    #     label="LSTM Layer 0 Weights",
    # )
    # param_tracker.track_parameter(
    #     layer_name="LSTM.0",
    #     param_type="bias",
    #     indices=list(range(40)),
    #     label="LSTM Layer 0 Biases",
    # )

    # --- Training loop ---
    for epoch in pbar:
        # net.train()
        train_mse = []
        train_log_lik = []

        train_batch_iter = GlobalBatchLoader.create_data_loader(
            dataset=train_data.dataset,
            order_mode=config.data_loader.order_mode,
            batch_size=config.data_loader.batch_size,
            shuffle=config.training.shuffle,
            seed=1,  # fixed for all seeds and runs
        )

        # Initialize look-back buffer and LSTM state container
        look_back_buffer = LookBackBuffer(
            input_seq_len=config.data_loader.input_seq_len, nb_ts=config.nb_ts
        )
        # Create layer_state_shapes dynamically based on config
        layer_state_shapes = {
            i: size for i, size in enumerate(config.model.lstm_hidden_sizes)
        }
        lstm_state_container = LSTMStateContainer(
            num_series=config.nb_ts, layer_state_shapes=layer_state_shapes
        )

        # get current sigma_v if not using AGVI
        if not config.use_AGVI:
            sigma_v = decaying_sigma_v[epoch]

        # log current model parameters
        if wandb_run is not None:
            log_payload = log_model_parameters(
                model=net,
                epoch=epoch,
                total_epochs=config.training.num_epochs,
                logging_frequency=2,
            )

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
                    (B * len(config.data_loader.output_col),),
                    sigma_v**2,
                    dtype=np.float32,
                )

            # prepare look_back buffer
            # Filter out padded indices (-1) before checking initialization status
            valid_indices_for_init = [i for i in indices if i != -1]
            if any(
                look_back_buffer.needs_initialization[i] for i in valid_indices_for_init
            ):
                look_back_buffer.initialize(
                    initial_mu=x[:, : config.data_loader.input_seq_len],
                    initial_var=np.zeros_like(
                        x[:, : config.data_loader.input_seq_len], dtype=np.float32
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
                new_mu=m_pred.reshape(B, -1),
                new_std=s_pred_total.reshape(B, -1),
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

                # Track updates
                # update_tracker.update(indices, mu_delta_slice)

            # update aleatoric uncertainty if using AGVI
            # if config.use_AGVI:
            #     (
            #         mu_v2bar_post,
            #         _,
            #     ) = update_aleatoric_uncertainty(
            #         mu_z0=m_pred,
            #         var_z0=v_pred,
            #         mu_v2bar=flat_m[1::2],
            #         var_v2bar=flat_v[1::2],
            #         y=y.flatten(),
            #     )
            #     var_y = mu_v2bar_post  # updated aleatoric uncertainty

            # Update LSTM states for the current batch
            lstm_state_container.update_states_from_net(indices, net)

            v_post = np.clip(v_post, a_min=1e-6, a_max=2.0)

            # Update look_back buffer
            look_back_buffer.update(
                new_mu=m_post.reshape(B, -1)[:, -1],
                new_var=v_post.reshape(B, -1),
                indices=indices,
            )

        # End of epoch
        train_mse = np.mean(train_mse)
        train_log_lik = np.mean(train_log_lik)

        # Apply accumulated embedding updates at the end of the epoch
        # if embeddings is not None:
        #     embeddings.apply_accumulated_updates()

        # Step tracker
        # update_tracker.step_epoch()
        # param_tracker.step_epoch(net)

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
            order_mode=config.data_loader.order_mode,
            batch_size=config.data_loader.batch_size,
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
                    (B * len(config.data_loader.output_col),),
                    sigma_v**2,
                    dtype=np.float32,
                )

            # prepare look_back buffer
            # Filter out padded indices (-1) before checking initialization status
            valid_indices_for_init = [i for i in indices if i != -1]
            if any(
                look_back_buffer.needs_initialization[i] for i in valid_indices_for_init
            ):
                look_back_buffer.initialize(
                    initial_mu=x[:, : config.data_loader.input_seq_len],
                    initial_var=np.zeros_like(
                        x[:, : config.data_loader.input_seq_len], dtype=np.float32
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
                new_mu=m_pred.reshape(B, -1),
                new_std=s_pred_total.reshape(B, -1),
                indices=indices,
                time_step=time_steps,
            )

            # Update look_back buffer
            look_back_buffer.update(
                new_mu=m_pred.reshape(B, -1)[:, -1],
                new_var=v_pred.reshape(B, -1)[:, -1],
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

            # log current lstm states
            # lstm_states_statistics = lstm_state_container.get_statistics()
            # # iterate over layers
            # for layer_idx in lstm_states_statistics.keys():
            #     for stat_name, stat_value in lstm_states_statistics[layer_idx].items():
            #         log_payload[f"LSTM.{layer_idx}_{stat_name}"] = stat_value

            # Send all logs for this epoch
            log_data(log_payload, wandb_run=wandb_run)

        # --- Track embeddings in 2D for this epoch ---
        # track_embedding_coordinates(
        #     embeddings=embeddings,
        #     config=config,
        #     coords_history=embedding_coords_history,
        # )

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

    # # Save 2D embedding trajectory if we tracked any
    # if embeddings is not None and len(embedding_coords_history) > 0:
    #     coords_arr = np.stack(
    #         embedding_coords_history, axis=0
    #     )  # (n_epochs_run, n_entities, 2)
    #     np.savez(
    #         os.path.join(embedding_dir, "embedding_coords_history.npz"),
    #         coords=coords_arr,
    #     )

    # # Plot embedding updates
    # if embeddings is not None:
    #     update_tracker.plot(embedding_dir, filename="embedding_updates_magnitude.svg")

    # # Plot parameter evolution
    # param_tracker.plot(output_dir)

    # --- Testing ---
    # net.eval()

    # reset LSTM states
    net.reset_lstm_states()

    # reset look-back buffer
    look_back_buffer.needs_initialization = [True for _ in range(config.nb_ts)]

    test_batch_iter = GlobalBatchLoader.create_data_loader(
        dataset=test_data.dataset,
        order_mode=config.data_loader.order_mode,
        batch_size=config.data_loader.batch_size,
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
            var_y = np.full(
                (B * len(config.data_loader.output_col),), sigma_v**2, dtype=np.float32
            )

        # rolling window mechanism for traffic and electricity datasets
        if config.data_loader.use_rolling_window:
            for i, ts_index in enumerate(indices):
                if time_steps[i] % config.data_loader.rolling_window_size == 0:
                    look_back_buffer.needs_initialization[ts_index] = True

        # prepare look_back buffer
        # Filter out padded indices (-1) before checking initialization status
        valid_indices_for_init = [i for i in indices if i != -1]
        if any(
            look_back_buffer.needs_initialization[i] for i in valid_indices_for_init
        ):
            look_back_buffer.initialize(
                initial_mu=x[:, : config.data_loader.input_seq_len],
                initial_var=np.zeros_like(
                    x[:, : config.data_loader.input_seq_len], dtype=np.float32
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

        s_pred_total = np.sqrt(v_pred + var_y)

        # Store predictions
        test_states.update(
            new_mu=m_pred.reshape(B, -1),
            new_std=s_pred_total.reshape(B, -1),
            indices=indices,
            time_step=time_steps,
        )

        # Update look_back buffer
        look_back_buffer.update(
            new_mu=m_pred.reshape(B, -1)[:, -1],
            new_var=v_pred.reshape(B, -1)[:, -1],
            indices=indices,
        )

    # End of epoch
    net.reset_lstm_states()

    # Run over each time series and re_scale it
    if config.data_loader.scale_method == "standard":
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


def eval_global_model(
    config,
    experiment_name: Optional[str] = None,
    wandb_run: Optional[Any] = None,
):
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
        usecols=config.data_loader.ts_to_use,
    ).values
    true_val = pd.read_csv(
        config.x_file[1],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.data_loader.ts_to_use,
    ).values
    true_test = pd.read_csv(
        config.x_file[2],
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=config.data_loader.ts_to_use,
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

    # create wandb metrics to log
    if wandb_run is not None:
        wandb_run.define_metric("rmse", summary="last")
        wandb_run.define_metric("mae", summary="last")
        wandb_run.define_metric("log_lik", summary="last")
        wandb_run.define_metric("p50", summary="last")
        wandb_run.define_metric("p90", summary="last")

    # Iterate over each time series and calculate metrics
    for i in tqdm(range(config.nb_ts), desc="Evaluating series"):

        # Get true values
        yt_train, yt_val, yt_test = (
            _trim_trailing_nans(true_train[config.data_loader.input_seq_len :, i]),
            _trim_trailing_nans(true_val[config.data_loader.input_seq_len :, i]),
            _trim_trailing_nans(true_test[config.data_loader.input_seq_len :, i]),
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

        # Forecast Plotting
        if config.evaluation.eval_plots:
            plot_series(
                ts_idx=i,
                y_true=yt_full,
                y_pred=ypred_full,
                s_pred=spred_full,
                out_dir=input_dir / "figures",
                val_test_indices=val_test_indices,
                std_factor=1,
            )

        # Metrics
        if config.evaluation.eval_metrics:

            # Standardize test with training mean and std
            if config.data_loader.scale_method == "standard":
                train_mean = np.nanmean(yt_train)
                train_std = np.nanstd(yt_train)
            else:
                # manual standardization
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

    # Calculate overall metrics
    if config.evaluation.eval_metrics:
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
                    f"{config.data_loader.ts_to_use[i]},{test_rmse_list[i]:.4f},{test_log_lik_list[i]:.4f},"
                    f"{test_mae_list[i]:.4f},{test_p50_list[i]:.4f},"
                    f"{test_p90_list[i]:.4f}\n"
                )
            f.write(
                f"Overall,{overall_rmse:.4f},{overall_log_lik:.4f},"
                f"{overall_mae:.4f},{overall_p50:.4f},"
                f"{overall_p90:.4f}\n"
            )

        overall_metrics_payload = {
            "rmse": overall_rmse,
            "mae": overall_mae,
            "log_lik": overall_log_lik,
            "p50": overall_p50,
            "p90": overall_p90,
        }

        if wandb_run is not None:
            log_data(overall_metrics_payload, wandb_run=wandb_run)

    # Embedding plots
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
                labels = [str(ts_id) for ts_id in config.data_loader.ts_to_use]

                start_similarity = cosine_similarity_matrix(start_embeddings_mu)
                final_similarity = cosine_similarity_matrix(final_embeddings_mu)

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

                if wandb_run is not None:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=final_similarity,
                            x=labels,
                            y=labels,
                            colorscale="RdBu",
                            zmin=-1.0,
                            zmax=1.0,
                        )
                    )
                    fig.update_layout(title=f"Cosine similarity", yaxis_scaleanchor="x")
                    log_data({f"embeddings/similarity": fig}, wandb_run=wandb_run)

                if config.evaluation.embed_plots:
                    plot_embeddings(
                        start_embeddings_mu,
                        config.nb_ts,
                        input_dir,
                        "embeddings/embeddings_mu_pca_start.svg",
                        labels=labels,
                    )
                    plot_embeddings(
                        final_embeddings_mu,
                        config.nb_ts,
                        input_dir,
                        "embeddings/embeddings_mu_pca_final.svg",
                        labels=labels,
                    )
                    if config.total_embedding_size <= 1:
                        print(
                            "  Skipping embedding PCA plots since embedding size <= 1."
                        )
                    else:
                        plot_similarity(
                            start_similarity,
                            embedding_dir / "embeddings_cosine_similarity_start.svg",
                            "Cosine Similarity (Start)",
                        )
                        plot_similarity(
                            final_similarity,
                            embedding_dir / "embeddings_cosine_similarity_final.svg",
                            "Cosine Similarity (Final)",
                        )
                        plot_similarity(
                            start_bhattacharyya,
                            embedding_dir
                            / "embeddings_bhattacharyya_distance_start.svg",
                            "Bhattacharyya Distance (Start)",
                            vmin=0.0,
                            vmax=bhatt_vmax,
                        )
                        plot_similarity(
                            final_bhattacharyya,
                            embedding_dir
                            / "embeddings_bhattacharyya_distance_final.svg",
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

            # Plot per-category embeddings
            categories = sorted(
                list(config.embeddings.mapped.embedding_map_sizes.keys())
            )

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
                    labels = config.embeddings.mapped.embedding_map_labels[category]

                    # Create sub-directory
                    category_plot_dir = embedding_dir / category
                    category_plot_dir.mkdir(parents=True, exist_ok=True)

                    # Plot Cosine Similarity
                    start_similarity = cosine_similarity_matrix(start_mu)
                    final_similarity = cosine_similarity_matrix(final_mu)

                    # Plot Bhattacharyya Distance
                    start_bhat = bhattacharyya_distance_matrix(start_mu, start_var)
                    final_bhat = bhattacharyya_distance_matrix(final_mu, final_var)
                    bhat_vmax = float(
                        max(np.nanmax(start_bhat), np.nanmax(final_bhat), 1e-12)
                    )

                    if wandb_run is not None:
                        fig = go.Figure(
                            data=go.Heatmap(
                                z=final_similarity,
                                x=labels,
                                y=labels,
                                colorscale="RdBu",
                                zmin=-1.0,
                                zmax=1.0,
                            )
                        )
                        fig.update_layout(
                            title=f"Cosine similarity {category}", yaxis_scaleanchor="x"
                        )
                        log_data(
                            {f"embeddings/similarity_{category}": fig},
                            wandb_run=wandb_run,
                        )

                    if config.evaluation.embed_plots:
                        # Plot PCA
                        plot_embeddings(
                            start_mu,
                            n_entities,
                            input_dir,  # base dir
                            f"embeddings/{category}/pca_start.svg",
                            labels=labels,
                        )
                        plot_embeddings(
                            final_mu,
                            n_entities,
                            input_dir,  # base dir
                            f"embeddings/{category}/pca_final.svg",
                            labels=labels,
                        )

                        plot_similarity(
                            start_similarity,
                            category_plot_dir / "cosine_similarity_start.svg",
                            f"Cosine Similarity (Start) - {category}",
                            labels=labels,
                        )
                        plot_similarity(
                            final_similarity,
                            category_plot_dir / "cosine_similarity_final.svg",
                            f"Cosine Similarity (Final) - {category}",
                            labels=labels,
                        )
                        plot_similarity(
                            start_bhat,
                            category_plot_dir / "bhattacharyya_distance_start.svg",
                            f"Bhattacharyya Distance (Start) - {category}",
                            vmin=0.0,
                            vmax=bhat_vmax,
                            labels=labels,
                        )
                        plot_similarity(
                            final_bhat,
                            category_plot_dir / "bhattacharyya_distance_final.svg",
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

            # Stitch Embeddings
            print("Stitching full time series embeddings for plotting...")
            try:
                # Load and filter map to the series we used, in the correct order
                if not os.path.exists(config.embeddings.mapped.embedding_map_dir):
                    raise FileNotFoundError(
                        f"Map file not found: {config.embeddings.mapped.embedding_map_dir}"
                    )

                map_df = pd.read_csv(
                    config.embeddings.mapped.embedding_map_dir
                ).set_index("ts_id")

                if config.data_loader.ts_to_use is None:
                    raise ValueError(
                        "config.ts_to_use is None, cannot stitch embeddings."
                    )

                # Re-order map based on ts_to_use
                map_df_ordered = map_df.loc[config.data_loader.ts_to_use]

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

                # Plot Stitched Embeddings
                print("Plotting stitched (full) time series embeddings...")
                labels = [str(ts_id) for ts_id in config.data_loader.ts_to_use]

                # Plot Cosine Similarity
                start_similarity = cosine_similarity_matrix(mu_stitched_start)
                final_similarity = cosine_similarity_matrix(mu_stitched_final)

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

                if wandb_run is not None:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=final_similarity,
                            x=labels,
                            y=labels,
                            colorscale="RdBu",
                            zmin=-1.0,
                            zmax=1.0,
                        )
                    )
                    fig.update_layout(title=f"Cosine similarity", yaxis_scaleanchor="x")
                    log_data({f"embeddings/similarity": fig}, wandb_run=wandb_run)

                if config.evaluation.embed_plots:
                    # Plot PCA
                    plot_embeddings(
                        mu_stitched_start,
                        config.nb_ts,
                        input_dir,
                        "embeddings/embeddings_mu_pca_start_stitched.svg",
                        labels=labels,
                    )
                    plot_embeddings(
                        mu_stitched_final,
                        config.nb_ts,
                        input_dir,
                        "embeddings/embeddings_mu_pca_final_stitched.svg",
                        labels=labels,
                    )

                    plot_similarity(
                        start_similarity,
                        embedding_dir
                        / "embeddings_cosine_similarity_start_stitched.svg",
                        "Cosine Similarity (Start) - Stitched",
                    )
                    plot_similarity(
                        final_similarity,
                        embedding_dir
                        / "embeddings_cosine_similarity_final_stitched.svg",
                        "Cosine Similarity (Final) - Stitched",
                    )

                    plot_similarity(
                        start_bhat,
                        embedding_dir
                        / "embeddings_bhattacharyya_distance_start_stitched.svg",
                        "Bhattacharyya Distance (Start) - Stitched",
                        vmin=0.0,
                        vmax=bhat_vmax,
                    )
                    plot_similarity(
                        final_bhat,
                        embedding_dir
                        / "embeddings_bhattacharyya_distance_final_stitched.svg",
                        "Bhattacharyya Distance (Final) - Stitched",
                        vmin=0.0,
                        vmax=bhat_vmax,
                    )

            except Exception as e:
                print(
                    f"  Warning: Failed to stitch and plot full embeddings. Error: {e}"
                )


def main(Train=True, Eval=True, log_wandb=True):

    list_of_seeds = [1]
    list_of_experiments = ["train100"]

    # Iterate over experiments and seeds
    for seed in list_of_seeds:
        for exp in list_of_experiments:
            print(f"Running experiment: {exp} with seed {seed}")

            # Model category
            model_category = "global"
            embed_category = "simple-embeddings"

            # Define experiment name
            experiment_name = f"seed{seed}/{exp}/{model_category}_{embed_category}_attn"

            # Load configuration
            config = Config.from_yaml(
                f"experiments/configurations/{model_category}_{embed_category}_HQ127.yaml"
            )

            config.seed = seed
            config.model.device = "cuda" if torch.cuda.is_available() else "cpu"
            config.evaluation.eval_plots = True
            config.evaluation.embed_plots = True

            config.embeddings.standard.embedding_init_file = (
                "out/autoencoder_embeddings_attn.npy"
            )

            # Convert config object to a dictionary for W&B
            config_dict = config.wandb_dict()
            config_dict["model_type"] = f"{model_category}_{embed_category}"

            # Display config
            config.display()

            if log_wandb:
                # Initialize W&B run
                run_id = f"{model_category}_{embed_category}_{exp}_seed{seed}".replace(
                    " ", ""
                )
                run = init_run(
                    project="tracking_weights_lstm",
                    name=run_id,
                    group=f"{model_category}_Seed{embed_category}",
                    tags=["normal"],
                    config=config_dict,
                    reinit=True,
                    save_code=True,
                )
            else:
                run = None

            if Train:
                # Train model
                train_global_model(
                    config, experiment_name=experiment_name, wandb_run=run
                )

            if Eval:
                # Evaluate model
                eval_global_model(
                    config, experiment_name=experiment_name, wandb_run=run
                )

            # Finish the W&B run
            if log_wandb:
                finish_run(run)


if __name__ == "__main__":
    main(True, True, False)

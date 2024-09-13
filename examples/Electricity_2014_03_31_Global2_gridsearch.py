import os
import sys
import fire
import numpy as np
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential
from pytagi import Normalizer as normalizer

from examples.data_loader import GlobalTimeSeriesDataloader

import matplotlib.pyplot as plt

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

'''
FIX SEED BEFORE YOU RUN THIS SCRIPT
'''


def main(num_epochs: int = 100, sigma_v: float = 2):
    """
    Run training for a time-series forecasting global model.
    Training is done on shuffling batches from all series.
    """

    # Dataset
    nb_ts = 370  # for electricity 370 and 963 for traffic
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 3
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    factors = [1.0] * nb_ts
    mean_train = [0.0] * nb_ts
    std_train = [1.0] * nb_ts
    covar_means = [0.0] * nb_ts
    covar_stds = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/electricity/electricity_2014_03_31_train.csv",
            date_time_file="data/electricity/electricity_2014_03_31_train_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            time_covariates=['hour_of_day', 'day_of_week'],
            global_scale='standard',
            # idx_as_feature=True,
            scale_covariates=True,
        )

        # Store scaling factors----------------------#
        # factors[ts] = train_dtl_.scale_i
        mean_train[ts] = train_dtl_.x_mean
        std_train[ts] = train_dtl_.x_std
        # -------------------------------------------#

        # store covariate means and stds
        covar_means[ts] = train_dtl_.covariate_means
        covar_stds[ts] = train_dtl_.covariate_stds

        val_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/electricity/electricity_2014_03_31_val.csv",
            date_time_file="data/electricity/electricity_2014_03_31_val_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            x_mean=mean_train[ts],
            x_std=std_train[ts],
            time_covariates=['hour_of_day', 'day_of_week'],
            global_scale='standard',
            # scale_i=factors[ts],
            scale_covariates=True,
            covariate_means=covar_means[ts],
            covariate_stds=covar_stds[ts],
            # idx_as_feature=True,
        )

        if ts == 0:
            train_dtl = train_dtl_
            val_dtl = val_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)
            val_dtl = concat_ts_sample(val_dtl, val_dtl_)


    # -------------------------------------------------------------------------#
    # Grid search
    # define the grid search parameters
    lstm_nodes_gs = [40, 60, 80]
    sigma_v_gs = [0.5, 0.1, 0.05, 0.01]
    batch_size_gs = [32, 64]
    lstm_layers_gs = [2, 3, 4]

    # build grid with all the combinations
    grid = [(l, s, b, lay) for l in lstm_nodes_gs for s in sigma_v_gs for b in batch_size_gs for lay in lstm_layers_gs]

    # create dictionary to store results
    d = {'lstm_nodes': [], 'sigma_v': [], 'batch_size': [], 'lstm_layers': [], 'mse_val': [], 'log_lik_val': []}


    # iterate over the grid
    for l, s, b, lay in grid:

        # Parameters
        lstm_nodes = l
        batch_size = b

        # Network
        if lay == 2:
            net = Sequential(
            LSTM(num_features, lstm_nodes, input_seq_len),
            LSTM(lstm_nodes, lstm_nodes, input_seq_len),
            LSTM(lstm_nodes, lstm_nodes, input_seq_len),
            Linear(lstm_nodes * input_seq_len, 1),
        )
        elif lay == 3:
            net = Sequential(
                LSTM(num_features, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                Linear(lstm_nodes * input_seq_len, 1),
            )
        elif lay == 4:
            net = Sequential(
                LSTM(num_features, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                LSTM(lstm_nodes, lstm_nodes, input_seq_len),
                Linear(lstm_nodes * input_seq_len, 1),
            )


        net.to_device("cuda")
        # net.set_threads(8)
        out_updater = OutputUpdater(net.device)

        # Training
        mses = []
        mses_val = []  # to save mse_val for plotting
        ll_val = []  # to save log likelihood for plotting

        # options for early stopping
        log_lik_optim = -1E100
        mse_optim = 1E100
        epoch_optim = 1
        early_stopping_criteria = 'log_lik'  # 'log_lik' or 'mse'
        patience = 10
        net_optim = []  # to save optimal net at the optimal epoch

        sigma_v = 2.0

        print(f"Training with: lstm_nodes={l}, sigma_v_lo={s}, batch_size={b}, lstm_layers={lay}")

        pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in pbar:
            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=True) #, weighted_sampling=True, weights=weights)

            # Decaying observation's variance
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=s, decaying_factor=0.99, curr_iter=epoch
            )
            var_y = np.full((batch_size * len(output_col),), sigma_v ** 2, dtype=np.float32)

            for x, y in batch_iter:
                # Feed forward
                m_pred, _ = net(x)

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

                pred = m_pred
                obs = y

                # Training metric
                mse = metric.mse(pred, obs)
                mses.append(mse)

            # -------------------------------------------------------------------------#
            # Validation
            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds = []
            var_preds = []
            y_val = []
            x_val = []

            for x, y in val_batch_iter:
                # Prediction
                m_pred, v_pred = net(x)

                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v ** 2)
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
            pbar.set_postfix(mse=f"{np.mean(mses):.4f}", mse_val=f"{mse_val:.4f}", log_lik_val=f"{log_lik_val:.4f}", sigma_v=f"{sigma_v:.4f}")

            # early-stopping
            if early_stopping_criteria == 'mse':
                if mse_val < mse_optim:
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net
            elif early_stopping_criteria == 'log_lik':
                if log_lik_val > log_lik_optim:
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net
            if epoch - epoch_optim > patience:
                # save results into dictionary
                d['lstm_nodes'].append(l)
                d['sigma_v'].append(s)
                d['batch_size'].append(b)
                d['lstm_layers'].append(lay)
                d['mse_val'].append(mse_optim)
                d['log_lik_val'].append(log_lik_optim)
                break

    # save results to csv
    import pandas as pd
    df = pd.DataFrame(d)
    df.to_csv('grid_search_results_std.csv', index=False)


def concat_ts_sample(data, data_add):
    """Concatenate two time series samples"""
    x_combined = np.concatenate((data.dataset["value"][0], data_add.dataset["value"][0]), axis=0)
    y_combined = np.concatenate((data.dataset["value"][1], data_add.dataset["value"][1]), axis=0)
    time_combined = np.concatenate((data.dataset["date_time"], data_add.dataset["date_time"]))
    # weights_combined = np.concatenate((data.dataset["weights"], data_add.dataset["weights"]))
    data.dataset["value"] = (x_combined, y_combined)
    data.dataset["date_time"] = time_combined
    # data.dataset["weights"] = weights_combined
    return data


def random_weighted_sampling(data: np.ndarray, weights: np.ndarray, num_samples: int) -> np.ndarray:
    """Random-weighted sampling"""
    return np.random.choice(data, num_samples, p=weights)


if __name__ == "__main__":
    fire.Fire(main)

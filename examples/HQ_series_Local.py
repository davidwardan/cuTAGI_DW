from typing import Optional

import os
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler, manual_seed
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential

from examples.data_loader import TimeSeriesDataloader


def main(
    num_epochs: int = 200,
    batch_size: int = 64,
    sigma_v: float = 2,
    lstm_nodes: int = 40,
    seed: int = 0,
):
    """Run training for time-series forecasting model"""
    # Dataset
    output_col = [0]
    num_features = 2
    input_seq_len = 52
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 52  # for rolling window predictions in the test set
    early_stopping_criteria = "log_lik"  # 'log_lik' or 'mse'
    patience = 10

    manual_seed(seed)

    # Network
    net = Sequential(
        LSTM(num_features, lstm_nodes, input_seq_len),
        LSTM(lstm_nodes, lstm_nodes, input_seq_len),
        LSTM(lstm_nodes, lstm_nodes, input_seq_len),
        Linear(lstm_nodes * input_seq_len, 1),
    )
    net.to_device("cuda")
    # net.set_threads(12)
    out_updater = OutputUpdater(net.device)

    # Loop over each time series in the benchmark
    nb_ts = 31
    ts_idx = np.arange(0, nb_ts)  # time series no.
    ytestPd = np.full((104, nb_ts), np.nan)
    SytestPd = np.full((104, nb_ts), np.nan)
    ytestTr = np.full((104, nb_ts), np.nan)

    for ts in ts_idx:
        # options for early stopping
        log_lik_optim = -1e100
        mse_optim = 1e100
        epoch_optim = 1
        net_optim = []  # to save optimal net at the optimal epoch

        train_dtl = TimeSeriesDataloader(
            x_file="data/hq_data/weekly/hq_train_weekly_values.csv",
            date_time_file="data/hq_data/weekly/hq_train_weekly_dates.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            ts_idx=ts,
            time_covariates=["week_of_year"],
        )
        val_dtl = TimeSeriesDataloader(
            x_file="data/hq_data/weekly/hq_val_weekly_values.csv",
            date_time_file="data/hq_data/weekly/hq_val_weekly_dates.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            ts_idx=ts,
            time_covariates=["week_of_year"],
        )
        test_dtl = TimeSeriesDataloader(
            x_file="data/hq_data/weekly/hq_test_weekly_values.csv",
            date_time_file="data/hq_data/weekly/hq_test_weekly_dates.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            x_mean=train_dtl.x_mean,
            x_std=train_dtl.x_std,
            ts_idx=ts,
            time_covariates=["week_of_year"],
        )

        # -------------------------------------------------------------------------#
        # Training
        mses = []
        pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in pbar:
            batch_iter = train_dtl.create_data_loader(batch_size, False)

            # Decaying observation's variance
            sigma_v = exponential_scheduler(
                curr_v=sigma_v, min_v=0.1, decaying_factor=0.99, curr_iter=epoch
            )
            var_y = np.full(
                (batch_size * len(output_col),), sigma_v**2, dtype=np.float32
            )
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

                # Training metric
                pred = normalizer.unstandardize(
                    m_pred, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
                )
                obs = normalizer.unstandardize(
                    y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
                )
                mse = metric.mse(pred, obs)
                mses.append(mse)

            # Validation
            val_batch_iter = val_dtl.create_data_loader(batch_size, shuffle=False)

            mu_preds = []
            var_preds = []
            y_val = []
            x_val = []

            for x, y in val_batch_iter:
                # Predicion
                m_pred, v_pred = net(x)

                mu_preds.extend(m_pred)
                var_preds.extend(v_pred + sigma_v**2)
                x_val.extend(x)
                y_val.extend(y)

            mu_preds = np.array(mu_preds)
            std_preds = np.array(var_preds) ** 0.5
            y_val = np.array(y_val)
            x_val = np.array(x_val)

            mu_preds = normalizer.unstandardize(
                mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )
            std_preds = normalizer.unstandardize_std(
                std_preds, train_dtl.x_std[output_col]
            )

            y_val = normalizer.unstandardize(
                y_val, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )

            # Compute log-likelihood for validation set
            mse_val = metric.mse(mu_preds, y_val)
            log_lik_val = metric.log_likelihood(
                prediction=mu_preds, observation=y_val, std=std_preds
            )

            # Progress bar
            pbar.set_description(
                f"Ts #{ts+1}/{nb_ts} | Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses)/len(mses):>7.3f}",
                refresh=True,
            )

            # early-stopping
            if early_stopping_criteria == "mse":
                if float(mse_val) < float(mse_optim):
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net.get_state_dict()
            elif early_stopping_criteria == "log_lik":
                if float(log_lik_val) > float(log_lik_optim):
                    mse_optim = mse_val
                    log_lik_optim = log_lik_val
                    epoch_optim = epoch
                    net_optim = net.get_state_dict()
            if int(epoch) - int(epoch_optim) > patience:
                break

        # -------------------------------------------------------------------------#
        # Testing
        net.load_state_dict(net_optim)
        # test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % (rolling_window)
            if RW_idx > 0:
                x[-RW_idx * num_features :: num_features] = mu_preds[-RW_idx:]
            #

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

        mu_preds = normalizer.unstandardize(
            mu_preds, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
        )
        std_preds = normalizer.unstandardize_std(std_preds, train_dtl.x_std[output_col])

        y_test = normalizer.unstandardize(
            y_test, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
        )

        # save test predicitons for each time series
        ytestPd[:, ts] = mu_preds.flatten()
        SytestPd[:, ts] = std_preds.flatten() ** 2
        ytestTr[:, ts] = y_test.flatten()

    out_dir = (
        "out/hq_"
        + str(epoch)
        + "_"
        + str(batch_size)
        + "_"
        + str(lstm_nodes)
        + "_method2"
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.savetxt(out_dir + "/ytestPd.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/SytestPd.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/ytestTr.csv", ytestTr, delimiter=",")

    # calculate metrics
    p50_tagi = metric.computeND(ytestTr, ytestPd)
    p90_tagi = metric.compute90QL(ytestTr, ytestPd, SytestPd)
    RMSE_tagi = metric.computeRMSE(ytestTr, ytestPd)

    # save metrics into a text file
    with open(out_dir + "/metrics.txt", "w") as f:
        f.write(f"ND/p50:    {p50_tagi}\n")
        f.write(f"p90:    {p90_tagi}\n")
        f.write(f"RMSE:    {RMSE_tagi}\n")
        f.write(f"Epoch:    {epoch_optim}\n")
        f.write(f"Batch size:    {batch_size}\n")
        f.write(f"Sigma_v:    {sigma_v}\n")
        f.write(f"LSTM nodes:    {lstm_nodes}\n")


if __name__ == "__main__":
    fire.Fire(main)

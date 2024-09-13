import os
import sys
import fire
import numpy as np
from tqdm import tqdm
import shutil

import pytagi.metric as metric
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential

from examples.data_loader import GlobalTimeSeriesDataloader

import matplotlib.pyplot as plt

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)


def main(num_epochs: int = 100, batch_size: int = 64, sigma_v: float = 0.5, lstm_nodes: int = 40):
    """
    Run training for a time-series forecasting global model.
    Training is done on shuffling batches from all series.
    """

    # Dataset
    nb_ts = 1 # for electricity 370 and 963 for traffic
    ts_idx = np.arange(0, nb_ts)
    output_col = [0]
    num_features = 3
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1
    rolling_window = 24 # for rolling window predictions in the test set

    pbar = tqdm(ts_idx, desc="Loading Data Progress")

    covar_means = [0.0] * nb_ts
    covar_stds = [1.0] * nb_ts

    for ts in pbar:
        train_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/traffic/traffic_2008_01_14_train.csv",
            date_time_file="data/traffic/traffic_2008_01_14_train_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            time_covariates=['hour_of_day', 'day_of_week'],
            scale_covariates=True,
            ts_idx=ts,
            # idx_as_feature=True,
        )

        # store covariate means and stds
        covar_means[ts] = train_dtl_.covariate_means
        covar_stds[ts] = train_dtl_.covariate_stds

        val_dtl_ = GlobalTimeSeriesDataloader(
            x_file="data/traffic/traffic_2008_01_14_val.csv",
            date_time_file="data/traffic/traffic_2008_01_14_val_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            time_covariates=['hour_of_day', 'day_of_week'],
            scale_covariates=True,
            covariate_means=covar_means[ts],
            covariate_stds=covar_stds[ts],
            ts_idx=ts,
            # idx_as_feature=True,
        )

        if ts == 0:
            train_dtl = train_dtl_
            val_dtl = val_dtl_
        else:
            train_dtl = concat_ts_sample(train_dtl, train_dtl_)
            val_dtl = concat_ts_sample(val_dtl, val_dtl_)

    # Network
    net = Sequential(
        LSTM(num_features, lstm_nodes, input_seq_len),
        LSTM(lstm_nodes, lstm_nodes, input_seq_len),
        LSTM(lstm_nodes, lstm_nodes, input_seq_len),
        Linear(lstm_nodes * input_seq_len, 1),
    )
    # net.to_device("cuda")
    # net.set_threads(8)
    out_updater = OutputUpdater(net.device)

    # Create output directory
    out_dir = ("david/output/traffic_" + str(num_epochs) + "_" + str(batch_size) + "_" + str(sigma_v)
               + "_" + str(lstm_nodes) + "_method2")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    mses_val = []  # to save mse_val for plotting
    ll_val = []  # to save log likelihood for plotting

    # options for early stopping
    log_lik_optim = -1E100
    mse_optim = 1E100
    epoch_optim = 1
    early_stopping_criteria = 'mse'  # 'log_lik' or 'mse'
    patience = 10
    net_optim = []  # to save optimal net at the optimal epoch

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.01, decaying_factor=0.99, curr_iter=epoch
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

            # Training metric
            mse = metric.mse(m_pred, y)
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
        pbar.set_postfix(mse=f"{np.mean(mses):.4f}", mse_val=f"{mse_val:.4f}", log_lik_val=f"{log_lik_val:.4f}",
                         sigma_v=f"{sigma_v:.4f}")

        # create a directory to save the model
        if not os.path.exists("best_model1/"):
            os.makedirs("best_model1/")

        # early-stopping
        if early_stopping_criteria == 'mse':
            if mse_val < mse_optim:
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net.save_csv("best_model1/")
                # net_optim = net
        elif early_stopping_criteria == 'log_lik':
            if log_lik_val > log_lik_optim:
                mse_optim = mse_val
                log_lik_optim = log_lik_val
                epoch_optim = epoch
                net.save_csv("best_model1/")
                # net_optim = net
        if epoch - epoch_optim > patience:
            break

        print(type(net_optim))
    # -------------------------------------------------------------------------#
        # fig, ax1 = plt.subplots()

        # # Set title for the plot
        # ax1.set_title('Validation Metrics', fontsize=16)

        # # Plot MSE on primary y-axis
        # ax1.set_xlabel('Epoch')
        # ax1.set_ylabel('MSE', color='steelblue')
        # ax1.plot(mses_val, color='steelblue', label='MSE')
        # ax1.tick_params(axis='y', labelcolor='steelblue')

        # # Plot Log Likelihood on secondary y-axis
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Log Likelihood', color='indianred')
        # ax2.plot(ll_val, color='indianred', label='Log Likelihood')
        # ax2.tick_params(axis='y', labelcolor='indianred')

        # # Adjust layout to make room for the title and legends
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # # Save the figure
        # fig.savefig(out_dir + "/validation_plot.png", dpi=300)
    # -------------------------------------------------------------------------#
    # save validation metrics into csv
    df = np.array([mses_val, ll_val]).T
    np.savetxt(out_dir + "/validation_metrics.csv", df, delimiter=",")
    # -------------------------------------------------------------------------#
    # load the optimal net
    net.load_csv("best_model1/")  # load optimal net
    shutil.rmtree("best_model1/") # remove the directory

    # save the model
    net.save_csv(out_dir + "/param/traffic_2008_01_14_net_pyTAGI.csv")

    # Testing
    pbar = tqdm(ts_idx, desc="Testing Progress")

    ytestPd = np.full((168, nb_ts), np.nan)
    SytestPd = np.full((168, nb_ts), np.nan)
    ytestTr = np.full((168, nb_ts), np.nan)
    for ts in pbar:

        test_dtl = GlobalTimeSeriesDataloader(
            x_file="data/traffic/traffic_2008_01_14_test.csv",
            date_time_file="data/traffic/traffic_2008_01_14_test_datetime.csv",
            output_col=output_col,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            num_features=num_features,
            stride=seq_stride,
            time_covariates=['hour_of_day', 'day_of_week'],
            scale_covariates=True,
            covariate_means=covar_means[ts],
            covariate_stds=covar_stds[ts],
            ts_idx=ts,
            # idx_as_feature=True,
        )

        # test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
        test_batch_iter = test_dtl.create_data_loader(1, shuffle=False)

        mu_preds = []
        var_preds = []
        y_test = []
        x_test = []

        # net = net_optim

        # Rolling window predictions
        for RW_idx_, (x, y) in enumerate(test_batch_iter):
            # Rolling window predictions
            RW_idx = RW_idx_ % rolling_window
            if RW_idx > 0:
                x[-RW_idx * num_features::num_features] = mu_preds[-RW_idx:]

            # Prediction
            m_pred, v_pred = net(x)

            mu_preds.extend(m_pred)
            var_preds.extend(v_pred + sigma_v ** 2)
            x_test.extend(x)
            y_test.extend(y)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # save test predictions for each time series
        ytestPd[:, ts] = mu_preds.flatten()
        SytestPd[:, ts] = std_preds.flatten() ** 2
        ytestTr[:, ts] = y_test.flatten()

    np.savetxt(out_dir + "/traffic_2008_01_14_ytestPd_pyTAGI.csv", ytestPd, delimiter=",")
    np.savetxt(out_dir + "/traffic_2008_01_14_SytestPd_pyTAGI.csv", SytestPd, delimiter=",")
    np.savetxt(out_dir + "/traffic_2008_01_14_ytestTr_pyTAGI.csv", ytestTr, delimiter=",")

    # -------------------------------------------------------------------------#
    # calculate metrics
    p50_tagi = metric.computeND(ytestTr, ytestPd)
    p90_tagi = metric.compute90QL(ytestTr, ytestPd, SytestPd)
    RMSE_tagi = metric.computeRMSE(ytestTr, ytestPd)
    # MASE_tagi = metric.computeMASE(ytestTr, ytestPd, ytrain, seasonality) # TODO: check if ytrain is correct

    # save metrics into a text file
    with open(out_dir + "/metrics.txt", "w") as f:
        f.write(f'ND/p50:    {p50_tagi}\n')
        f.write(f'p90:    {p90_tagi}\n')
        f.write(f'RMSE:    {RMSE_tagi}\n')
        f.write(f'Epoch:    {epoch_optim}\n')
        f.write(f'Batch size:    {batch_size}\n')
        f.write(f'Sigma_v:    {sigma_v}\n')
        f.write(f'LSTM nodes:    {lstm_nodes}\n')
        # f.write(f'MASE:    {MASE_tagi}\n')

    # rename the directory
    out_dir_ = "david/output/traffic_" + str(epoch_optim) + "_" + str(batch_size) + "_" + str(
        round(sigma_v, 3)) + "_" + str(lstm_nodes) + "_method2_seed3"
    os.rename(out_dir, out_dir_)


def concat_ts_sample(data, data_add):
    x_combined = np.concatenate((data.dataset["value"][0], data_add.dataset["value"][0]), axis=0)
    y_combined = np.concatenate((data.dataset["value"][1], data_add.dataset["value"][1]), axis=0)
    time_combined = np.concatenate((data.dataset["date_time"], data_add.dataset["date_time"]))
    data.dataset["value"] = (x_combined, y_combined)
    data.dataset["date_time"] = time_combined
    return data


if __name__ == "__main__":
    fire.Fire(main)
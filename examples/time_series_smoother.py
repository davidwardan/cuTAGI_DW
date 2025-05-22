from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from examples.data_loader import TimeSeriesDataloader
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler, manual_seed
from pytagi.nn import SLSTM, OutputUpdater, Sequential, SLinear


def main(
    num_epochs: int = 50, batch_size: int = 1, sigma_v: float = 1, marhaba_t: int = -1
):
    """Run training for time-series forecasting model"""
    # Dataset
    output_col = [0]
    num_features = 1
    input_seq_len = 24
    output_seq_len = 1
    seq_stride = 1
    # Number of observations before training time to be inferred. These
    # obervations are nan in training data.
    infer_window_len = 48

    train_dtl = TimeSeriesDataloader(
        x_file="data/toy_time_series_smoother/x_train_sin_smoother.csv",
        date_time_file="data/toy_time_series_smoother/x_train_sin_smoother_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        # time_covariates=["hour_of_day", "day_of_week"],
        # keep_last_time_cov=True,
    )
    test_dtl = TimeSeriesDataloader(
        x_file="data/toy_time_series_smoother/x_test_sin_smoother.csv",
        date_time_file="data/toy_time_series_smoother/x_test_sin_smoother_datetime.csv",
        output_col=output_col,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        num_features=num_features,
        stride=seq_stride,
        x_mean=train_dtl.x_mean,
        x_std=train_dtl.x_std,
        # time_covariates=["hour_of_day", "day_of_week"],
        # keep_last_time_cov=True,
    )

    # Set random seed
    manual_seed(235)

    # Network
    net = Sequential(
        SLSTM(num_features + input_seq_len - 1, 10, 1),
        SLinear(10, 1),
    )

    # net.to_device("cuda")
    net.set_threads(1)  # multi-processing is slow on a small net
    net.input_state_update = True
    net.num_samples = train_dtl.dataset["value"][0].shape[0]
    out_updater = OutputUpdater(net.device)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    # Initialize the sequence length
    mu_sequence = np.ones(input_seq_len, dtype=np.float32)
    pbar = tqdm(range(num_epochs), desc="Training Progress")

    lstm_states = []

    for epoch in pbar:
        print(net.get_lstm_states())

        batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

        # Decaying observation's variance
        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.3, decaying_factor=0.99, curr_iter=epoch
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)
        y_train = []

        # for x, y in batch_iter:
        for idx_sample, (x, y) in enumerate(batch_iter):

            # replace nan in input x by the lstm_prediction:
            if idx_sample < input_seq_len + infer_window_len:
                x = replace_with_prediction(x, mu_sequence)

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

            # store lstm_states
            lstm_states.append(net.get_lstm_states())

            # Training metric
            pred = normalizer.unstandardize(
                m_pred,
                train_dtl.x_mean[output_col],
                train_dtl.x_std[output_col],
            )
            y_train.append(y)
            obs = normalizer.unstandardize(
                y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
            )

            mse = metric.mse(pred, obs)
            mses.append(mse)

            # Add new prediction to mu_sequence
            mu_sequence = np.append(mu_sequence, m_pred)
            mu_sequence = mu_sequence[-input_seq_len:]

        # print(net.get_slstm_smooth_states()[0]["mu_h_priors"])
        # print(net.get_slstm_smooth_states()[0]["mu_h_posts"])
        # print(net.get_slstm_smooth_states()[0]["mu_h_smooths"])

        # import sys
        # sys.exit()

        # Smoother
        mu_zo_smooth, var_zo_smooth = net.smoother()
        zo_smooth_std = np.array(var_zo_smooth) ** 0.5
        mu_sequence = np.ones(input_seq_len, dtype=np.float32)

        # Figures for each epoch for debugging
        # t = np.arange(len(mu_zo_smooth))
        # t_train = np.arange(len(y_train))
        # plt.figure()
        # plt.plot(t_train, y_train, color="r")
        # plt.plot(t, mu_zo_smooth, color="b")
        # plt.fill_between(
        #     t,
        #     mu_zo_smooth - zo_smooth_std,
        #     mu_zo_smooth + zo_smooth_std,
        #     alpha=0.2,
        #     label="1 Std Dev",
        # )
        # filename = f"saved_results/smoother#{epoch}.png"
        # plt.savefig(filename)
        # plt.close()

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {np.nansum(mses)/np.sum(~np.isnan(mses)):>7.2f}",
            refresh=True,
        )

    # Plot final smoothed values
    t = np.arange(len(mu_zo_smooth))
    t_train = np.arange(len(y_train))
    plt.figure(figsize=(12, 12))
    plt.title("Smoothed SLSTM Output", fontsize=1.1 * 28, fontweight="bold")
    plt.plot(t_train, y_train, color="k", lw=3, label=r"$y_{true}$")
    plt.plot(t, mu_zo_smooth, color="r", lw=3, label=r"$\mathbb{E}[Y^{'}]$")
    plt.fill_between(
        t,
        mu_zo_smooth - zo_smooth_std,
        mu_zo_smooth + zo_smooth_std,
        color="r",
        alpha=0.3,
        label=r"$\mathbb{{E}}[Y^{{'}}]\pm{}\sigma$".format(1),
    )
    plt.legend(
        loc="upper right",
        edgecolor="black",
        fontsize=28,
        ncol=1,
        framealpha=0.3,
        frameon=False,
    )
    plt.xlabel(r"$x$", fontsize=28)
    plt.ylabel(r"$y$", fontsize=28)
    plt.ylim(-3, 3)
    plt.tick_params(axis="both", which="both", direction="inout", labelsize=28)
    filename = f"saved_results/smoothed_look_back_toy_time_series.png"
    plt.savefig(filename)
    plt.close()

    # -------------------------------------------------------------------------#
    # set lstm states at a certain time step
    net.set_lstm_states(lstm_states[marhaba_t])

    # Testing
    test_batch_iter = test_dtl.create_data_loader(batch_size, shuffle=False)
    mu_preds = []
    var_preds = []
    y_test = []
    x_test = []

    for RW_idx_, (x, y) in enumerate(test_batch_iter):
        # Rolling window predictions
        RW_idx = RW_idx_ % (input_seq_len)
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

    # Compute log-likelihood
    mse = metric.mse(mu_preds, y_test)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_test, std=std_preds
    )

    y_train = normalizer.unstandardize(
        y_train,
        train_dtl.x_mean[output_col],
        train_dtl.x_std[output_col],
    )

    # Visualization of all data with the prediction
    plt.figure(figsize=(10, 2))
    # add all data
    plt.plot(
        pd.to_datetime(train_dtl.dataset["date_time"][24:]),
        y_train,
        color="red",
    )
    plt.plot(
        pd.to_datetime(test_dtl.dataset["date_time"][-24:]),
        y_test,
        label="y_test",
        color="red",
    )
    plt.plot(
        pd.to_datetime(test_dtl.dataset["date_time"][-24:]),
        mu_preds,
        label="mu_preds",
        color="blue",
    )
    plt.fill_between(
        pd.to_datetime(test_dtl.dataset["date_time"][-24:]),
        mu_preds - std_preds,
        mu_preds + std_preds,
        color="blue",
        alpha=0.2,
    )
    plt.title(f"Test MSE: {mse:.2f} | Log-likelihood: {log_lik:.2f}")
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.grid()
    plt.show()


def replace_with_prediction(x, mu_sequence):
    nan_indices = np.where(np.isnan(x))[0]
    x[nan_indices] = mu_sequence[nan_indices]
    return x


if __name__ == "__main__":
    main(marhaba_t=-1)

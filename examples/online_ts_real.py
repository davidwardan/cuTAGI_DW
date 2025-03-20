import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

import pytagi.metric as metric
from pytagi import exponential_scheduler, manual_seed, Utils
from pytagi.nn import SLSTM, SLinear, OutputUpdater, Sequential


plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

import matplotlib as mpl

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
    }
)

# set line width to 1
mpl.rcParams["lines.linewidth"] = 1

# Instantiate the Utils class
utils = Utils()


# Prepare data into windows
def prepare_windows(data, window_size):
    x = []
    for i in range(len(data) - window_size + 1):
        x.append(data[i : i + window_size])
    return np.array(x, dtype=np.float32)


def main(y, t, test_index, window_size, sigma_v, change_points=None):

    train_data = y[:test_index]
    test_data = y[test_index:]

    train_y = train_data.copy()
    test_y = test_data.copy()

    # Visualize the synthetic data
    train_t = t[:test_index]
    test_t = t[test_index:]

    # Get max and min values for y
    y_max = np.max(y) + 0.25
    y_min = np.min(y) - 0.25

    # global parameters
    output_col = [0]
    num_features = 1
    input_seq_len = 52
    output_seq_len = 1
    batch_size = 1  # only set to 1

    plt.figure(figsize=(6, 1))
    plt.axvspan(
        train_t[0],
        train_t[-1],
        facecolor="dodgerblue",
        alpha=0.2,
        label="Online Learning",
        edgecolor="none",
        linewidth=0,
    )
    plt.plot(train_t, train_data, "r", label=r"$y_{true}$")
    plt.plot(test_t, test_data, "r")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(y_min, y_max)
    plt.legend(loc=(0.2, 1.01), ncol=2, frameon=False, columnspacing=0.5)
    plt.savefig(
        "./out/real_time_series.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Load data into windows
    train_data = prepare_windows(train_data, window_size)

    # Fixed TAGI seed
    manual_seed(42)

    # Network
    net = Sequential(
        SLSTM(input_seq_len, 50, 1),
        SLSTM(50, 50, 1),
        SLinear(50, 1),
    )
    net.set_threads(1)  # only set to 1
    out_updater = OutputUpdater(net.device)
    net.num_samples = window_size + input_seq_len
    var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

    # Training
    mu_preds = []
    S_preds = []
    mses = []
    log_liks = []

    first_lookback = np.ones(input_seq_len, dtype=np.float32)

    for idx, window in tqdm(
        enumerate(train_data), desc="Processing windows", total=len(train_data)
    ):
        window = np.concatenate((first_lookback, window))  # add first lookback
        x_rolled, y_rolled = utils.create_rolling_window(
            window, output_col, input_seq_len, output_seq_len, num_features, 1
        )

        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=1, decaying_factor=0.99, curr_iter=idx
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        # update first lookback for the next window
        first_lookback = window[1 : input_seq_len + 1]

        window_mse = 0
        window_log_lik = 0
        # iterate over the rolled windows of the smoothing window
        with tqdm(total=len(x_rolled), desc=f"Window {idx}", leave=False):
            for i, (x, y) in enumerate(zip(x_rolled, y_rolled)):
                # forward pass
                mu_pred, S_pred = net(x)
                if idx == 0:
                    mu_preds.append(mu_pred)
                    S_preds.append(S_pred)
                # elif i == len(x_rolled) - 1:
                #     mu_preds.append(mu_pred)
                #     S_preds.append(S_pred)

                # update output
                out_updater.update(
                    output_states=net.output_z_buffer,
                    mu_obs=y,
                    var_obs=var_y,
                    delta_states=net.input_delta_z_buffer,
                )

                # Feed backward
                net.backward()
                net.step()

                # Compute metrics
                mse = metric.mse(mu_pred, y)
                mses.append(mse)
                window_mse += mse

                log_lik = metric.log_likelihood(y, mu_pred, S_pred)
                log_liks.append(log_lik)
                window_log_lik += log_lik

            # do a one-step prediction
            some_x = np.concatenate((x_rolled[-1], y_rolled[-1]))
            mu_pred, S_pred = net(some_x[-input_seq_len:])
            mu_preds.append(mu_pred)
            S_preds.append(S_pred)

        # Smooth values
        if idx == train_data.shape[0] - 1:
            # net.smoother(online=False)
            pass
        else:
            net.smoother(online=True)

        # Print average MSE for this window
        tqdm.write(
            f"Window {idx}, Average MSE: {window_mse/window_size:.6f}, Average Log-lik: {window_log_lik/window_size:.3f}"
        )

    # Compute residuals
    # remove the last element
    mu_preds.pop()
    S_preds.pop()

    residuals = train_y - np.array(mu_preds).flatten()

    # Create a figure with two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

    mu_preds = np.array(mu_preds).flatten()
    S_preds = np.array(S_preds).flatten()

    # Subplot 1: Visualize the predictions
    ax1.axvspan(
        train_t[0] + pd.DateOffset(weeks=window_size),
        train_t[-1],
        facecolor="dodgerblue",
        alpha=0.2,
        label="Online Learning",
        edgecolor="none",
        linewidth=0,
    )
    ax1.axvspan(
        train_t[0],
        train_t[0] + pd.DateOffset(weeks=window_size),
        alpha=0.3,
        facecolor="green",
        label=r"$\mathtt{D}$",
        edgecolor="none",
        linewidth=0,
    )
    ax1.plot(train_t, train_y, "r", label=r"$y_{true}$", linewidth=1)
    ax1.plot(train_t, mu_preds, "b", label=r"$\mathbb{E}[Y']$", linewidth=1)
    ax1.fill_between(
        train_t,
        mu_preds - np.sqrt(S_preds),
        mu_preds + np.sqrt(S_preds),
        facecolor="blue",
        alpha=0.3,
        label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$",
    )
    ax1.plot(test_t, test_y, "r")
    ax1.set_ylabel("Value")
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc=(0.0, 1.01), ncol=5, frameon=False, columnspacing=0.5)

    # Subplot 2: Visualize the residuals
    ax2.axvspan(
        train_t[0],
        train_t[0] + pd.DateOffset(weeks=window_size),
        facecolor="green",
        alpha=0.3,
        edgecolor="none",
        linewidth=0,
    )
    ax2.plot(train_t, residuals, "k-", label=r"$r_t$", linewidth=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.set_ylim(y_min, y_max)
    ax2.legend(loc=(0.4, 1.01), ncol=1, frameon=False, columnspacing=0.5)

    fig.tight_layout()
    fig.savefig(
        "./out/real_predict.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Test the model
    input_seq = train_y[-input_seq_len:].copy()
    input_seq = np.array(input_seq, dtype=np.float32)
    recursive_mu_preds = []
    recursive_S_preds = []

    # Define the forecasting horizon as the length of test_data
    for y_test in test_data:

        # Forward
        mu_pred, S_pred = net(input_seq)
        recursive_mu_preds.append(mu_pred)
        recursive_S_preds.append(S_pred)

        # remove the first element and append the new prediction
        input_seq = np.roll(input_seq, -1)
        input_seq[-1] = mu_pred.item()

    # Visualize the predictions
    recursive_mu_preds = np.array(recursive_mu_preds).flatten()
    recursive_S_preds = np.array(recursive_S_preds).flatten()
    plt.figure(figsize=(6, 1))
    plt.plot(train_t, train_y, "r")
    plt.axvspan(
        train_t[0] + pd.DateOffset(weeks=window_size),
        train_t[-1],
        alpha=0.2,
        facecolor="dodgerblue",
        label="Online Learning",
        edgecolor="none",
        linewidth=0,
    )
    plt.plot(train_t, mu_preds, "b")
    plt.axvspan(
        train_t[0],
        train_t[0] + pd.DateOffset(weeks=window_size),
        facecolor="green",
        alpha=0.3,
        edgecolor="none",
        linewidth=0,
        label=r"$\mathtt{D}$",
    )
    plt.fill_between(
        train_t,
        mu_preds - np.sqrt(S_preds),
        mu_preds + np.sqrt(S_preds),
        facecolor="blue",
        alpha=0.3,
    )
    plt.plot(test_t, test_y, "r", label=r"$y_{true}$")
    plt.plot(test_t, recursive_mu_preds, "b", label=r"$\mathbb{E}[Y']$")
    plt.fill_between(
        test_t,
        recursive_mu_preds - np.sqrt(recursive_S_preds),
        recursive_mu_preds + np.sqrt(recursive_S_preds),
        facecolor="blue",
        alpha=0.3,
        label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$",
    )
    plt.legend(loc=(-0.01, 1.01), ncol=5, frameon=False, columnspacing=0.5)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(y_min, y_max)
    plt.savefig(
        "./out/real_forecast.pdf",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    # Calculate the MSE and log likelihood for the test data
    test_mse = metric.mse(test_y, recursive_mu_preds)
    test_log_likelihood = metric.log_likelihood(
        test_y, recursive_mu_preds, recursive_S_preds
    )

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test Log Likelihood: {test_log_likelihood:.6f}")


if __name__ == "__main__":

    # Fix random seed
    np.random.seed(42)

    data_csv = "./data/hq_data/weekly/hq_train_weekly_values.csv"
    dates_csv = "./data/hq_data/weekly/hq_train_weekly_dates.csv"

    # read data
    df_time = pd.read_csv(dates_csv)
    df_time = df_time.iloc[:, 0]

    df_data = pd.read_csv(data_csv)
    col_name = df_data.columns[20]
    y_values = df_data.iloc[:, 20]

    # use 10% of the data for testing
    test_index = int(0.9 * len(y_values))
    y = y_values.values
    t = pd.to_datetime(df_time.values)

    main(y, t, test_index, window_size=52, sigma_v=4)

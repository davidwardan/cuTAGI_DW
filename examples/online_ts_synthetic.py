import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys

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


# Define function to generate time series data with changing amplitude and frequency
def generate_changing_amplitude_sine(
    frequency=1, phase=0, sampling_rate=100, duration=10, change_points=None
):
    """
    Generate a sine wave time series with variable amplitude and frequency,
    ensuring continuity at changepoints by adjusting the phase.

    If `change_points` is None, a constant amplitude and frequency are used.
    Otherwise, the amplitude and frequency change at the specified time points,
    and the phase is updated to keep the sine wave continuous at each changepoint.

    Parameters
    ----------
    frequency : float, optional
        Default frequency of the sine wave (default is 1). This is used if a change point
        does not specify a frequency.
    phase : float, optional
        Initial phase in radians (default is 0).
    sampling_rate : int, optional
        Number of samples per second (default is 100).
    duration : int or float, optional
        Duration of the signal.
    change_points : list of tuple, optional
        Each tuple should specify (time, amplitude) or (time, amplitude, frequency).
        The amplitude and frequency change at these time points.

    Returns
    -------
    tuple
        t : ndarray
            Time points.
        y : ndarray
            Sine wave values.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    if change_points is None:
        y = np.sin(2 * np.pi * frequency * t + phase)
    else:
        y = np.zeros_like(t)
        # Initialize with the default frequency and phase for the first segment
        current_phase = phase
        current_freq = frequency

        # Process each segment defined by change_points
        for i in range(len(change_points) - 1):
            cp = change_points[i]
            start_time = cp[0]
            amplitude = cp[1]
            seg_freq = cp[2] if len(cp) > 2 else frequency

            # For segments after the first, adjust phase to ensure continuity
            if i > 0:
                # t_c is the current changepoint time
                t_c = start_time
                # Adjust phase so that:
                # sin(2*pi*seg_freq*t_c + new_phase) = sin(2*pi*current_freq*t_c + current_phase)
                current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
                    2 * np.pi * seg_freq * t_c
                )
                current_freq = seg_freq

            # Determine end time for this segment
            next_cp = change_points[i + 1]
            end_time = next_cp[0]
            mask = (t >= start_time) & (t < end_time)
            y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)

        # Handle the final segment
        last_cp = change_points[-1]
        start_time = last_cp[0]
        amplitude = last_cp[1]
        seg_freq = last_cp[2] if len(last_cp) > 2 else frequency
        if len(change_points) > 1:
            t_c = start_time
            current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
                2 * np.pi * seg_freq * t_c
            )
        mask = t >= start_time
        y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)
    return t, y


# Prepare data into windows
def prepare_windows(data, window_size):
    x = []
    for i in range(len(data) - window_size + 1):
        x.append(data[i : i + window_size])
    return np.array(x, dtype=np.float32)


def adjust_parameters(net, idx, window_size, change_points):
    """
    Adjust the network's state dictionary at changepoints.

    For each changepoint (except the first one), if the current index matches the changepoint time minus the window_size,
    update the variances of the weights and biases by adding 2e-4.

    Parameters:
        net: The neural network whose parameters are to be adjusted.
        idx: The current window index in the training loop.
        window_size: The size of the sliding window.
        change_points: A list of changepoint tuples.
    """
    if change_points is not None:
        for change in change_points[1:]:
            if idx == change[0] - window_size:
                state_dict = net.state_dict()
                for layer_name, (mu_w, var_w, mu_b, var_b) in state_dict.items():
                    var_w = [x + 2e-4 for x in var_w]
                    var_b = [x + 2e-4 for x in var_b]
                    state_dict[layer_name] = (mu_w, var_w, mu_b, var_b)
                net.load_state_dict(state_dict)


def main(
    y, t, test_index, window_size, sigma_v, change_points=None, intervention=False
):

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
    input_seq_len = 12
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
        "./out/time_series.pdf", bbox_inches="tight", pad_inches=0, transparent=True
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
            curr_v=sigma_v, min_v=0.5, decaying_factor=0.99, curr_iter=idx
        )
        var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

        # update first lookback for the next window
        first_lookback = window[1 : input_seq_len + 1]

        window_mse = 0
        window_log_lik = 0
        # iterate over the rolled windows of the smoothing window
        with tqdm(total=len(x_rolled), desc=f"Window {idx}", leave=False):
            for i, (x, y) in enumerate(zip(x_rolled, y_rolled)):

                # adjust network parameters at changepoints
                if intervention:
                    adjust_parameters(net, idx, window_size, change_points)

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
    mu_preds.pop()
    S_preds.pop()

    residuals = train_y - np.array(mu_preds).flatten()

    # Create a figure with two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

    mu_preds = np.array(mu_preds).flatten()
    S_preds = np.array(S_preds).flatten()

    # Subplot 1: Visualize the predictions
    ax1.axvspan(
        window_size,
        train_t[-1],
        facecolor="dodgerblue",
        alpha=0.2,
        label="Online Learning",
        edgecolor="none",
        linewidth=0,
    )
    ax1.axvspan(
        train_t[0],
        train_t[0] + window_size,
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
        train_t[0] + window_size,
        facecolor="green",
        alpha=0.3,
        edgecolor="none",
        linewidth=0,
    )
    ax2.plot(train_t, residuals, "k-", label=r"$r_t$", linewidth=1)
    if change_points is not None and intervention:
        for change in change_points[1:]:
            ax2.axvline(x=change[0], color="r", linestyle="--")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.set_ylim(y_min, y_max)
    ax2.legend(loc=(0.4, 1.01), ncol=1, frameon=False, columnspacing=0.5)

    fig.tight_layout()
    fig.savefig(
        "./out/synthetic_predict.pdf",
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
        window_size,
        train_t[-1],
        alpha=0.2,
        facecolor="dodgerblue",
        label="Online Learning",
        edgecolor="none",
        linewidth=0,
    )
    plt.plot(train_t, mu_preds, "b")
    plt.axvspan(
        0,
        window_size,
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
    if change_points is not None and intervention:
        for change in change_points[1:]:
            plt.axvline(x=change[0], color="r", linestyle="--")
    plt.legend(loc=(-0.01, 1.01), ncol=5, frameon=False, columnspacing=0.5)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(y_min, y_max)
    plt.savefig(
        "./out/synthetic_forecast.pdf",
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

    # Generate synthetic data
    frequency = 1 / 24  # One cycle per 24 hours
    phase = 0  # Initial phase
    sampling_rate = 1  # 1 sample per hour
    duration = 1 / frequency * 25  # Total duration
    change_points = [(0, 1), (24 * 8, 1.5), (24 * 12, 1), (24 * 16, 1.5, 1 / 48)]

    t, y = generate_changing_amplitude_sine(
        frequency=frequency,
        phase=phase,
        sampling_rate=sampling_rate,
        duration=duration,
        change_points=change_points,
    )

    # Get index for test split
    num_test_cycles = 2
    period = 48
    test_duration = num_test_cycles * period
    test_index = len(y) - int(test_duration * sampling_rate)

    main(
        y,
        t,
        test_index,
        window_size=24,
        sigma_v=10,
        change_points=change_points,
        intervention=True,
    )

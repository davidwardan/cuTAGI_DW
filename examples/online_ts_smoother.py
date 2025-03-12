import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler, manual_seed, Utils
from pytagi.nn import SLSTM, SLinear, OutputUpdater, Sequential

from examples.data_loader import TimeSeriesDataloader

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "figure.figsize": [6, 2],
        "font.size": 12,
    }
)


# Instantiate the Utils class
utils = Utils()

# fix random seed
np.random.seed(42)


# Define function to generate time series data with changing amplitude
def generate_changing_amplitude_sine(
    frequency=1, phase=0, sampling_rate=100, duration=10, change_points=None
):
    """
    Generate a sine wave time series with variable amplitude.

    If ``change_points`` is None, a constant amplitude of 1 is used.
    Otherwise, the amplitude changes at the given time points.

    Parameters
    ----------
    frequency : float, optional
        Frequency of the sine wave (default is 1).
    phase : float, optional
        Initial phase in radians (default is 0).
    sampling_rate : int, optional
        Number of samples per second (default is 100).
    duration : int or float, optional
        Total duration of the time series in seconds (default is 10).
    change_points : list of tuple, optional
        Each tuple should specify (time, amplitude). The amplitude changes
        at these time points.

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
        # Use a constant amplitude of 1 across the entire signal
        y = np.sin(2 * np.pi * frequency * t + phase)
    else:
        y = np.zeros_like(t)
        for i in range(len(change_points) - 1):
            start_time, amp = change_points[i]
            end_time, _ = change_points[i + 1]
            mask = (t >= start_time) & (t < end_time)
            y[mask] = amp * np.sin(2 * np.pi * frequency * t[mask] + phase)

        last_time, last_amp = change_points[-1]
        mask = t >= last_time
        y[mask] = last_amp * np.sin(2 * np.pi * frequency * t[mask] + phase)

    return t, y


# Generate time series data with changing amplitude
# Set parameters for hourly data with each cycle having 24 data points (i.e., one day)
frequency = 1 / 24  # One cycle per 24 hours
phase = 0
sampling_rate = 1  # 1 sample per hour
duration = 480  # Total duration in hours (e.g., 10 days of data)
# change_points = [(0, 1), (5, 2), (10, 0.5), (15, 1.5)]
change_points = [(0, 1), (24 * 10, 2)]

t, y = generate_changing_amplitude_sine(
    frequency=frequency,
    phase=phase,
    sampling_rate=sampling_rate,
    duration=duration,
    change_points=change_points,
)

# global parameters
output_col = [0]
num_features = 1
input_seq_len = 1
output_seq_len = 1
window_size = 24  # window size for online training
batch_size = 1  # so far only batch_size=1 is supported
sigma_v = 4  # initial sigma_v


# prepare data into windows
def prepare_data(data, window_size):
    x = []
    for i in range(len(data) - window_size + 1):
        x.append(data[i : i + window_size])
    return np.array(x, dtype=np.float32)

# split data into training and testing
split_ratio = 0.8

train_data = y[: int(0.8 * len(y))]
test_data = y[int(0.8 * len(y)) :]

train_y = train_data.copy()
test_y = test_data.copy()

# Visualize the synthetic data
train_t = t[: int(0.8 * len(y))]
test_t = t[int(0.8 * len(y)) :]

plt.figure()
plt.plot(train_t, train_data, "b-", label="Training Data")
plt.plot(test_t, test_data, "r-", label="Testing Data")
plt.legend(loc=(0.2, 1.01), ncol=2, frameon=False)
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("time_series.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# data = normalizer(data)
train_data = prepare_data(train_data, window_size)

# Network
net = Sequential(
    SLSTM(input_seq_len, 10, 1),
    SLSTM(10, 10, 1),
    SLinear(10, 1),
)
manual_seed(42)  # TAGI seed
net.set_threads(8)  # multi-processing is slow on a small net
out_updater = OutputUpdater(net.device)
net.num_samples = (
    window_size + input_seq_len
)  # equal to the window size + lookback period
net.input_state_update = True
var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

mu_preds = []
S_preds = []
mses = []
# Training
first_lookback = np.ones(input_seq_len, dtype=np.float32)

i = 0
# mu_preds = [0.0] * (train_data.shape[0] + train_data.shape[1] - 1)
# S_preds = [0.0] * (train_data.shape[0] + train_data.shape[1] - 1)
mu_preds = []
S_preds = []
parameters = []  # store parameters for each window

prediction_idx = 0
total_mse = 0

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
    prediction_idx = idx

    # iterate over the rolled windows of the smoothing window
    for i, (x, y) in enumerate(
        tqdm(
            zip(x_rolled, y_rolled),
            desc=f"Window {idx}",
            leave=False,
            total=len(x_rolled),
        )
    ):
        # forward pass
        mu_pred, S_pred = net(x)
        if idx == 0:
            mu_preds.append(mu_pred)
            S_preds.append(S_pred)
        # mu_preds[prediction_idx] = mu_pred
        # S_preds[prediction_idx] = S_pred
        prediction_idx += 1

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

        # Compute MSE
        mse = metric.mse(mu_pred, y)
        mses.append(mse)
        total_mse += mse

    # Smooth values
    net.smoother(online=True)

    # append the last prediction
    if idx > 0:
        mu_preds.append(mu_pred)
        S_preds.append(S_pred)

    # Print average MSE for this window
    avg_mse = total_mse / prediction_idx if prediction_idx > 0 else 0
    print(f"Window {idx}, Average MSE: {avg_mse:.6f}, Current σ: {sigma_v:.3f}")

# Visualize the predictions
plt.figure()
mu_preds = np.array(mu_preds).flatten()
S_preds = np.array(S_preds).flatten()
plt.plot(train_t, train_y, "k", label=r"$y_{true}$")
plt.plot(train_t, mu_preds, "r", label=r"$\mathbb{E}[Y']$")
plt.fill_between(
    train_t,
    mu_preds - np.sqrt(S_preds),
    mu_preds + np.sqrt(S_preds),
    facecolor="red",
    alpha=0.3,
    label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(1),
)
plt.axvspan(0, window_size, alpha=0.2, color="forestgreen", zorder=0)
plt.legend(loc=(0.2, 1.01), ncol=3, frameon=False)
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("pred.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# Test the model
x = np.concatenate((train_y[-input_seq_len:], test_data))
x_rolled, y_rolled = utils.create_rolling_window(
    x, output_col, input_seq_len, output_seq_len, num_features, 1
)

mu_preds = []
S_preds = []
for x, y in zip(x_rolled, y_rolled):

    # Forward pass
    mu_pred, S_pred = net(x)
    mu_preds.append(mu_pred)
    S_preds.append(S_pred)

# Visualize the predictions
mu_preds = np.array(mu_preds).flatten()
S_preds = np.array(S_preds).flatten()
plt.figure()
plt.plot(train_t, train_y, "k")
plt.axvspan(train_t[0], train_t[-1], alpha=0.2)
plt.plot(test_t, test_y, "k", label=r"$y_{true}$")
plt.plot(test_t, mu_preds, "r", label=r"$\mathbb{E}[Y']$")
plt.fill_between(
    test_t,
    mu_preds - np.sqrt(S_preds),
    mu_preds + np.sqrt(S_preds),
    facecolor="red",
    alpha=0.3,
    label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(1),
)
plt.legend(loc=(0.2, 1.01), ncol=3, frameon=False)
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("pred_future.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

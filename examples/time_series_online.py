# %%
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
from pytagi.nn import SLSTM, SLinear, OutputUpdater, Sequential, Linear, LSTM

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

# Parameters for the sinusoidal data
frequency = 1
phase = 0
sampling_rate = 100
duration = 10

# Define time/amplitude segments (time, amplitude)
change_points = [(0, 1), (5, 2), (10, 0.5), (15, 1.5)]

# Generate time points
t = np.linspace(0, duration, int(sampling_rate * duration))
y = np.zeros_like(t)

# Generate data with amplitude changes
for i in range(len(change_points) - 1):
    start_time, amp = change_points[i]
    end_time, _ = change_points[i + 1]
    mask = (t >= start_time) & (t < end_time)
    y[mask] = amp * np.sin(2 * np.pi * frequency * t[mask] + phase)

# Handle final segment
last_time, last_amp = change_points[-1]
mask = t >= last_time
y[mask] = last_amp * np.sin(2 * np.pi * frequency * t[mask] + phase)

# Visualize the synthetic data
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
plt.savefig("time_series.svg", bbox_inches="tight", pad_inches=0, transparent=True)


# define function to split data into windows
def split_into_windows(data, window_size):
    """
    Splits the input array `data` into windows of size `window_size` with a stride of 1,
    and associates each window with its forecast (the next value).

    Args:
        data (np.ndarray): Input array.
        window_size (int): Size of each window.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: 2D array where each row is a window of size `window_size` (x).
            - np.ndarray: 1D array of forecast points corresponding to each window (y).
    """
    num_windows = len(data) - window_size
    x = np.array([data[i : i + window_size] for i in range(num_windows)])
    y = np.array([data[i + window_size] for i in range(num_windows)])
    return x, y


window_size = 20
batch_size = 1
output_col = [0]
X, Y = split_into_windows(y, window_size)
num_windows = X.shape[1]


input_seq_len = 10
output_seq_len = 1

# Network
net = Sequential(
    SLSTM(input_seq_len, 20, 1),
    SLSTM(20, 20, 1),
    SLinear(20, 1),
)
net.set_threads(1)  # multi-processing is slow on a small net
out_updater = OutputUpdater(net.device)
sigma_v = 0.3
net.num_samples = window_size + input_seq_len
net.input_state_update = True
var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

mses = []
y_preds = []
prev_states = np.zeros(input_seq_len, dtype=np.float32)
for x, y in zip(X, Y):

    x = x.astype(np.float32)
    y = np.array([y], dtype=np.float32)

    # concatenate ones to beginning of x with input seq_len
    x = np.concatenate((prev_states, x), axis=0)
    prev_states = x[-input_seq_len:]

    # Feed forward
    y_pred, _ = net(x)
    print(len(y_pred))
    y_preds.append(y_pred[-1])

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

    # Compute MSE
    mse = metric.mse(y_pred, y)
    mses.append(mse)

    # smooth states
    mu_zo_smooth, var_zo_smooth = net.smoother()

# visualize true data with predicted data
plt.plot(t[window_size:], Y, label="True data")
plt.plot(t[window_size:], y_preds, label="Predicted data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc=(0.25, 1.04), ncol=2, frameon=False)
plt.savefig(
    "time_series_online.svg", bbox_inches="tight", pad_inches=0, transparent=True
)

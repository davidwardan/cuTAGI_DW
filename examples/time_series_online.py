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

# Parameters for the sinusoidal data
frequency = 1  # Frequency of the sine wave
amplitude = 1  # Amplitude of the sine wave
phase = 0  # Phase shift
sampling_rate = 100  # Points per unit time
duration = 10  # Duration in seconds

# Generate time points
t = np.linspace(0, duration, int(sampling_rate * duration))

# Generate sinusoidal data
y = amplitude * np.sin(2 * np.pi * frequency * t + phase)

# # Visualize the synthetic data
# plt.figure(figsize=(10, 5))
# plt.plot(t, y)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.show()


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


window_size = 10
batch_size = 1
output_col = [0]
X, Y = split_into_windows(y, window_size)
num_windows = X.shape[1]


input_seq_len = 10
output_seq_len = 1

# Network
net = Sequential(
    SLSTM(input_seq_len, 40, 1),
    SLSTM(40, 40, 1),
    SLinear(40, 1),
)
net.set_threads(1)  # multi-processing is slow on a small net
out_updater = OutputUpdater(net.device)
sigma_v = 1.0
net.num_samples = window_size + input_seq_len
net.input_state_update = True
var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

mses = []
for x, y in zip(X, Y):

    x = x.astype(np.float32)
    y = np.array([y], dtype=np.float32)

    # concatenate ones to beginning of x with input seq_len
    nan_input = np.ones(input_seq_len, dtype=np.float32)
    x = np.concatenate((nan_input, x), axis=0)

    y_pred, _ = net(x)

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

    mse = metric.mse(y_pred, y)
    mses.append(mse)

    # smooth states
    net.smoother()
    mu, var = net.get_outputs_smoother()

    # print(mu_zo_smooth)

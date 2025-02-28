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
frequency = 1
phase = 0
sampling_rate = 100
duration = 20
# change_points = [(0, 1), (5, 2), (10, 0.5), (15, 1.5)]

t, y = generate_changing_amplitude_sine(
    frequency=frequency,
    phase=phase,
    sampling_rate=sampling_rate,
    duration=duration,
    # change_points=change_points,
)

y_true = y.copy()

# Visualize the synthetic data
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("time_series.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# global parameters
output_col = [0]
num_features = 1
input_seq_len = 5
output_seq_len = 1
window_size = 20  # window size for online training
batch_size = 1  # so far only batch_size=1 is supported
sigma_v = 2


# prepare data into windows
def prepare_data(data, window_size):
    x = []
    for i in range(len(data) - window_size + 1):
        x.append(data[i : i + window_size])
    return x


# prepare data
data = y
# data = normalizer(data)
data = prepare_data(data, window_size)
data = np.array(data, dtype=np.float32)

# Network
net = Sequential(
    SLSTM(input_seq_len, 10, 1),
    SLSTM(10, 10, 1),
    # SLSTM(10, 10, 1),
    SLinear(10, 1),
)
manual_seed(42)  # TAGI seed
net.set_threads(1)  # multi-processing is slow on a small net
out_updater = OutputUpdater(net.device)
net.num_samples = window_size + input_seq_len  # TODO: what should this be?
net.input_state_update = True
var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

mu_preds = []
S_preds = []
mses = []
# Training
first_lookback = np.ones(
    input_seq_len, dtype=np.float32
)  # set to 1 for the first lookback


i = 0
mu_preds = [0.0] * (len(data) * (window_size - input_seq_len + 1))
S_preds = [0.0] * (len(data) * (window_size - input_seq_len + 1))

for idx, window in tqdm(enumerate(data)):
    window = np.concatenate((first_lookback, window))  # add first lookback
    x_rolled, y_rolled = utils.create_rolling_window(
        window, output_col, input_seq_len, output_seq_len, num_features, 1
    )

    # update first lookback for the next window
    first_lookback = window[1 : input_seq_len + 1]

    # prediction index
    prediction_idx = idx

    # iterate over the rolled windows
    for x, y in zip(x_rolled, y_rolled):
        x = x.astype(np.float32)
        y = np.array(y, dtype=np.float32)
        # print(prediction_idx)

        # forward pass
        mu_pred, S_pred = net(x)
        mu_preds[prediction_idx] = mu_pred
        S_preds[prediction_idx] = S_pred
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

        # get hidden and cell states
        hidden_states = net.get_all_hidden_states()
        cell_states = net.get_all_cell_states()
        print(len(hidden_states['SLSTM(10,10)_1']))
        print(len(cell_states['SLSTM(10,10)_1']))

        # Smooth values
        net.smoother()

        smoothed_hidden_states = net.get_all_hidden_states()
        smoothed_cell_states = net.get_all_cell_states()
        print(len(smoothed_hidden_states['SLSTM(10,10)_1']))
        print(len(smoothed_cell_states['SLSTM(10,10)_1']))

        sys.exit()

        # Compute MSE
        mse = metric.mse(mu_pred, y)
        mses.append(mse)

# slice the predictions
mu_preds = mu_preds[:prediction_idx]
S_preds = S_preds[:prediction_idx]

# Visualize the predictions
plt.figure()
y_preds = np.array(mu_preds).flatten()
S_preds = np.array(S_preds).flatten()
plt.plot(y_true, "k", label=r"$y_{true}$")
plt.plot(mu_preds, "r", label=r"$\mathbb{E}[Y']$")
plt.fill_between(
    range(len(y_preds)),
    y_preds - np.sqrt(S_preds),
    y_preds + np.sqrt(S_preds),
    facecolor="red",
    alpha=0.3,
    label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(1),
)
plt.legend(loc=(0.2, 1.01), ncol=3, frameon=False)
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("pred.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

# synthesize more data for prediction
data_forecast = y_true
data_forecast = data_forecast[-(250 + input_seq_len) :]
x = data_forecast.astype(np.float32)

x_rolled, y_rolled = utils.create_rolling_window(
    x, output_col, input_seq_len, output_seq_len, num_features, 1
)

mu_preds = []
S_preds = []
for x, y in zip(x_rolled, y_rolled):
    x = x.astype(np.float32)

    # forward pass
    mu_pred, S_pred = net(x)
    mu_preds.append(mu_pred)
    S_preds.append(S_pred)

# Visualize the predictions
mu_preds = np.array(mu_preds).flatten()
S_preds = np.array(S_preds).flatten()
plt.figure()
plt.plot(y_rolled[input_seq_len:], "k", label=r"$y_{true}$")
plt.plot(mu_preds[input_seq_len:], "r", label=r"$\mathbb{E}[Y']$")
plt.fill_between(
    range(len(mu_preds[input_seq_len:])),
    mu_preds[input_seq_len:] - np.sqrt(S_preds[input_seq_len:]),
    mu_preds[input_seq_len:] + np.sqrt(S_preds[input_seq_len:]),
    facecolor="red",
    alpha=0.3,
    label=r"$\mathbb{{E}}[Y'] \pm {} \sigma$".format(1),
)
plt.legend(loc=(0.2, 1.01), ncol=3, frameon=False)
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("pred_future.pdf", bbox_inches="tight", pad_inches=0, transparent=True)

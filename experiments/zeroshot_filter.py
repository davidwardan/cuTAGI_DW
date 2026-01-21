import os
import numpy as np
from typing import Optional

from pathlib import Path

from examples.data_loader import (
    TimeSeriesDataloader,
)
from pytagi import Normalizer as normalizer
import pytagi.metric as metric

from experiments.utils import (
    build_model,
    States,
    LookBackBuffer,
    calculate_updates,
    plot_series,
    prepare_input,
)

from experiments.data_loader import (
    TimeSeriesDataBuilder,
    GlobalBatchLoader,
)

# Plotting defaults
import matplotlib as mpl
import matplotlib.pyplot as plt

# Update matplotlib parameters in a single dictionary
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,  # Set line width to 1
    }
)


# --------------------------------------------------------------


time_covariates = ["week_of_year"]
num_features = 1 + len(time_covariates)
input_seq_len = 52
seed = 99
device = "cpu"
model_input_size = input_seq_len + num_features - 1

x_file = [
    "data/hq/train100/split_train_values.csv",
]
date_file = [
    "data/hq/train100/split_train_datetimes.csv",
]
ts_idx = 17  # index of the time series to filter


data = TimeSeriesDataBuilder(
    x_file=x_file[0],
    date_time_file=date_file[0],
    input_seq_len=input_seq_len,
    output_seq_len=1,
    stride=1,
    time_covariates=time_covariates,
    scale_method="standard",
    order_mode="by_window",
    ts_to_use=[ts_idx],
)

pretrained_weights_global = (
    "experiments/pretrained_weights/global_noembeddings_seed1.pth"
)

pretrained_weights_local = (
    "experiments/pretrained_weights/local_17_noembeddings_seed1.pth"
)

# Build model
global_net, _ = build_model(
    input_size=model_input_size,
    use_AGVI=True,
    seed=seed,
    device=device,
    init_params=pretrained_weights_global,
    shift_biases=False,
)

local_net, _ = build_model(
    input_size=model_input_size,
    hidden_sizes=[40, 40],
    use_AGVI=True,
    seed=seed,
    device=device,
    init_params=pretrained_weights_local,
    shift_biases=False,
)

# Create output directory
output_dir = f"out/zeroshot/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def filter(net, data):

    # Initalize states
    cap = 853
    prior_states = States(nb_ts=1, total_time_steps=cap)
    posterior_states = States(nb_ts=1, total_time_steps=cap)

    # Initialize separate uncertainty storage
    prior_states.epistemic = np.full((1, cap), np.nan, dtype=np.float32)
    prior_states.aleatoric = np.full((1, cap), np.nan, dtype=np.float32)
    posterior_states.epistemic = np.full((1, cap), np.nan, dtype=np.float32)
    posterior_states.aleatoric = np.full((1, cap), np.nan, dtype=np.float32)

    net.eval()

    batch_iter = GlobalBatchLoader.create_data_loader(
        dataset=data.dataset,
        order_mode="by_window",
        batch_size=1,
        shuffle=False,
        seed=1,  # fixed for all seeds and runs
    )

    # Initialize look-back buffer and LSTM state container
    look_back_buffer = LookBackBuffer(input_seq_len=input_seq_len, nb_ts=1)

    y_true = []
    std_y = []
    for (x, y), _, w_id in batch_iter:

        y_true.append(y.flatten())

        # prepare look_back buffer
        if look_back_buffer.needs_initialization[0]:
            look_back_buffer.initialize(
                initial_mu=x[:, :input_seq_len],
                initial_var=np.zeros_like(x[:, :input_seq_len], dtype=np.float32),
                indices=[0],
            )

        # prepare input
        x, var_x = prepare_input(
            x=x,
            var_x=None,
            look_back_mu=(look_back_buffer.mu),
            look_back_var=(look_back_buffer.var),
            indices=np.array([0]),
        )

        # Feedforward
        m_pred, v_pred = net(x, var_x)

        # Specific to AGVI
        flat_m = np.ravel(m_pred)
        flat_v = np.ravel(v_pred)

        m_pred = flat_m[::2]  # even indices
        v_pred = flat_v[::2]  # even indices
        var_y = flat_m[1::2]  # odd indices var_v

        s_aleatoric = np.sqrt(var_y)  # scale down aleatoric uncertainty
        std_y.append(s_aleatoric)

        s_epistemic = np.sqrt(v_pred)

        # Store prior states
        prior_states.update(
            new_mu=m_pred,
            new_std=s_epistemic,
            indices=[0],
            time_step=w_id.item(),
        )

        # Update
        m_post, v_post = calculate_updates(
            net,
            _,
            m_pred,
            v_pred,
            y.flatten(),
            use_AGVI=True,
            var_y=var_y,  # scale down aleatoric uncertainty
            train_mode=False,
        )

        # TODO: check if clipping is necessary
        v_post = np.clip(v_post, a_min=1e-6, a_max=2.0)

        s_post_epistemic = np.sqrt(v_post)

        # Store posterior states
        posterior_states.update(
            new_mu=m_post,
            new_std=s_post_epistemic,
            indices=[0],
            time_step=w_id.item(),
        )

        # Update look_back buffer
        look_back_buffer.update(
            new_mu=m_post,
            new_var=v_post,
            indices=[0],
        )

    return prior_states, posterior_states, np.array(y_true), np.array(std_y)


def plot_results(prior_states, posterior_states, y_true, std_y, data, output_dir=output_dir):

    # Unstandardize results
    mean = data.x_mean[0][0]
    std = data.x_std[0][0]

    prior_states.mu = normalizer.unstandardize(prior_states.mu, mean, std)
    prior_states.std = normalizer.unstandardize_std(prior_states.std, std)

    posterior_states.mu = normalizer.unstandardize(posterior_states.mu, mean, std)
    posterior_states.std = normalizer.unstandardize_std(posterior_states.std, std)

    y_true = normalizer.unstandardize(y_true, mean, std)
    std_y = normalizer.unstandardize_std(std_y, std)

    # plot results
    plt.figure(figsize=(10, 3))
    plt.plot(y_true, label=r"$y_{true}$", color="r")

    # Plot Prior
    # Epistemic
    plt.fill_between(
        np.arange(len(prior_states.mu.flatten())),
        prior_states.mu.flatten() - prior_states.std.flatten(),
        prior_states.mu.flatten() + prior_states.std.flatten(),
        color="b",
        alpha=0.3,
        edgecolor="none",
        label=r"$\mathbb{{E}}[Y']_{t\mid t-1} \pm \sigma_{epistemic}$",
    )
    # Aleatoric
    plt.fill_between(
        np.arange(len(prior_states.mu.flatten())),
        prior_states.mu.flatten() - prior_states.std.flatten() - std_y.flatten(),
        prior_states.mu.flatten() + prior_states.std.flatten() + std_y.flatten(),
        color="green",
        alpha=0.3,
        edgecolor="none",
        # label=r"$\mathbb{{E}}[Y']_{t\mid t-1} \pm \sigma_{total}$",
    )

    plt.plot(
        prior_states.mu.flatten(), label=r"$\mathbb{E}[Y']_{t\mid t-1}$", color="b"
    )

    # # Plot Posterior
    # # Epistemic
    # plt.fill_between(
    #     np.arange(len(posterior_states.mu.flatten())),
    #     posterior_states.mu.flatten() - posterior_states.std.flatten(),
    #     posterior_states.mu.flatten() + posterior_states.std.flatten(),
    #     color="g",
    #     alpha=0.3,
    #     label=r"$\mathbb{{E}}[Y']_{t\mid t} \pm \sigma_{epistemic}$",
    # )
    # # Aleatoric
    # plt.fill_between(
    #     np.arange(len(posterior_states.mu.flatten())),
    #     posterior_states.mu.flatten() - posterior_states.std.flatten() - std_y.flatten(),
    #     posterior_states.mu.flatten() + posterior_states.std.flatten() + std_y.flatten(),
    #     color="yellow",
    #     alpha=0.3,
    #     label=r"$\mathbb{{E}}[Y']_{t\mid t} \pm \sigma_{total}$",
    # )

    plt.plot(
        posterior_states.mu.flatten(), label=r"$\mathbb{E}[Y']_{t\mid t}$", color="k"
    )

    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "zeroshot_filtering.svg"),
        bbox_inches="tight",
    )
    plt.close()


prior_states, posterior_states, y_true, std_y = filter(local_net, data)

plot_results(
    prior_states, posterior_states, y_true, std_y, data, output_dir=output_dir + "local"
)

prior_states, posterior_states, y_true, std_y = filter(global_net, data)

plot_results(
    prior_states, posterior_states, y_true, std_y, data, output_dir=output_dir + "global"
)

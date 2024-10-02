import fire
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Setting the LaTeX parameters in matplotlib (optional)
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering in the plot
plt.rcParams['font.family'] = 'serif'  # Use a serif font like LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # (Optional) Load extra LaTeX packages
plt.rcParams['font.size'] = 14  # Set the global font size for the plot
cmap = plt.get_cmap("winter")  # tab10 has 10 distinct colors

import pytagi.metric as metric
from examples.data_loader import RegressionDataLoader
from examples.time_series_forecasting import PredictionViz
from pytagi import Normalizer
from pytagi.nn import Linear, OutputUpdater, ReLU, Sequential


def main(num_epochs: int = 10, batch_size: int = 20, sigma_v: float = 0.2):
    """Run training for the regression"""
    # Dataset
    x_train_file = "./data/toy_example/x_train_1D.csv"
    y_train_file = "./data/toy_example/y_train_1D.csv"

    train_dtl = RegressionDataLoader(x_file=x_train_file, y_file=y_train_file)

    # Network
    net = Sequential(
        Linear(1, 50),
        ReLU(),
        Linear(50, 1),
    )
    # net.to_device("cuda")
    # net.set_threads(8)

    out_updater = OutputUpdater(net.device)
    var_y = np.full((batch_size,), sigma_v ** 2, dtype=np.float32)

    # -------------------------------------------------------------------------#
    # Training
    mses = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    plt.figure(figsize=(12, 8))
    for epoch in pbar:
        if epoch == 0:
            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)
            var_x = np.full(batch_size, 0.1)
            x_mean, x_std = train_dtl.x_mean, train_dtl.x_std
            y_mean, y_std = train_dtl.y_mean, train_dtl.y_std
        else:
            train_dtl = RegressionDataLoader(x_file=f"./dw_out/x_train_1D_updated_{epoch}.csv",
                                             y_file=y_train_file, x_mean=x_mean, x_std=x_std, y_mean=y_mean,
                                             y_std=y_std)
            batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

        updated_input = []
        for x, y in batch_iter:
            color = cmap(epoch / num_epochs)  # Cycle through the colors if more than 10 epochs
            plt.scatter(x, y, color=color, label=f'Epoch {epoch + 1}')

            # Feed forward
            m_pred, _ = net(x, var_x)

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

            (delta_x, delta_var_x) = net.get_input_states()

            # Update the input states
            updated_x = x + delta_x * var_x
            updated_var_x = var_x + var_x * delta_var_x * var_x

            updated_input.extend(updated_x)

            # Training metric
            pred = Normalizer.unstandardize(m_pred, train_dtl.y_mean, train_dtl.y_std)
            obs = Normalizer.unstandardize(y, train_dtl.y_mean, train_dtl.y_std)
            mse = metric.mse(pred, obs)
            mses.append(mse)

        # Progress bar
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}| mse: {sum(mses) / len(mses):>7.2f}",
            refresh=True,
        )

        # save the input states
        pd.DataFrame(Normalizer.unstandardize(np.array(updated_input), x_mean, x_std)).to_csv(
            f"./dw_out/x_train_1D_updated_{epoch + 1}.csv", index=False, header=["x"]
        )

    # -------------------------------------------------------------------------#
    # Plotting
    plt.xlabel(r"$\text{x}$")
    plt.ylabel(r"$\text{y}$")
    plt.legend()
    plt.savefig("./dw_out/regression_embed.pdf")
    plt.show()

if __name__ == "__main__":
    fire.Fire(main)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# define a function to genertate a toy time series
def generate_sine(
    length: int = 240,
    noise_level: float = 0.0,
    amplitude: float = 1.0,
    period: float = 24.0,
):
    """
    Generate a toy time series with a sinusoidal pattern and some noise.

    Parameters:
    - length: int, the number of data points in the time series
    - noise_level: float, the standard deviation of the Gaussian noise added to the signal
    - period: float, the period of the sinusoidal wave

    Returns:
    - time: np.ndarray, the time points
    - signal: np.ndarray, the generated time series signal
    """
    time = np.arange(length)
    # Assume each point is one hour; 24 points = 1 day
    signal = np.sin(
        2 * np.pi * time / period
    ) * amplitude + noise_level * np.random.randn(length)

    return time, signal


def generate_square(
    length: int = 240,
    noise_level: float = 0.0,
    amplitude: float = 1.0,
    period: float = 24.0,
):
    """
    Generate a toy time series with a square wave pattern and some noise.

    Parameters:
    - length: int, the number of data points in the time series
    - noise_level: float, the standard deviation of the Gaussian noise added to the signal
    - period: float, the period of the square wave

    Returns:
    - time: np.ndarray, the time points
    - signal: np.ndarray, the generated time series signal
    """
    time = np.arange(length)
    # Generate a square wave with a period of 24 (hourly, 1 day)
    signal = np.sign(
        np.sin(2 * np.pi * time / period)
    ) * amplitude + noise_level * np.random.randn(length)

    return time, signal


def generate_traingular(
    length: int = 240,
    noise_level: float = 0.0,
    amplitude: float = 1.0,
    period: float = 24.0,
):
    """
    Generate a toy time series with a triangular wave pattern and some noise.

    Parameters:
    - length: int, the number of data points in the time series
    - noise_level: float, the standard deviation of the Gaussian noise added to the signal
    - period: float, the period of the triangular wave

    Returns:
    - time: np.ndarray, the time points
    - signal: np.ndarray, the generated time series signal
    """
    time = np.arange(length)
    # Generate a triangular wave with a period of 24 (hourly, 1 day)
    signal = (
        2 * np.abs(2 * (time / period - np.floor(time / period + 0.5))) * amplitude
        - 1
        + noise_level * np.random.randn(length)
    )

    return time, signal


def main(seed=42):

    np.random.seed(seed)  # For reproducibility

    # Define default parameters
    default_length = 240
    default_noise = 0.0
    base_period = 24.0

    # Wave types and modifiers
    wave_funcs = [
        ("sine", generate_sine),
        ("square", generate_square),
        ("triangular", generate_traingular),
    ]
    amp_mods = [1.0, 1.5, 0.5]
    per_mods = [1.0, 1.5, 0.5]

    # Prepare datetime index
    start_time = pd.Timestamp.now().floor("h")
    index = pd.date_range(start=start_time, periods=default_length, freq="h")

    # Generate all combinations
    data = {}
    for name, func in wave_funcs:
        for amp in amp_mods:
            for per_mod in per_mods:
                _, signal = func(
                    length=default_length,
                    noise_level=default_noise,
                    amplitude=amp,
                    period=base_period * per_mod,
                )
                col_name = f"{name}_amp{amp}_per{per_mod}"
                data[col_name] = signal

    # Build DataFrame and save to CSV
    df_values = pd.DataFrame(data, index=index)
    df_values.to_csv("time_series_values.csv", index=False)
    # ------------------------------------------------------------------
    # Create train / test splits (80 % train, 20 % test) and
    # save separate CSV files for values and datetimes
    # ------------------------------------------------------------------
    num_points     = len(df_values)
    test_size      = int(0.2 * num_points)
    input_seq_len  = 24  # one‑day look‑back

    # Turn the DateTimeIndex into a regular column for easier I/O
    df_full = df_values.reset_index().rename(columns={"index": "datetime"})

    # 80 % train  |  last 20 % (+look‑back) test
    train_df = df_full.iloc[:-test_size].reset_index(drop=True)
    test_df  = df_full.iloc[-test_size - input_seq_len :].reset_index(drop=True)

    # Column list excluding the datetime column
    value_cols = [c for c in df_full.columns if c != "datetime"]

    # ----- Write CSVs -----
    train_df[value_cols].to_csv("train_values.csv", index=False)
    train_df[["datetime"]].to_csv("train_datetimes.csv", index=False)

    test_df[value_cols].to_csv("test_values.csv", index=False)
    test_df[["datetime"]].to_csv("test_datetimes.csv", index=False)

    # Plot each time series variant in its own subplot
    n_cols = 3
    n_rows = int(np.ceil(len(df_values.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3), sharex=True)

    for ax, col in zip(axes.flatten(), df_values.columns):
        ax.plot(df_values.index, df_values[col], color="red", label=col)
        ax.legend()
        ax.set_ylabel("Value")

    # Turn off unused axes
    for ax in axes.flatten()[len(df_values.columns):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("time_series_plot_individual.png")
    plt.show()


if __name__ == "__main__":
    main()

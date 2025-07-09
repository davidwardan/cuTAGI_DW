import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# define a function to genertate a toy time series
def generate_sine(length: int = 480, noise_level: float = 0.0, amplitude: float = 1.0):
    """
    Generate a toy time series with a sinusoidal pattern and some noise.

    Parameters:
    - length: int, the number of data points in the time series
    - noise_level: float, the standard deviation of the Gaussian noise added to the signal

    Returns:
    - time: np.ndarray, the time points
    - signal: np.ndarray, the generated time series signal
    """
    time = np.arange(length)
    # Assume each point is one hour; 24 points = 1 day
    signal = np.sin(2 * np.pi * time / 24) * amplitude + noise_level * np.random.randn(
        length
    )

    return time, signal


def generate_square(
    length: int = 480, noise_level: float = 0.0, amplitude: float = 1.0
):
    """
    Generate a toy time series with a square wave pattern and some noise.

    Parameters:
    - length: int, the number of data points in the time series
    - noise_level: float, the standard deviation of the Gaussian noise added to the signal

    Returns:
    - time: np.ndarray, the time points
    - signal: np.ndarray, the generated time series signal
    """
    time = np.arange(length)
    # Generate a square wave with a period of 24 (hourly, 1 day)
    signal = np.sign(
        np.sin(2 * np.pi * time / 24)
    ) * amplitude + noise_level * np.random.randn(length)

    return time, signal


def generate_traingular(
    length: int = 480, noise_level: float = 0.0, apmplitude: float = 1.0
):
    """
    Generate a toy time series with a triangular wave pattern and some noise.

    Parameters:
    - length: int, the number of data points in the time series
    - noise_level: float, the standard deviation of the Gaussian noise added to the signal

    Returns:
    - time: np.ndarray, the time points
    - signal: np.ndarray, the generated time series signal
    """
    time = np.arange(length)
    # Generate a triangular wave with a period of 24 (hourly, 1 day)
    signal = (
        2 * np.abs(2 * (time / 24 - np.floor(time / 24 + 0.5))) * apmplitude
        - 1
        + noise_level * np.random.randn(length)
    )

    return time, signal


def main(seed=42):

    np.random.seed(seed)  # For reproducibility

    # Generate time series data
    time, sin_signal1 = generate_sine()
    _, square_signal1 = generate_square()

    # generate a varient of sin_signal1
    _, sin_signal2 = generate_sine(amplitude=1.5)
    _, square_signal2 = generate_square(amplitude=1.5)

    # Plot the generated time series
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(time, sin_signal1, label="Sine Wave 1", color="blue")
    axs[0].plot(time, sin_signal2, label="Sine Wave 2", color="red", linestyle="--")
    axs[0].set_title("Sine Wave Time Series")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[1].plot(time, square_signal1, label="Square Wave 1", color="orange")
    axs[1].plot(
        time, square_signal2, label="Square Wave    2", color="green", linestyle="--"
    )
    axs[1].set_title("Square Wave Time Series")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    plt.tight_layout()
    # Save time series values to CSV and split into train/test sets (80% train, 20% test)
    num_points = len(time)
    start_time = pd.Timestamp.now().floor("H")
    dates = pd.date_range(start=start_time, periods=num_points, freq="H")
    # Build DataFrame with datetime and signal values
    df = pd.DataFrame(
        {
            "datetime": dates,
            "sine1": sin_signal1,
            "sine2": sin_signal2,
            "square1": square_signal1,
            "square2": square_signal2,
        }
    )
    test_size = int(0.2 * num_points)
    input_seq_len = 24
    # Split DataFrame
    train_df = df.iloc[:-test_size].reset_index(drop=True)
    test_df = df.iloc[-test_size - input_seq_len :].reset_index(drop=True)
    # Save train set: values and datetime separately
    train_df[["sine1", "sine2", "square1", "square2"]].to_csv(
        "train_values.csv", index=False
    )
    train_df[["datetime"]].to_csv("train_datetimes.csv", index=False)
    # Save test set: values and datetime separately
    test_df[["sine1", "sine2", "square1", "square2"]].to_csv(
        "test_values.csv", index=False
    )
    test_df[["datetime"]].to_csv("test_datetimes.csv", index=False)
    plt.show()


if __name__ == "__main__":
    main()

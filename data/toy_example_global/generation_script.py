import pandas as pd
import numpy as np
import os
from scipy import signal

print("Starting time series generation (variable lengths)...")

# --- 1. Define Parameters ---

N_SERIES = 100  # Total number of time series
LOOKBACK = 24  # Overlap window for val/test sets
START_DATE = "2020-01-01"

# Define the MIN and MAX possible lengths for any given time series
MIN_SERIES_DAYS = 10
MAX_SERIES_DAYS = 40
MIN_SERIES_LENGTH = 24 * MIN_SERIES_DAYS
MAX_SERIES_LENGTH = 24 * MAX_SERIES_DAYS


def _generate_series_noise(num_steps: int, base_std: float) -> np.ndarray:
    """Create heteroscedastic, temporally correlated noise with rare spikes."""
    if num_steps <= 0:
        return np.empty(0, dtype=float)

    # Use an AR(1) filter for short-term correlation
    white_for_ar = np.random.normal(scale=base_std, size=num_steps + 4)
    correlated = signal.lfilter([1.0], [1.0, -0.65], white_for_ar)[-num_steps:]

    # Low-frequency drift component stabilised around zero
    drift = np.cumsum(np.random.normal(scale=base_std * 0.12, size=num_steps))
    drift -= drift.mean()

    # Occasional spikes emulate real measurement glitches
    spike_mask = np.random.rand(num_steps) < 0.03
    spikes = np.zeros(num_steps)
    if spike_mask.any():
        spikes[spike_mask] = np.random.normal(
            scale=base_std * 2.5, size=spike_mask.sum()
        )

    return correlated + drift + spikes


# --- 2. Generate Individual Time Series of Varying Lengths ---

print(f"Generating {N_SERIES} series with random lengths...")

all_series_values = []
for i in range(N_SERIES):
    # Determine a random length for the current series
    current_length = np.random.randint(MIN_SERIES_LENGTH, MAX_SERIES_LENGTH + 1)
    time_vector = np.arange(current_length)

    # Base properties
    mean_offset = np.random.uniform(-5, 5)
    noise_std = np.random.uniform(0.1, 0.4)

    # Start with base mean
    series = np.full(current_length, mean_offset)

    # Add 1 to 3 different wave components
    num_waves = np.random.randint(1, 4)
    for _ in range(num_waves):
        amplitude = np.random.uniform(2.0, 15.0)
        period = np.random.choice([6, 12, 24])
        phase_shift = np.random.uniform(0, 2 * np.pi)
        wave_type = np.random.choice(["sin", "cos", "triangle", "square"])
        wave_arg = 2 * np.pi * time_vector / period + phase_shift

        if wave_type == "sin":
            series += amplitude * np.sin(wave_arg)
        elif wave_type == "cos":
            series += amplitude * np.cos(wave_arg)
        elif wave_type == "triangle":
            series += amplitude * signal.sawtooth(wave_arg, width=0.5)
        elif wave_type == "square":
            series += amplitude * signal.square(wave_arg)

    # Add complex noise and append to our list
    series += _generate_series_noise(current_length, noise_std)
    all_series_values.append(series)

# --- 3. Pad Series and Create Master DataFrames ---

# Find the length of the longest series, which will be our canvas size
max_length = max(len(s) for s in all_series_values)
print(f"Padding all series to the maximum length of {max_length} steps.")

padded_series_map = {}
for i, series in enumerate(all_series_values):
    padding_size = max_length - len(series)
    # Pad with np.nan for the values
    padding = np.full(padding_size, np.nan)
    padded_series_map[f"ts_{i}"] = np.concatenate([series, padding])

# Create the master values DataFrame
values_df = pd.DataFrame(padded_series_map)

# Create the master dates DataFrame, using the full date range for all columns
master_dates = pd.date_range(start=START_DATE, periods=max_length, freq="h")
dates_df = pd.DataFrame({f"ts_{i}": master_dates for i in range(N_SERIES)})


# --- 4. Split Data into Train, Validation, and Test ---

# Define split points proportionally based on the max_length
split_train_idx = int(max_length * 0.7)
split_val_idx = int(max_length * 0.85)

# --- Train Set ---
train_values = values_df.iloc[:split_train_idx]
train_dates = dates_df.iloc[:split_train_idx]

# --- Validation Set ---
val_start_idx = split_train_idx - LOOKBACK
val_values = values_df.iloc[val_start_idx:split_val_idx]
val_dates = dates_df.iloc[val_start_idx:split_val_idx]

# --- Test Set ---
test_start_idx = split_val_idx - LOOKBACK
test_values = values_df.iloc[test_start_idx:]
test_dates = dates_df.iloc[test_start_idx:]

# --- 5. Save to CSV Files ---

output_dir = "data/toy_example_global"
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured output directory exists: {output_dir}")

files_to_save = {
    f"{output_dir}/toy_ts_train_values.csv": train_values,
    f"{output_dir}/toy_ts_train_dates.csv": train_dates,
    f"{output_dir}/toy_ts_val_values.csv": val_values,
    f"{output_dir}/toy_ts_val_dates.csv": val_dates,
    f"{output_dir}/toy_ts_test_values.csv": test_values,
    f"{output_dir}/toy_ts_test_dates.csv": test_dates,
}

# Save all files
for filename, df in files_to_save.items():
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with shape {df.shape}")

print("\n--- Summary ---")
print(
    f"Series lengths generated between {MIN_SERIES_LENGTH} and {MAX_SERIES_LENGTH} steps."
)
print(f"All DataFrames padded to max length: {max_length}")
print(f"Lookback window: {LOOKBACK}\n")

print(f"Train set shape: {train_values.shape}")
print(f"Validation set shape: {val_values.shape}")
print(f"Test set shape: {test_values.shape}")

print("\nAll files generated successfully.")
 
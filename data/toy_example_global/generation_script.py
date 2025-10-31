import os
import numpy as np
import pandas as pd
from scipy import signal

# ----------------------------
# 1) GENERATION PARAMETERS
# ----------------------------
N_SERIES = 8
MAX_DAYS = 60
MIN_DAYS_FRAC = 0.5  # Series will be 50% to 100% of MAX_DAYS
PERIOD = 24

# Percentage-based splits
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# TEST_FRAC will be the remaining ~0.15

START_DATETIME = np.datetime64("2020-01-01T00:00:00")

# Use the same categories as before
WAVES = ["sin", "cos", "square", "triangle"]
AMPLITUDES = [2.0, 1.0]
CATEGORY_DEFINITIONS = tuple(
    {"wave": w, "amplitude": a} for w in WAVES for a in AMPLITUDES
)

print(f"Generating {N_SERIES} series with individual splits...")

# ----------------------------
# 2) GENERATE & SPLIT SERIES INDIVIDUALLY
# ----------------------------

# Dictionaries to hold the separate data chunks
train_chunks = {}
val_chunks = {}
test_chunks = {}
train_date_chunks = {}
val_date_chunks = {}
test_date_chunks = {}

# We'll also collect metadata as we go
metadata_rows = []

np.random.seed(42)  # For reproducible random lengths


def gen_wave(wave: str, amplitude: float, arg: np.ndarray) -> np.ndarray:
    if wave == "sin":
        return amplitude * np.sin(arg)
    if wave == "cos":
        return amplitude * np.cos(arg)
    if wave == "triangle":
        return amplitude * signal.sawtooth(arg, width=0.5)
    return amplitude * signal.square(arg)


for i in range(N_SERIES):
    series_id = f"ts_{i}"
    category = CATEGORY_DEFINITIONS[i % len(CATEGORY_DEFINITIONS)]

    # --- A. Generate one full series of variable length ---
    current_days = np.random.randint(int(MAX_DAYS * MIN_DAYS_FRAC), MAX_DAYS + 1)
    current_len = current_days * 24

    t = np.arange(current_len)
    arg = 2 * np.pi * t / PERIOD

    y_full = gen_wave(category["wave"], category["amplitude"], arg)
    dates_full = START_DATETIME + np.arange(current_len) * np.timedelta64(1, "h")

    # --- B. Calculate split points *for this specific series* ---
    train_end = int(current_len * TRAIN_FRAC)
    val_end = int(current_len * (TRAIN_FRAC + VAL_FRAC))

    # --- C. Slice the series into its chunks ---
    y_train = y_full[:train_end]
    y_val = y_full[train_end:val_end]
    y_test = y_full[val_end:]
    train_dates = dates_full[:train_end]
    val_dates = dates_full[train_end:val_end]
    test_dates = dates_full[val_end:]

    # --- D. Add the chunks to their respective dictionaries ---
    train_chunks[series_id] = y_train
    val_chunks[series_id] = y_val
    test_chunks[series_id] = y_test
    train_date_chunks[series_id] = train_dates
    val_date_chunks[series_id] = val_dates
    test_date_chunks[series_id] = test_dates

    # --- E. Save metadata ---
    metadata_rows.append(
        {
            "series_id": series_id,
            "wave": category["wave"],
            "amplitude": category["amplitude"],
            "total_length": current_len,
            "train_length": len(y_train),
            "val_length": len(y_val),
            "test_length": len(y_test),
        }
    )

print("Generation and in-memory splitting complete.")

# ----------------------------
# 3) CREATE DATAFRAMES (THE AUTOMATIC PADDING)
# ----------------------------

def to_padded_dataframe(chunks: dict[str, np.ndarray], dtype=None, fill_value=None) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()

    columns = list(chunks.keys())
    max_len = max(len(values) for values in chunks.values())
    if max_len == 0:
        return pd.DataFrame(columns=columns)

    if dtype is None:
        sample_values = next(iter(chunks.values()))
        dtype = np.asarray(sample_values).dtype

    np_dtype = np.dtype(dtype)

    if fill_value is None:
        if np.issubdtype(np_dtype, np.floating):
            fill_value = np.nan
        elif np.issubdtype(np_dtype, np.datetime64):
            fill_value = np.datetime64("NaT")
        else:
            fill_value = None
            np_dtype = np.dtype(object)

    data = np.full((max_len, len(columns)), fill_value, dtype=np_dtype)
    for idx, column in enumerate(columns):
        values = np.asarray(chunks[column], dtype=np_dtype)
        length = values.shape[0]
        if length:
            data[:length, idx] = values

    return pd.DataFrame(data, columns=columns)

print("Creating DataFrames... pandas will now add NaN padding.")

train_values_df = to_padded_dataframe(train_chunks, dtype=np.float64, fill_value=np.nan)
val_values_df = to_padded_dataframe(val_chunks, dtype=np.float64, fill_value=np.nan)
test_values_df = to_padded_dataframe(test_chunks, dtype=np.float64, fill_value=np.nan)
train_dates_df = to_padded_dataframe(train_date_chunks, dtype="datetime64[ns]")
val_dates_df = to_padded_dataframe(val_date_chunks, dtype="datetime64[ns]")
test_dates_df = to_padded_dataframe(test_date_chunks, dtype="datetime64[ns]")

metadata_df = pd.DataFrame(metadata_rows)

# ----------------------------
# 4) SAVE TO CSV
# ----------------------------
out_dir = "data/toy_example_individual_splits"
os.makedirs(out_dir, exist_ok=True)

to_save = {
    f"{out_dir}/toy_ts_train_values.csv": train_values_df,
    f"{out_dir}/toy_ts_train_dates.csv": train_dates_df,
    f"{out_dir}/toy_ts_val_values.csv": val_values_df,
    f"{out_dir}/toy_ts_val_dates.csv": val_dates_df,
    f"{out_dir}/toy_ts_test_values.csv": test_values_df,
    f"{out_dir}/toy_ts_test_dates.csv": test_dates_df,
    f"{out_dir}/toy_ts_metadata.csv": metadata_df,
}

for fn, df in to_save.items():
    df.to_csv(fn, index=False)
    print(f"Saved {fn}  shape={df.shape}")

# ----------------------------
# 5) SUMMARY
# ----------------------------
print("\n--- Summary (from metadata) ---")
print(metadata_df.to_string(index=False))

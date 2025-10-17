import numpy as np
import pandas as pd
import os
from examples.data_loader import (
    GlobalTimeSeriesDataloader,
)  # Make sure data_loader.py is accessible

# --- Create Dummy Data Files ---
# Create dummy values file
values_data = {
    "series_A": [10, 11, 12, 13, 14, np.nan], # Length becomes 5 after trimming
    "series_B": [100, 101, 102, 103, 104, 105],  # Length is 6
    "series_C": [1000, 1001, 1002, 1003, 1004, np.nan], # Length becomes 5 after trimming
}
# Write to CSV with a header, which the loader will skip
pd.DataFrame(values_data).to_csv("dummy_values.csv", index=False)

# Create dummy dates file
dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=len(values_data["series_A"])))
dates_data = {"series_A": dates, "series_B": dates, "series_C": dates}
# Write to CSV with a header
pd.DataFrame(dates_data).to_csv("dummy_dates.csv", index=False)


print("--- Created dummy data files ---")

# --- Configuration for the Test ---
INPUT_SEQ_LEN = 4
OUTPUT_SEQ_LEN = 1
NUM_FEATURES = 2  # 1 (target) + 1 (week_of_year covariate)
BATCH_SIZE = 3
NUM_SERIES = 3

# Calculated input size for the model (based on your code)
# This is the expected shape of Xb
expected_input_size = INPUT_SEQ_LEN + NUM_FEATURES - 1
print(f"Expected model input size: {expected_input_size}\n")


# --- Function to Test a Dataloader ---
def inspect_dataloader(mode):
    print(f"--- Testing order_mode = '{mode}' ---")

    # Instantiate the dataloader
    dtl = GlobalTimeSeriesDataloader(
        x_file="dummy_values.csv",
        date_time_file="dummy_dates.csv",
        output_col=[0],
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        num_features=NUM_FEATURES,
        stride=1,
        time_covariates=["week_of_year"],
        keep_last_time_cov=True,
        scale_method="standard",
        order_mode=mode,
        ts_to_use=[0, 1, 2],
    )

    # Create a generator
    batch_generator = dtl.create_data_loader(
        batch_size=BATCH_SIZE, shuffle=False, include_ids=True  # Crucial for debugging!
    )

    print("Fetching first 2 batches...")
    for i, (Xb, Yb, Sb, Kb) in enumerate(batch_generator):
        if i >= 2:
            break
        print(f"\nBatch {i+1}:")
        print(f"  Xb shape: {Xb.shape}")
        # Validate the shape
        assert Xb.shape[1] == expected_input_size, "Shape mismatch!"
        print(f"  Yb shape: {Yb.shape}")
        # These are the IDs of the time series in the batch (0=A, 1=B, 2=C)
        print(f"  Series IDs (Sb): {Sb}")
        # These are the window indices *within* each time series
        print(f"  Window IDs (Kb): {Kb}")


# --- Run the Inspection ---
inspect_dataloader(mode="by_window")
inspect_dataloader(mode="by_series")


# --- Clean up dummy files ---
os.remove("dummy_values.csv")
os.remove("dummy_dates.csv")

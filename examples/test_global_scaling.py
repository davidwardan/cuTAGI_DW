import numpy as np
import pandas as pd
import os
from examples.data_loader import GlobalTimeSeriesDataloader

# --- Create Dummy Data Files ---
values_data = {
    "series_A": [10, 11, 12, 13, 14, 15, 16, 17],
    "series_B": [100, 101, 102, 103, 104, 105, 106, 107],
    "series_C": [
        2000,
        2010,
        2020,
        2030,
        2040,
        2050,
        2060,
        2070,
    ],  # Changed for varied std
}
pd.DataFrame(values_data).to_csv("dummy_values.csv", index=False)

dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=8))
dates_data = {"series_A": dates, "series_B": dates, "series_C": dates}
pd.DataFrame(dates_data).to_csv("dummy_dates.csv", index=False)

print("--- Created dummy data files ---")

# --- 1. Manual Calculation Step ---
print("\n--- Manually Calculating Expected Scaled Values ---")

# Load raw data
raw_values_df = pd.read_csv("dummy_values.csv")
raw_dates_df = pd.read_csv("dummy_dates.csv")
num_series = raw_values_df.shape[1]
num_timesteps = raw_values_df.shape[0]

# Manually calculate per-series mean and std for the TARGET
target_means = raw_values_df.mean().values
target_stds = raw_values_df.std(
    ddof=0
).values  # Use ddof=0 for population std like numpy
print(f"Target Per-Series Means (μ): {np.round(target_means, 2)}")
print(f"Target Per-Series Stds (σ): {np.round(target_stds, 2)}")


# Manually create and calculate global mean/std for the COVARIATES
all_covariates = []
for col in raw_dates_df.columns:
    # --- THIS IS THE CORRECTED LINE ---
    weeks = pd.to_datetime(raw_dates_df[col]).dt.isocalendar().week.astype(float).values
    all_covariates.append(weeks)

# Stack them to calculate global stats
covariates_matrix = np.stack(all_covariates, axis=1)  # Shape (T, N)
covariate_mean = np.mean(covariates_matrix)
covariate_std = np.std(covariates_matrix)
print(f"Global Covariate Mean (μ): {np.round(covariate_mean, 2)}")
print(f"Global Covariate Std (σ): {np.round(covariate_std, 2)}")


# Manually scale everything to create our ground truth
expected_scaled_matrix = np.zeros((num_timesteps, num_series, 2))  # T x N x Features

# Scale targets (feature 0)
for i in range(num_series):
    expected_scaled_matrix[:, i, 0] = (
        raw_values_df.iloc[:, i] - target_means[i]
    ) / target_stds[i]

# Scale covariates (feature 1)
# Handle division by zero if std is 0
if covariate_std > 0:
    expected_scaled_matrix[:, :, 1] = (
        covariates_matrix - covariate_mean
    ) / covariate_std
else:
    expected_scaled_matrix[:, :, 1] = 0.0


print("\nManually scaled first 3 steps of Series A (Target, Covariate):")
print(np.round(expected_scaled_matrix[:3, 0, :], 4))


# --- 2. Dataloader Instantiation and Verification ---
print("\n--- Verifying Dataloader Output ---")

INPUT_SEQ_LEN = 4
dtl = GlobalTimeSeriesDataloader(
    x_file="dummy_values.csv",
    date_time_file="dummy_dates.csv",
    output_col=[0],
    input_seq_len=INPUT_SEQ_LEN,
    output_seq_len=1,
    num_features=2,
    stride=1,
    time_covariates=["week_of_year"],
    keep_last_time_cov=True,
    scale_method="standard",
    order_mode="by_series",  # Easier to reconstruct one series at a time
    ts_to_use=list(range(num_series)),
)

# Fetch all data from the generator
batch_generator = dtl.create_data_loader(
    batch_size=100, shuffle=False, include_ids=True
)
X_all, Y_all, S_all, K_all = next(iter(batch_generator))

# Reconstruct the first series (series 0) from the dataloader's output
series_0_indices = np.where(S_all == 0)[0]
X_series_0 = X_all[series_0_indices]

# The dataloader output Xb has shape (L + F - 1)
# -> [target_t0, target_t1, ..., target_tL-1, cov_tL-1]
# We can reconstruct the scaled target values directly
reconstructed_target_series_0 = X_series_0[0, :INPUT_SEQ_LEN]
# For subsequent windows, we just append the last value
for i in range(1, len(X_series_0)):
    reconstructed_target_series_0 = np.append(
        reconstructed_target_series_0, X_series_0[i, INPUT_SEQ_LEN - 1]
    )


# --- 3. Assert and Verify ---
try:
    # Compare the reconstructed target values with our manually scaled ones
    np.testing.assert_allclose(
        reconstructed_target_series_0,
        expected_scaled_matrix[
            : len(reconstructed_target_series_0), 0, 0
        ],  # Target is feature 0
        rtol=1e-6,
        err_msg="Target scaling does NOT match!",
    )
    print("✅ SUCCESS: Target variable scaling is correct.")

    # Let's also check the covariate scaling for the last step of the first window
    reconstructed_cov_val = X_series_0[
        0, -1
    ]  # Last element is the last step's covariate
    expected_cov_val = expected_scaled_matrix[
        INPUT_SEQ_LEN - 1, 0, 1
    ]  # Covariate is feature 1

    np.testing.assert_allclose(
        reconstructed_cov_val,
        expected_cov_val,
        rtol=1e-6,
        err_msg="Covariate scaling does NOT match!",
    )
    print("✅ SUCCESS: Covariate scaling is correct.")

except AssertionError as e:
    print(f"❌ FAILURE: {e}")

# --- Clean up dummy files ---
os.remove("dummy_values.csv")
os.remove("dummy_dates.csv")

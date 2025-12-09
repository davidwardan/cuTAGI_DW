import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, studentized_range


def perform_mcb_test(error_df, alpha=0.05, title="MCB Test"):
    """
    Performs Hsu's MCB test and plots the results.

    Args:
        error_df: DataFrame where Index = Blocks (e.g., Series or Time)
                  and Columns = Models. Values are errors (lower is better).
        alpha: Significance level (default 0.05).
        title: Plot title.
    """
    n_blocks = len(error_df)
    k_models = len(error_df.columns)

    # 1. Calculate Ranks (Row-wise)
    # We rank the errors for each block (1 = best/lowest error)
    ranks = error_df.rank(axis=1)

    # 2. Calculate Mean Rank for each model
    mean_ranks = ranks.mean()

    # 3. Calculate Critical Distance
    # We use the Studentized Range distribution (q)
    # df = degrees of freedom = (n_blocks - 1) * (k_models - 1) is standard for Friedman-like
    # but for simple MCB often infinite df is approximated or N-1 used.
    # The standard formula for MCB Critical Distance is:
    # CD = q_alpha * sqrt( k(k+1) / (6*n) )

    q_val = studentized_range.ppf(1 - alpha, k_models, np.inf)
    critical_distance = q_val * np.sqrt((k_models * (k_models + 1)) / (6 * n_blocks))

    print(f"--- Results for {title} ---")
    print(f"Critical Distance: {critical_distance:.4f}")

    # 4. Identify the Best Model (Lowest Mean Rank)
    best_model = mean_ranks.idxmin()
    best_rank = mean_ranks.min()

    # Identify indistinguishable models (within CD of the best)
    threshold = best_rank + critical_distance
    in_group = mean_ranks[mean_ranks <= threshold].index.tolist()
    print(f"Best Model: {best_model}")
    print(f"Statistically Indistinguishable group: {in_group}\n")

    # 5. Plotting
    plt.figure(figsize=(10, 6))

    # Sort models by rank for cleaner plotting
    sorted_ranks = mean_ranks.sort_values()

    # Plot points
    plt.scatter(sorted_ranks.index, sorted_ranks.values, color="black", zorder=3)

    # Plot MCB intervals
    # The interval is [rank - 0.5*CD, rank + 0.5*CD] usually for visual overlap,
    # but strictly for MCB against the best, we often just visualize the CD bar
    # relative to the best. A common visualization is plotting the mean rank
    # with a one-sided bar or a centered interval of length CD.
    # Here we plot simple error bars representing the Critical Distance.
    plt.errorbar(
        sorted_ranks.index,
        sorted_ranks.values,
        yerr=critical_distance / 2,
        fmt="none",
        capsize=5,
        color="gray",
    )

    # Highlight the "Best" zone
    plt.axhline(
        y=best_rank + critical_distance,
        color="r",
        linestyle="--",
        label=f"Significance Threshold ({alpha})",
    )
    plt.axhline(y=best_rank, color="g", linestyle="--", alpha=0.5, label="Best Rank")

    # Fill the "Winning Zone"
    plt.fill_between(
        range(len(sorted_ranks)),
        best_rank,
        best_rank + critical_distance,
        color="green",
        alpha=0.1,
    )

    plt.title(f"{title}\n(Lower Rank is Better)")
    plt.ylabel("Mean Rank")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()


# ==========================================
# 1. Generate Dummy Traffic Data
# ==========================================
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
sensors = [f"Sensor_{i}" for i in range(1, 21)]  # 20 Sensors
models = ["LSTM", "ARIMA", "XGBoost", "Transformer"]

data_rows = []
for s in sensors:
    for d in dates:
        # Simulate errors: XGBoost is generally best (lowest error), ARIMA worst
        base_err = np.random.uniform(10, 50)
        errs = {
            "Series_ID": s,
            "Date": d,
            "LSTM": base_err + np.random.normal(0, 5),
            "ARIMA": base_err + np.random.normal(5, 5),  # Worse
            "XGBoost": base_err + np.random.normal(-2, 4),  # Best
            "Transformer": base_err + np.random.normal(0, 6),
        }
        data_rows.append(errs)

df_traffic = pd.DataFrame(data_rows)

# ==========================================
# 2. Case A: Overall Performance (Across Sensors)
# ==========================================
# Logic: We average the error for each sensor first to create "blocks"
# Rows = Sensors, Columns = Models
overall_pivot = df_traffic.groupby("Series_ID")[models].mean()

perform_mcb_test(overall_pivot, title="Overall Performance (Aggregated by Sensor)")

# ==========================================
# 3. Case B: Per-Series Performance (Temporal Blocks)
# ==========================================
# Logic: We pick ONE sensor and treat time intervals (Dates) as "blocks"
# Rows = Dates, Columns = Models

target_sensor = "Sensor_5"
series_pivot = df_traffic[df_traffic["Series_ID"] == target_sensor].set_index("Date")[
    models
]

perform_mcb_test(series_pivot, title=f"Per-Series Performance ({target_sensor})")

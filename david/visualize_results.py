import numpy as np
import pandas as pd

# from examples.data_loader import TimeSeriesDataloader
from examples.data_loader import GlobalTimeSeriesDataloader
from examples.time_series_forecasting import PredictionViz
from matplotlib import pyplot as plt

output_col = [0]
num_features = 3
input_seq_len = 24
output_seq_len = 1
seq_stride = 1
idx = 14
#
file_dir = '/Users/davidwardan/PycharmProjects/cuTAGI_DW/david/output/electricity_11_64_0.43_40_method2_std/'


# test_dtl = GlobalTimeSeriesDataloader(
#     x_file="data/traffic/traffic_2008_01_14_test.csv",
#     date_time_file="data/traffic/traffic_2008_01_14_test_datetime.csv",
#     output_col=output_col,
#     input_seq_len=input_seq_len,
#     output_seq_len=output_seq_len,
#     num_features=num_features,
#     stride=seq_stride,
#     ts_idx=idx,
#     time_covariates=['hour_of_day', 'day_of_week'],
# )

test_dtl = GlobalTimeSeriesDataloader(
    x_file="data/electricity/electricity_2014_03_31_test.csv",
    date_time_file="data/electricity/electricity_2014_03_31_test_datetime.csv",
    output_col=output_col,
    input_seq_len=input_seq_len,
    output_seq_len=output_seq_len,
    num_features=num_features,
    stride=seq_stride,
    ts_idx=idx,
    time_covariates=['hour_of_day', 'day_of_week'],
)

# load the predictions into a pandas dataframe
mu_pred = pd.read_csv(file_dir + "electricity_2014_03_31_ytestPd_pyTAGI.csv", header=None)
var_pred = pd.read_csv(file_dir + "electricity_2014_03_31_SytestPd_pyTAGI.csv", header=None)
y_test = pd.read_csv(file_dir + "electricity_2014_03_31_ytestTr_pyTAGI.csv", header=None)
#
# mu_pred = pd.read_csv(file_dir + "traffic_2008_01_14_ytestPd_pyTAGI.csv", header=None)
# var_pred = pd.read_csv(file_dir + "traffic_2008_01_14_SytestPd_pyTAGI.csv", header=None)
# y_test = pd.read_csv(file_dir + "traffic_2008_01_14_ytestTr_pyTAGI.csv", header=None)

# get the true and predicted values for the idx-th time series
mu_preds = mu_pred.iloc[:, idx].values
std_preds = var_pred.iloc[:, idx].values ** 0.5
y_test = y_test.iloc[:, idx].values

# create the prediction visualization object
# Viz
viz = PredictionViz(task_name="Global LSTM", data_name="Electricity")

# img_title = 'TS #{}. Traffic_2008_01_14 Global Model'.format(
#     idx + 1)  #r"$\text{Time Series Forecasting}$" + " " + str(idx)

img_title = 'TS #{}. Electricity_2014_03_31 Global Model'.format(
    idx + 1)  #r"$\text{Time Series Forecasting}$" + " " + str(idx)

# Visualization
viz.plot_predictions(
    x_test=test_dtl.dataset["date_time"][: len(y_test)],
    y_test=y_test,
    y_pred=mu_preds,
    sy_pred=std_preds,
    std_factor=1,
    label="{}".format(idx+1),
    title=img_title,
    time_series=True,
    )

# plt.savefig(file_dir + "electricity_ts{}.png".format(idx+1))
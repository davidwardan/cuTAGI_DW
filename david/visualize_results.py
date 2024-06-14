import numpy as np
import pandas as pd

# from examples.data_loader import TimeSeriesDataloader
from examples.data_loader import GlobalTimeSeriesDataloader
from examples.time_series_forecasting import PredictionViz

output_col = [0]
num_features = 3
input_seq_len = 24
output_seq_len = 1
seq_stride = 1
idx = 14
#
file_dir = '/Users/davidwardan/PycharmProjects/cuTAGI_DW/david/output/traffic_24_32_0.01_100/'
#
#
# load the data into a pandas dataframe
train_dtl = GlobalTimeSeriesDataloader(
    x_file="data/traffic/traffic_2008_01_14_train.csv",
    date_time_file="data/traffic/traffic_2008_01_14_train_datetime.csv",
    output_col=output_col,
    input_seq_len=input_seq_len,
    output_seq_len=output_seq_len,
    num_features=num_features,
    stride=seq_stride,
    ts_idx=idx,
)
#
test_dtl = GlobalTimeSeriesDataloader(
    x_file="data/traffic/traffic_2008_01_14_test.csv",
    date_time_file="data/traffic/traffic_2008_01_14_test_datetime.csv",
    output_col=output_col,
    input_seq_len=input_seq_len,
    output_seq_len=output_seq_len,
    num_features=num_features,
    stride=seq_stride,
    x_mean=train_dtl.x_mean,
    x_std=train_dtl.x_std,
    ts_idx=idx,
)

# load the predictions into a pandas dataframe
mu_pred = pd.read_csv(file_dir + "traffic_2008_01_14_ytestPd_pyTAGI.csv", header=None)
var_pred = pd.read_csv(file_dir + "traffic_2008_01_14_SytestPd_pyTAGI.csv", header=None)
y_test = pd.read_csv(file_dir + "traffic_2008_01_14_ytestTr_pyTAGI.csv", header=None)


# get the true and predicted values for the idx-th time series
mu_preds = mu_pred.iloc[:, idx].values
std_preds = var_pred.iloc[:, idx].values ** 0.5
y_test = y_test.iloc[:, idx].values

# create the prediction visualization object
# Viz
viz = PredictionViz(task_name="forecasting", data_name="sin_signal")

img_title = 'TS #{}. Traffic_2008_01_14 Global Model'.format(idx+1)#r"$\text{Time Series Forecasting}$" + " " + str(idx)

# Visualization
viz.plot_predictions(
    x_test=test_dtl.dataset["date_time"][: len(y_test)],
    y_test=y_test,
    y_pred=mu_preds,
    sy_pred=std_preds,
    std_factor=1,
    label="time_series_forecasting",
    title=img_title,
    time_series=True,
)
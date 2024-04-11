# import libraries
import pandas as pd
import numpy as np
import datetime

# read data from the csv file
df_traffic = pd.read_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/traffic.csv")
df_elec = pd.read_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/electricity.csv")

# get temporal data
def get_time_series(df, start_time):
    time_series = pd.date_range(start_time, periods=df.shape[0], freq='H')
    return time_series

# get the time series for the traffic data
start_time = datetime.datetime(2024, 12, 27, 0)

time_series_traffic = get_time_series(df_traffic, start_time)
time_series_elec = get_time_series(df_elec, start_time)

# add the temporal data to the last column fo the data frame
df_traffic['time'] = time_series_traffic
df_elec['time'] = time_series_elec

# save the data to a csv file
# df_traffic.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/traffic.csv", index=False)
# df_elec.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/electricity.csv", index=False)

# split the data into training and testing data
def split_data(df, split_ratio):
    split_index = int(df.shape[0] * split_ratio)
    train_data = df.iloc[:split_index, :]
    test_data = df.iloc[split_index:, :]
    return train_data, test_data

train_traffic, test_traffic = split_data(df_traffic, 0.8)

train_elec, test_elec = split_data(df_elec, 0.8)

# save the data to a csv file
train_traffic.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/train_traffic.csv", index=False)
test_traffic.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/test_traffic.csv", index=False)
train_elec.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/train_elec.csv", index=False)
test_elec.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/test_elec.csv", index=False)
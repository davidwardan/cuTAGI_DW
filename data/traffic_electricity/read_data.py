# import libraries
import pandas as pd
import datetime
import numpy as np

# read traffic data
traffic_data = np.load("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/electricity.npy")

# read electricity data
elec_data = np.load("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/traffic.npy")

# read the data into a pandas dataframe
#%%
df_traffic = pd.DataFrame(traffic_data)

# read the data into a pandas dataframe
df_elec = pd.DataFrame(elec_data)


#%%
# print shape of the data
print(df_traffic.shape)
print(df_elec.shape)

#%%
# name all the columns TS# where # is the column number
df_traffic.columns = ['TS'+str(i) for i in range(df_traffic.shape[1])]
df_elec.columns = ['TS'+str(i) for i in range(df_elec.shape[1])]

#%%
# print the head of the data
print(df_traffic.head())
print(df_elec.head())

#%%
# specify the first hour of the measurements
def get_time_series(df, start_time):
    time_series = pd.date_range(start_time, periods=df.shape[0], freq='H')
    return time_series

# get the time series for the traffic data
start_time = datetime.datetime(2024, 12, 27, 0)

time_series_traffic = get_time_series(df_traffic, start_time)
time_series_elec = get_time_series(df_elec, start_time)


#%%
# add the time series to the data
df_traffic['time'] = time_series_traffic
df_elec['time'] = time_series_elec

#%%
# print the head of the data
print(df_traffic.head(10))
print(df_elec.head(10))

#%%
# save the data to a csv file
df_traffic.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/traffic.csv", index=False)
df_elec.to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/electricity.csv", index=False)

#%%
## format the data for the model
# split the data into 80% training and 20% testing data
train_traffic = df_traffic.iloc[:int(df_traffic.shape[0]*0.8)]
test_traffic = df_traffic.iloc[int(df_traffic.shape[0]*0.8):]

train_elec = df_elec.iloc[:int(df_elec.shape[0]*0.8)]
test_elec = df_elec.iloc[int(df_elec.shape[0]*0.8):]

#%%
# print the shape of the data
print('Shape of the traffic training data: {}'.format(train_traffic.shape))
print('Shape of the traffic testing data: {}'.format(test_traffic.shape))

print('Shape of the electricity training data: {}'.format(train_elec.shape))
print('Shape of the electricity testing data: {}'.format(test_elec.shape))

#%%
# save the first time series to a csv file
train_traffic['TS0'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/Xtrain_traffic.csv", index=False)
test_traffic['TS0'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/Xtest_traffic.csv", index=False)
train_traffic['time'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/train_time_traffic.csv", index=False)
test_traffic['time'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/test_time_traffic.csv", index=False)

train_elec['TS0'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/Xtrain_elec.csv", index=False)
test_elec['TS0'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/Xtest_elec.csv", index=False)
train_elec['time'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/train_time_elec.csv", index=False)
test_elec['time'].to_csv("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/test_time_elec.csv", index=False)


#%%
# plot the first time series showing the training and testing data
import matplotlib.pyplot as plt
plt.figure(figsize=(40,10))
plt.plot(train_traffic['time'], train_traffic['TS0'], label='Training Data', color='steelblue')
plt.plot(test_traffic['time'], test_traffic['TS0'], label='Testing Data', color='seagreen')
plt.xlabel('Time', fontsize=20)
plt.ylabel('Traffic', fontsize=20)
plt.title('Traffic Data', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/traffic_data.png")

plt.figure(figsize=(40,10))
plt.plot(train_elec['time'], train_elec['TS0'], label='Training Data', color='steelblue')
plt.plot(test_elec['time'], test_elec['TS0'], label='Testing Data', color='seagreen')
plt.xlabel('Time', fontsize=20)
plt.ylabel('Electricity', fontsize=20)
plt.title('Electricity Data', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/electricity_data.png")

# plot the first time series without showing the training and testing data
plt.figure(figsize=(40,10))
plt.plot(df_traffic['time'], df_traffic['TS0'], color='steelblue')
plt.xlabel('Time', fontsize=20)
plt.ylabel('Traffic', fontsize=20)
plt.title('Traffic Data', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/traffic_data_all.png")

plt.figure(figsize=(40,10))
plt.plot(df_elec['time'], df_elec['TS0'], color='steelblue')
plt.xlabel('Time', fontsize=20)
plt.ylabel('Electricity', fontsize=20)
plt.title('Electricity Data', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("/Users/davidwardan/PycharmProjects/cuTAGI_DW/data/traffic_electricity/electricity_data_all.png")

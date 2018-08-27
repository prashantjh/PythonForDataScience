# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:23:58 2017

@author: prashant
"""
#https://github.com/christsaizyt/US-Stock-Market-Prediction-by-LSTM

import quandl
import matplotlib.pyplot as plt

my_data = quandl.get("NSE/CNX_NIFTY")

# Explore the data #
my_data.describe()

my_data.head()
my_data.tail()

my_data.index

# Dimension of dataset
my_data.shape
# - 6611 X 6

# Column names:
my_data.dtypes

# Rename shares traded column
my_data.rename(columns={'Shares Traded':'volume'}, inplace = True)

# Filtering one year data & viewing it:
data_2016 = my_data.ix['2016-01-01':'2016-12-31']
data_2016.head()
data_2016.tail()

#==============================================================================
# Field available :
#  - OHLC
#  - Shares Tradded
#  - Turnover (Rs. Cr.)
#New Columns to Add:
#    - Previous Close
#    - Percentage Change
#    - Day of Week (1 - Monday, 7 - Sunday)
#    - Difference b/w open date and prev. close date (1, 2, etc.)
#==============================================================================

# Adding a date column same as index column
my_data['datetime'] = my_data.index

# Adding Previous close & Percentage Change column
my_data['prevClose'] = my_data.Close.shift(1)
my_data['perChange'] = 100*(my_data.Close - my_data.prevClose)/my_data.prevClose

# Adding Day, Month Column
my_data['day'] = my_data.index.day
my_data['month'] = my_data.index.month
my_data['year'] = my_data.index.year

# Day of week; 1 - Monday, 7  - Sunday
my_data['dayOfWeek'] = my_data['datetime'].dt.dayofweek
my_data['dayOfWeek'] = my_data['dayOfWeek'] + 1

# Next day opening price
my_data['nextOpen'] = my_data.Open.shift(-1)


## Plotting the columns ##
col_groups = [0, 1, 2, 3, 4, 5]
i = 1
values = my_data.values

# Plot each column:
plt.figure()
for group in col_groups:
    plt.subplot(len(col_groups), 1, i)
    plt.plot(values[:, group])
    plt.title(my_data.columns[group], y = 0.5, loc = 'left')
    i += 1
plt.show()


#==============================================================================
# Data Selection for Training & Testing
#==============================================================================
train = my_data.ix['2015-01-01':'2017-08-31']
test = my_data.ix['2017-09-01':'2017-09-30']

# Missing value imputation with the value of the previous one
train = train.fillna(method = 'pad')
train = train.fillna(method = 'bfill')
test = test.fillna(method = 'pad')
test = test.fillna(method = 'bfill')

# Normalized Features
scaler = MinMaxScaler(feature_range=(0, 1))

train_values = train.values[:, [0,1,2,3,4,5, -1]]
train_values = train_values.astype('float')
scaled_train = scaler.fit_transform(train_values)

test_values = test.values[:, [0,1,2,3,4,5, -1]]
test_values = test_values.astype('float')
scaled_test = scaler.fit_transform(test_values)


# Split into input and outputs
train_X, train_Y = scaled_train[:, [0,1,2,3,4,5]], scaled_train[:, [-1]]
test_X, test_Y = scaled_test[:, [0,1,2,3,4,5]], scaled_test[:, [-1]]



# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


#==============================================================================
# LSTM Network
#==============================================================================
import numpy as np
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# Designing Network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer = 'adam')

# Fit Network
history = model.fit(train_X, train_Y, epochs = 50, batch_size=72, validation_data=(test_X, test_Y), verbose = True, shuffle = False)
                    
# Plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'test')
pyplot.legend()
pyplot.show()


## Evaluate the Model
yhat = model.predict(test_X)

# Invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# Invert scaling for actual
test_Y = test_Y.reshape((len(test_Y), 1))
inv_y = concatenate((test_Y, test_X[:, 0:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]



# RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print(rmse)

pyplot.plot(inv_y, label = 'actual')
pyplot.plot(inv_yhat, label = 'model')
pyplot.legend()
pyplot.show()


#==============================================================================
# Predicting for Next 3 Months
#==============================================================================
test2 = my_data.ix['2017-10-01':'2017-10-31']

# Missing value imputation with the value of the previous one
test2 = test2.fillna(method = 'pad')
test2 = test2.fillna(method = 'bfill')

# Normalized Features
#scaler = MinMaxScaler(feature_range=(0, 1))

test_values = test2.values[:, [0,1,2,3,4,5, -1]]
test_values = test_values.astype('float')
scaled_test = scaler.fit_transform(test_values)

# Split into input and outputs
test2_X, test2_Y = scaled_test[:, [0,1,2,3,4,5]], scaled_test[:, [-1]]

# Reshape input to be 3D [samples, timesteps, features]
test2_X = test2_X.reshape((test2_X.shape[0], 1, test2_X.shape[1]))

## New Data Prediction
yhat = model.predict(test2_X)

# Invert scaling for forecast
test2_X = test2_X.reshape((test2_X.shape[0], test2_X.shape[2]))
inv_yhat = concatenate((yhat, test2_X[:, 0:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# Invert scaling for actual
test2_Y = test2_Y.reshape((len(test2_Y), 1))
inv_y = concatenate((test2_Y, test2_X[:, 0:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# RMSE
rmse2 = sqrt(mean_squared_error(inv_y, inv_yhat))
print(rmse2)

# Plot
pyplot.plot(inv_y, label = 'actual')
pyplot.plot(inv_yhat, label = 'model')
pyplot.legend()
pyplot.show()


#==============================================================================
# Next Day Prediction
#==============================================================================

def next_day_pred(df, model, X_min, X_max):
    # Normalized Features
    test_values = df.values[-1, [0,1,2,3,4,5, -1]]
    #test_values = test_values.reshape((1, test_values.shape[0]))
    test_values = test_values.astype('float')
    scaled_test = (test_values - X_min)/(X_max - X_min)
    scaled_test = scaled_test.reshape((1, scaled_test.shape[0]))    
    #scaled_test = scaler.fit_transform(test_values)
    # Split into input and outputs
    test2_X, test2_Y = scaled_test[:, [0,1,2,3,4,5]], scaled_test[:, [-1]]   
    # Reshape input to be 3D [samples, timesteps, features]
    test2_X = test2_X.reshape((test2_X.shape[0], 1, test2_X.shape[1]))    
    ## New Data Prediction
    yhat = model.predict(test2_X)    
    # Invert scaling for forecast
    inv_yhat = yhat*(X_max[-1] - X_min[-1]) + X_min[-1]
    return inv_yhat


X_min = []
X_max = []
for i in range(0, train_values.shape[1]):
    X_min.append(min(train_values[:,i]))
    X_max.append(max(train_values[:,i]))
    
X_max = np.array(X_max)
X_min = np.array(X_min)
print(X_min, X_max)

df = my_data.ix['2017-11-01':'2017-11-16']
df = df.fillna(method = 'pad')
print(df)
next_day_pred(df, model, X_min, X_max)    



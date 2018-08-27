# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 09:49:34 2017

@author: prashant
"""

import os
os.chdir("E:/LearningPath/Projects/PythonForDS/code")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv('C:/Users/prashant/Downloads/international-airline-passengers.csv', usecols=[1], engine = 'python', skipfooter = 3)

plt.plot(dataframe)

# fix random seed for reproducibility
np.random.seed(1234)

dataset = dataframe.values
dataset = dataset.astype('float32')

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 00:12:02 2017

@author: prashant
"""

import os
os.chdir("E:/Learning/Projects/PythonForDS/PythonLearning/code/")


#==============================================================================
# # 1. How can you build logistic regression model in python
#==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# Data Pre-processing
data = sm.datasets.fair.load_pandas().data
data.describe()

data['affair'] = (data.affairs > 0).astype(int)

data.groupby('affair').mean()

# Preparing data for logistic regression
# Create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb

y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
religious + educ + C(occupation) + C(occupation_husb)',
data, return_type = 'dataframe')

print(X.columns)

# fix column names of X
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})


# Flatten y into a 1-D array
y = np.ravel(y)

# Logistic regression
model = LogisticRegression()
model = model.fit(X, y)

model.score(X, y)

y.mean()

                        








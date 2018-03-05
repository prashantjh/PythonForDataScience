#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:35:02 2018

@author: prashant
"""

# Tutorial with Pandas
# Dataset: Loan Prediction from Analytics Vidhya

# Set working directory
import os
os.chdir('/home/prashant/PRASHANT/Learning/Python/Tutorial/Pandas/')

## Reading the data
import pandas as pd
import numpy as np
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Shape of Data - Dimension
train.shape, test.shape

# Column names
train.columns

#==============================================================================
# 1. Boolean Indexing
#==============================================================================
# .loc for label based filtering - 'a':'f' has both included
# .iloc for integer position based filtering - 0 to length-1 of the axis

# Select Gender, Education, Loan_Status where Gender = Female, Education = Not Graduate & Loan_Status = Y
train.loc[(train["Gender"]=='Female') & (train["Education"]=='Not Graduate') & (train["Loan_Status"]=='Y'), ["Loan_ID","Gender","Education","Loan_Status"]]


#==============================================================================
# 2. Iteration over row of dataframe
#==============================================================================

# Load column types
coltypes = pd.read_csv('./data/datatypes.csv')
coltypes

# Check current types
train.dtypes

# Iterate through each row and assign feature type to each row
for i, row in coltypes.iterrows():
    if row.type == 'categorical':
        train[row.feature] = train[row.feature].astype(np.object)
    elif row.type == 'continuous':
        train[row.feature] = train[row.feature].astype(np.float)

print(train.dtypes)


#==============================================================================
# 3. Apply Function
#==============================================================================

# Find missing value in each row and column

# create a new funtion
def num_missing(x):
    return(sum(x.isnull()))

# applying per column
train.apply(num_missing, axis=0) # Axis=0 is for each column
train.apply(lambda x: sum(x.isnull()), axis=0)  # Aliter

# applying per row
sum(train.apply(num_missing, axis=1))   # All the missing values


#==============================================================================
# 4. Imputing Missing values
#==============================================================================

# Impute Categorical values with Mode and Continuous value with Median/Mean
from scipy.stats import mode
train.Gender.mode()[0]

train['Gender'].fillna(train.Gender.mode()[0], inplace = True)
train.Married.fillna(train.Married.mode()[0], inplace = True)
#train.Dependents.fillna(train.Dependents.mode()[0], inplace = True)
train.Self_Employed.fillna(train.Self_Employed.mode()[0], inplace = True)
#train.Credit_History.fillna(train.Credit_History.mode()[0], inplace = True)

#train.LoanAmount.fillna(train.LoanAmount.median(), inplace = True)
#train.Loan_Amount_Term.fillna(train.Loan_Amount_Term.median(), inplace = True)

train.apply(lambda x: sum(x.isnull()), axis=0)  # Aliter


#==============================================================================
# 5. Pivot Table
#==============================================================================

# Determine pivot table - Impute Loan Amount by mean of Gender, Married & Self_Employed
impute_group = train.pivot_table(values=['LoanAmount'], index = ['Gender', 'Married', 'Self_Employed'], aggfunc = np.mean)
impute_group


#==============================================================================
# 6. MultiIndexing
#==============================================================================

# Multiindexing is done using tuple
# Iterate through the data to find missing value for LoanAmount
for i, row in train.loc[train.LoanAmount.isnull(), :].iterrows():
    ind = tuple([row.Gender, row.Married, row.Self_Employed])
    train.loc[i, 'LoanAmount'] = impute_group.loc[ind].values[0]

train.apply(lambda x: sum(x.isnull()), axis=0)


#==============================================================================
# 7. Crosstab
#==============================================================================

# Cross-tab of Credit_History & Loan_Status
pd.crosstab(train.Credit_History, train.Loan_Status, margins = True)

# Converting to percentage using apply function
def percConvert(ser):
    return(100*ser/float(ser[-1]))

pd.crosstab(train.Credit_History, train.Loan_Status, margins = True).apply(percConvert, axis=1)


#==============================================================================
# 8. Merging DataFrame
#==============================================================================

# Merge train data with a temp DF of property rates
prop_rates = pd.DataFrame([1000, 5000, 12000], index = ['Rural', 'Semiurban', 'Urban'], columns = ['rates'])

train_merged = train.merge(right=prop_rates, how='inner', left_on='Property_Area', right_index=True, sort = False)

# Pivot table Property area, rates with Credit_History
train_merged.pivot_table(values='Credit_History', index=['Property_Area', 'rates'], aggfunc=len)

# Pivot table Property area, rates with Credit_History & LoanAmount
train_merged.pivot_table(values=['Credit_History', 'LoanAmount'], index=['Property_Area', 'rates'], aggfunc=[len, np.mean])



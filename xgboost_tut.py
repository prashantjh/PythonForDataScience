#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 07:56:46 2018

@author: prashant
"""

# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.getcwd()
os.chdir('/home/prashant/PRASHANT/Learning/Python/Tutorial/XGBoost/')

# Read training data
train_full = pd.read_csv('./data/train_full.csv', encoding = 'ISO-8859-1')  # encoding parameter to handle utf-8 encoding data
test_full = pd.read_csv('./data/test_full.csv', encoding = 'ISO-8859-1') 

# Data Sample
train_full.head(2), test_full.head(2)

# Dimension
train_full.shape, test_full.shape
train_full.columns, test_full.columns

# Column types
train_full.dtypes, test_full.dtypes

#### Data Processing ####

# Column names
train_full.columns
train_cols = ('ID', 'Gender', 'City', 'Monthly_Income', 'DOB', 'Lead_Creation_Date',
       'Loan_Amount_Applied', 'Loan_Tenure_Applied', 'Existing_EMI',
       'Employer_Name', 'Salary_Account', 'Mobile_Verified', 'Var5', 'Var1',
       'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Interest_Rate',
       'Processing_Fee', 'EMI_Loan_Submitted', 'Filled_Form', 'Device_Type',
       'Var2', 'Source', 'Var4', 'LoggedIn', 'Disbursed')

#==============================================================================
# Data Preparation Steps:
# City variable dropped because of too many categories
# DOB converted to Age | DOB dropped
# EMI_Loan_Submitted_Missing created which is 1 if EMI_Loan_Submitted was missing else 0 | Original variable EMI_Loan_Submitted dropped
# EmployerName dropped because of too many categories
# Existing_EMI imputed with 0 (median) since only 111 values were missing
# Interest_Rate_Missing created which is 1 if Interest_Rate was missing else 0 | Original variable Interest_Rate dropped
# Lead_Creation_Date dropped because made little intuitive impact on outcome
# Loan_Amount_Applied, Loan_Tenure_Applied imputed with median values
# Loan_Amount_Submitted_Missing created which is 1 if Loan_Amount_Submitted was missing else 0 | Original variable Loan_Amount_Submitted dropped
# Loan_Tenure_Submitted_Missing created which is 1 if Loan_Tenure_Submitted was missing else 0 | Original variable Loan_Tenure_Submitted dropped
# LoggedIn, Salary_Account dropped
# Processing_Fee_Missing created which is 1 if Processing_Fee was missing else 0 | Original variable Processing_Fee dropped
# Source – top 2 kept as is and all others combined into different category
# Numerical and One-Hot-Coding performed
# 
#==============================================================================



#==============================================================================
# 1. Data Processing
#==============================================================================

# Keep same columns from train & test
test_full['Disbursed'] = 0
train_full.drop(['LoggedIn'], axis=1, inplace = True)

# Combine Train & Test
train_full['type'] = 'Train'
test_full['type'] = 'Test'

data=pd.concat([train_full, test_full], ignore_index = True)


## Missing Value Count
#data.apply(lambda x: sum(x.isnull()), axis=0)


#Impute missing values
#Loan_Amount_Applied, Loan_Tenure_Applied, Existing_EMI
data.Loan_Amount_Applied.fillna(data.Loan_Amount_Applied.median(), inplace = True)
data.Loan_Tenure_Applied.fillna(data.Loan_Tenure_Applied.median(), inplace = True)
data.Existing_EMI.fillna(data.Existing_EMI.median(), inplace = True)

## Binary Variables
data['Loan_Amount_Submitted_ind'] = np.where(data.Loan_Amount_Submitted > 0, 1, 0)
data['Loan_Tenure_Submitted_ind'] = np.where(data.Loan_Tenure_Submitted > 0, 1, 0)
data['Interest_Rate_ind'] = np.where(data.Interest_Rate > 0, 1, 0)
data['Processing_Fee_ind'] = np.where(data.Processing_Fee > 0, 1, 0)
data['EMI_Loan_Submitted_ind'] = np.where(data.EMI_Loan_Submitted > 0, 1, 0)
#data.Filled_Form = data.Filled_Form.apply(lambda x: 1 if x=='Y' else 0)

## DOB column - Get Age and drop this column
data['Age'] = data.DOB.apply(lambda x: 118 - int(x[-2:]))

## Categorical Column
data.Source = data.Source.apply(lambda x: 'others' if x not in ['S122', 'S133'] else x)
data.Var2 = data.Var2.apply(lambda x: 'others' if x not in ['B', 'G', 'C'] else x)
data.Var1 = data.Var1.apply(lambda x: 'others' if x not in ['HBXX', 'HBXC', 'HBXB'] else x)

## One-Hot Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col_to_encode = ['Gender', 'Source', 'Var1', 'Var2', 'Mobile_Verified', 'Filled_Form']
for col in col_to_encode:
    data[col] = le.fit_transform(data[col])

data=pd.get_dummies(data, columns=col_to_encode, drop_first = True)
data.dtypes

## Drop unnecessary columns
# City, Age, Employer_Name, Salary_Account, Device_Type, DOB, Lead_Creation_Date, Var1, 
data.drop(['City', 'Loan_Amount_Submitted', 'Loan_Tenure_Submitted', 'Interest_Rate', 'Processing_Fee', 'EMI_Loan_Submitted', 'Employer_Name', 'Salary_Account', 'Device_Type', 'DOB', 'Lead_Creation_Date'], axis = 1, inplace = True)


#==============================================================================
# 2. Separate Train/Test Data
#==============================================================================

train = data.loc[data.type == 'Train']
train.drop(['type'], axis = 1, inplace = True)

test = data.loc[data.type == 'Test']
test.drop(['type', 'Disbursed'], axis = 1, inplace = True)



#==============================================================================
# 3. XGBoost Modeling
#==============================================================================

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4


target = 'Disbursed'
IDcol = 'ID'

def modelfit(alg, dtrain, predictors, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgbtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgbtrain, num_boost_round=alg.get_params()['n_estimators'], nfold = cv_folds, metrics = 'auc', early_stopping_rounds= early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators = cvresult.shape[0])
        
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric = 'auc')
    
    # Predict training set -
    dtrain_predictions = alg.predict(dtrain[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1])
    
    # Print model report -
    print("\nModel report:")
    print("Accuracy: %.4g"% metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    #print("AUC score (train): %f"% metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar', title = 'Feature Importances')
    plt.ylabel('Feature Imp. Score')
    

# Choose all predictors except target and ID
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier( learning_rate=0.1,
                     n_estimators=200,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic',
                     scale_pos_weight=1,
                     seed=27)

modelfit(xgb1, train, predictors)
        
        

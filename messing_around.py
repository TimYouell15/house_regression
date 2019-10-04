#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:19:17 2019

@author: youellt
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

file_dir = os.getcwd()
sys.path.append(file_dir)

csv_train = file_dir + r'/Documents/house-prices-advanced-regression-techniques/data/train.csv'
csv_test = file_dir + r'/Documents/house-prices-advanced-regression-techniques/data/test.csv'

# csv_train = os.path.join(file_dir, "data/train.csv")
# csv_test = os.path.join(file_dir, "data/test.csv")

df_csv = os.path.join(csv_train)
# csv_test = os.path.join(file_dir, "data/test.csv")

df = pd.read_csv(df_csv)
# X_test = pd.read_csv(csv_test)

train, test = train_test_split(df, test_size=0.3)

y_key = 'SalePrice'

X_train = train.drop(y_key, axis=1)
X_test = test.drop(y_key, axis=1)
Y_train = train[y_key].values.reshape(-1, 1)
Y_test = test[y_key].values.reshape(-1, 1)

X = X_train[['LotFrontage', 'LotArea']]
mi = Model_imputer(target_key='LotFrontage', predictive_keys=['LotArea'], model=LinearRegression())
mi.fit(X_train)

Y = X_test[['LotFrontage', 'LotArea']]
check = mi.transform(X_test)
check['LotFrontage'].isnull().value_counts()


df['LotFrontage'].isnull().value_counts()

plt.scatter(X_train['LotFrontage'], X_train['LotArea'])
plt.hist(X_train['LotFrontage'], bins=50)


'''
Try just a simple imputation first then maybe try a linear model to predict Lot frontage
'''

no_nulls = df[~df['LotFrontage'].isnull()]
X_no_nulls = no_nulls['LotArea'].values.reshape(-1, 1)
Y_no_nulls = no_nulls['LotFrontage'].values.reshape(-1, 1)


regr = LinearRegression()
regr.fit(X_no_nulls, Y_no_nulls)

print(regr.coef_)
print(regr.intercept_)



print(r2_score(no_nulls['LotFrontage'], no_nulls['LotFrontage']))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Oct  4 10:01:58 2019

@author: youellt
"""

import os
import sys
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import (
    PowerTransformer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost.sklearn import XGBRegressor
# from sklearn.linear_model import LinearRegression

from house_classes import (
        Column_selector, 
        # Model_imputer,
        ApplyTransformer,
        DummyTransformer
        )

from house_functions import (
        ZoneTransformer
        )

# SEED = 12345

# train['MSZoning'].isnull().value_counts()
# train['MSZoning'].value_counts()


def feature_engineering():
    '''
    Does the feature engineering (column selection, imputation,
    dummification etc) on the dataset.
    '''
    
    # 1. Numeric keys
    numeric_keys = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                    'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                    'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    numeric_pipeln = Pipeline([
        ('select', Column_selector(key=numeric_keys)),
        ('impute', SimpleImputer(strategy='median'))])
    
    # 2. Zone keys
    zoning_keys = ['RL', 'RM', 'FV', 'RH', 'ALL']
    zoning_pipeln = Pipeline([
            ('selector', Column_selector(key='MSZoning')),
            ('filter', ApplyTransformer(func=ZoneTransformer)),
            ('dummies', DummyTransformer(zoning_keys, **{'prefix': 'ZONE'}))
            ])
    
    
    # how old when sold feature?
    
    
    # 2. Frontage keys
#    frontage_keys = ['LotFrontage', 'LotArea']
#    front_target_key = 'LotFrontage'
#    front_prediction_keys = ['LotArea']
#    mi = Model_imputer(target_key = 'LotFrontage',
#                       predictive_keys = ['LotArea'],
#                       model=LinearRegression())
#    frontage_pipeln = Pipeline(mi)
            
            
            # ('imputer', Model_imputer(front_target_key, front_prediction_keys, LinearRegression()))
            # ('selector2', Column_selector(key=front_target_key))
    
    
    return [('numeric', numeric_pipeln),
            ('zoning', zoning_pipeln)]


def xgboost_params():
    xgb_regr = XGBRegressor(n_estimators=500,
                            learning_rate=0.01,
                            max_depth=5,
                            subsample=0.8,
                            booster='gbtree',
                            objective='reg:squarederror',
                            min_samples_leaf=5,
                            n_jobs=4,
                            random_state=42)

    return TransformedTargetRegressor(
        regressor=xgb_regr,
        transformer=PowerTransformer(method='yeo-johnson',
                                     standardize=True))


def feature_engineering_pipe():
    """Feature engineering pipeline"""

    features = FeatureUnion(feature_engineering())

    return Pipeline([
        ('engineer', features)])
       
        
def model(model=xgboost_params()):
    return Pipeline([('xgb', model)])


def model_pipeline():
    return Pipeline(feature_engineering_pipe().steps + 
                    model().steps)


if __name__ == '__main__':
    
    file_dir = os.getcwd()
    sys.path.append(file_dir)
    
    df_csv = os.path.join(file_dir, "data/train.csv")
    # csv_test = os.path.join(file_dir, "data/test.csv")
    
    df = pd.read_csv(df_csv)
    # X_test = pd.read_csv(csv_test)
    
    train, test = train_test_split(df, test_size=0.3,
                                   shuffle=False)
    
    y_key = 'SalePrice'
    
    # X_train = train.drop(y_key, axis=1)
    # X_test = test.drop(y_key, axis=1)
    Y_train = train[y_key].values.reshape(-1, 1)
    Y_test = test[y_key].values.reshape(-1, 1)
    
    # Y_train = train[[y_key]]
    
    # print(train['LotFrontage'])
    
    '''
    check to see how the pipeline is working
    
    feat_eng = pipeline()
    feat_eng.fit(X_train, Y_train)
    print('Finished Engineering!')
    train_eng = feat_eng.transform(X_train)
    print(train_eng)
    
    test_eng = feat_eng.transform(X_test)
    # print(test_eng)
    
    model = model_build()
    model.fit(train_eng, Y_train)
    Y_preds = model.predict(test_eng)
    print(Y_preds)
    
    .predict applies the same feature engineering onto X_test as in the
    fitting did in .fit to the training set
    '''
    
    new_model = model_pipeline()
    new_model.fit(train, Y_train)

    Y_preds = new_model.predict(test)
    df_res = pd.DataFrame({'Actual': Y_test.flatten(),
                           'Predicted': Y_preds.flatten()})
    print(df_res)
    print(r2_score(Y_test, Y_preds))
    print(mean_absolute_error(Y_test, Y_preds))
    
    
    
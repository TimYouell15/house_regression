#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:26:05 2019

@author: youellt
"""
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression


'''
Python script that contains customised sklearn transformers used in the data
engineering process.
'''

'''
BaseEstimator is a base class for all estimators in scikit-learn.
TransformerMixin class is a Mix in class for all transformers in scikit-learn.
'''


class Column_selector(BaseEstimator, TransformerMixin):
    ''' Transformer to select columns from a dataset.
    
    :param key: --->  a string or list of strings that represent a 
    column name in a dataset
    
    // Example //
    >>> X = pd.DataFrame([[1.0, 'a'], [2.0, 'b']],
                     columns = ['cost', 'cat'])
    
    1) key = list of strings, e.g. subset of dataframe (['col'])
    >>> cs = Column_selector(key=['cost', 'cat'])
    >>> cs.transform(X)
       cost cat
    0   1.0   a
    1   2.0   b
    
    >>> type(cs.transform(X))
    pandas.core.frame.DataFrame
    
    2) key = string, e.g. one column/series ('col')
    >>> cs = Column_selector(key='cost')
    >>> cs.transform(X)
       cost
    0   1.0
    1   2.0
    
    >>> type(cs.transform(X))
    pandas.core.frame.Series
    '''
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y=None):
        '''
        Fits the selector on X, where X is a pandas dataframe
        '''
        return self
    
    def transform(self, X, y=None):
        '''
        Transforms X according to the selector.
        
        returns a subset of X, i.e a column/series or dataframe.
        
        Pandas Series if key = string ('')
        Pandas Dataframe if key = list (['', ''])
        '''
        return X[self.key]
    

class Model_imputer(BaseEstimator, TransformerMixin):
    ''' Imputes values based on a different model such as linear regression.
    
    :param str target_key:  --->  feature to be imputed ('y'). If null, ignored.
    :param list predictive_keys:  --->  features used in the model ('x'). If null, ignored.
    :param sklearn.base.BaseEstimator model:  --->  sklearn model
    
    // Example //
    y = mx + c
    
    >>> X = pd.DataFrame([[1.1] , [2.2], [3.7], [4.9]], columns = ['x'])
    >>> X['y'] = 2 * X['x'] + 0.7
    >>> mi = Model_imputer(target_key='y',
                           predictive_keys=['x'],
                           model=LinearRegression())
    >>> mi.fit(X)
    >>> round(mi.model.coef_[0], 1)
    2.0
    >>> round(mi.model.intercept_, 1)
    0.7
    
    >>> Y = pd.DataFrame([[1.1, np.nan], [np.nan, np.nan]], columns=['x', 'y'])
    >>> mi.transform(Y)
         x    z
    0  1.1  2.9
    1  NaN  NaN
    '''
    def __init__(self, target_key, predictive_keys, model):
        '''
        Initiliased by:
        1) target_key = what you want imputing
        2) predictive keys = the features used to build the model to predict 
                             the target variable to be imputed
        3) model = the model used to do the calculation.
        '''
        self.target_key = target_key
        self.predictive_keys = predictive_keys
        self.model = model
        
    def fit(self, X, y=None):
        '''
        creates a new dataframe from the selection of target and predictors.
        '''
        x = X[[self.target_key] + self.predictive_keys]
        x = x.replace([np.inf, np.inf], np.nan)
        
        '''
        Fits the model on data without any nulls.
        '''
        fit_mask = x.notnull().all(axis=1)
        self.model.fit(x[self.predictive_keys][fit_mask],
                       x[self.target_key][fit_mask])
        
    def transform(self, X, y=None):
        if X[self.target_key].isnull().any():
            x = X[[self.target_key] + self.predictive_keys]
            x = x.replace([np.inf, np.inf], np.nan)
            
            trsf_mask = x[[self.target_key]].isnull().join(
                    x[self.predictive_keys].notnull()).all(axis=1)
            
            X.loc[:, self.target_key][trsf_mask] = self.model.predict(
                    x[self.predictive_keys][trsf_mask])
        return X


class ApplyTransformer(BaseEstimator, TransformerMixin):
    """
    Apply arbitrary function to pandas dataframe
    """
    def __init__(self, func, col=None, col_suffix='_f', **kwargs):
        self.func = func
        self.col = col
        self.col_suffix = col_suffix
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.col is None:
            return X.apply(self.func, **self.kwargs)
        else:
            X[self.col + self.col_suffix] = X[self.col].apply(self.func)
            return X


class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    Dummy Transformer
    """

    def __init__(self, keys=None, **kwargs):
        self.keys = keys
        self.kwargs = kwargs

    def fit(self, X, y=None):
        if self.keys is None:
            self.keys = pd.Series(X).unique()
        self.keys = pd.Series(self.keys)
        self.keys = self.keys[self.keys.notnull()]
        if 'prefix' in self.kwargs:
            prefixes = [self.kwargs['prefix']]*len(self.keys)
            self.r_keys = pd.Series(
                [prefix + '_' + key
                 for key, prefix in zip(self.keys, prefixes)])
        else:
            self.r_keys = self.keys
        return self

    def transform(self, X, y=None):
        X = pd.Series(X)
        X = pd.concat([X, self.keys])
        return pd.get_dummies(X, **self.kwargs)[self.r_keys][:-len(self.keys)]



  
    
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


class PandasFeatureUnion(_BaseComposition, TransformerMixin):
    """
    Simple version of `sklearn.pipeline.FeatureUnion
    <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html>`_
    that preserves pandas DataFrame type. Applies transformers and \
    concatenates output. Uses `Parallel
    <https://joblib.readthedocs.io/en/latest/parallel.html>`_
    to execute transformers in (embarassingly) parallel.

    :param list transformer_list: List of tuples ("name", \
    sklearn_transformer) i.e. transformer must have ``fit_transform`` method
    :param int n_jobs: number of processors to use

    **Example**

    >>> from sklearn.pipeline import Pipeline
    >>> X = pd.DataFrame([[1.0, 'a'], [2.0, 'a'], [np.nan, 'b']],
    ...                  columns=['cost', 'cat'])
    >>> dt = DummyTransformer(**{'prefix': 'cat'})
    >>> dummy_pipe = Pipeline([("select", ColumnSelector("cat")),
    ...                        ("dummy", dt)])
    >>> pi = PandasImputer(cols=['cost'])
    >>> pi_pipe = Pipeline([("imputer", pi),
    ...                     ("selector", ColumnSelector(['cost']))])
    >>> transformer_list = [("imputer", pi_pipe),
    ...                     ("encoder", dummy_pipe)]
    >>> pfu = PandasFeatureUnion(transformer_list)
    >>> pfu.fit_transform(X)
       cost  cat_a  cat_b
    0   1.0      1      0
    1   2.0      1      0
    2   1.5      0      1
    """

    def __init__(self, transformer_list, n_jobs=1):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs

    @staticmethod
    def _fit_one_transformer(transformer, X, y):
        label, transformer_ = transformer
        return label, transformer_.fit(X, y)

    @staticmethod
    def _transform_one(transformer, X, y):
        _, transformer_ = transformer
        return transformer_.transform(X)

    def fit(self, X, y=None):

        self.transformer_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one_transformer)(trans, X, y)
            for trans in self.transformer_list)
        return self

    def transform(self, X, y=None):
        Xout = Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform_one)(trans, X, y)
            for trans in self.transformer_list)
        Xout = pd.concat(Xout, axis=1)
        return Xout


  
    
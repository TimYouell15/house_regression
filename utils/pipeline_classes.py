#!/usr/bin/env python

import sys
sys.path.append(r'/home/tyouell/Documents/Coding/house_price_regression/')

import datetime
import itertools

import pandas as pd
import numpy as np

import sklearn.linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

from utils.pipeline_functions import (
    get_day, get_weekday, is_weekend, get_month,
    get_season, get_hour, get_days, one_day_more,
    get_time_of_day, dam_area_2_nulls, get_year
)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select single columns from data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.key]
    
 
class ColumnConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to select single columns from data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.key].fillna(-1, inplace=True)


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


class GroupbyTransformer(BaseEstimator, TransformerMixin):
    """
    Create Lookup tables to past/training values
    Map to test/val features
    """
    def __init__(self, key1, key2, key3, func, tkey=None, tdelay=180):
        self.key1 = key1
        self.key2 = key2
        self.key3 = key3
        self.func = func
        self.lookup_ = None
        self.tkey = tkey
        self.tdelay = tdelay

    def fit(self, X, y=None):
        if self.tkey is not None:
            cutoff = X[self.tkey].max() - datetime.timedelta(days=self.tdelay)
            mask = X[self.tkey] > cutoff
        else:
            mask = ~pd.Series(index=X.index, dtype=bool)
        self.lookup_ = X[mask].groupby(self.key1)[self.key2].apply(self.func)
        return self

    def transform(self, X, y=None):
        X = X.set_index(self.key1)
        X[self.key3] = self.lookup_
        X = X.reset_index()
        return X


class GroupbyImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values using historical lookups
    """
    def __init__(self, key1, key2, func):

        self.key1 = key1
        key_combs_ = [[list(ii) for ii in itertools.combinations(key1, jj)]
                      for jj in reversed(range(1, len(self.key1) + 1))]
        key_combs_ = [kk for kk in itertools.chain.from_iterable(key_combs_)]

        self.key_combs_ = key_combs_
        self.Nkey_ = len(key_combs_)

        self.key2 = key2
        self.func = func

    def fit(self, X, y=None):

        self.st_ = []
        for ii, lookup_this_ in enumerate(self.key_combs_):
            gt = GroupbyTransformer(
                lookup_this_,
                self.key2,
                'summary_' + self.key2 + '_' + '{:d}'.format(ii),
                self.func)
            self.st_.append(gt.fit(X))
        return self

    def transform(self, X, y=None):
        for ii, st_this_ in enumerate(self.st_):
            Y = st_this_.transform(X)
            X[self.key2].fillna(
                Y['summary_' + self.key2 + '_' + '{:d}'.format(ii)],
                inplace=True)
        return X


class ModelImputer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 target_key,
                 predictive_keys,
                 glm=sklearn.linear_model.LinearRegression()):

        self.target_key = target_key
        self.predictive_keys = predictive_keys
        self.glm = glm

    def fit(self, X, y=None):
        X = X[[self.target_key] + self.predictive_keys].replace([np.inf, -np.inf],
                                                                np.nan)

        fit_mask = X[[self.target_key] + self.predictive_keys].notnull().all(axis=1)

        self.glm.fit(X[self.predictive_keys][fit_mask],
                     X[self.target_key][fit_mask])
        return self

    def transform(self, X, y=None):
        if X[self.target_key].isnull().any():
            x = X[[self.target_key] + self.predictive_keys].replace([np.inf, -np.inf],
                                                                    np.nan)
            transform_mask = x[[self.target_key]].isnull().join(
                x[self.predictive_keys].notnull()).all(axis=1)

            X.loc[:, self.target_key][transform_mask] = self.glm.predict(
                x[self.predictive_keys][transform_mask])
        return X


class RatesImputer(BaseEstimator, TransformerMixin):
    """
    Fills in any missing labour rates based on Fault/Non-Fault
    Rule
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        components = ['Parts', 'Paint']
        keys = ['audatex_' + component + '_Labour_Rate'
                for component in components]
        liability_keys = [('Fault', 27.5), ('Non-Fault', 42.5)]
        for liability, rate in liability_keys:
            liability_mask = X['cla_liability'] == liability
            # .loc avoids SettingWithCopyWarning (rather than inplace=True)
            X.loc[liability_mask, keys] = X.loc[liability_mask, keys].fillna(value=rate)
        X.loc[:, keys] = X.loc[:, keys].fillna(value=30)
        return X


class OrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, col, ordinal_dict=None, na_action='fill'):
        self.col = col
        self.ordinal_dict = ordinal_dict
        self.na_action = na_action

    def fit(self, X, y=None):
        if self.ordinal_dict is None:
            vc_kwargs = {'sort': True, 'ascending': True}
            val_counts = X[self.col].value_counts(**vc_kwargs)

            # Sets the rarest occurrences to -1 and rescales
            # others accordingly. Categories that don't appear in
            # test set will be set to -1 in transform.
            idx, vals = val_counts.index, val_counts.values
            vals_min = vals.min()
            min_occur = val_counts.value_counts().loc[vals_min]
            vals[vals == vals_min] = -1

            self.ordinal_dict = dict((id_, i) if val > -1 else (id_, -1)
                                     for i, (id_, val) in enumerate(zip(idx, vals),
                                                                    -min_occur))
        return self

    def transform(self, X, y=None):
        X[self.col + '_enc'] = X[self.col].map(self.ordinal_dict)
        if self.na_action != 'ignore':
            X[self.col + '_enc'] = X[self.col + '_enc'].fillna(-1).astype(np.int32)
        return X


class PandasImputer(BaseEstimator, TransformerMixin):

    def __init__(self, cols, missing_values=np.nan, strategy="mean", **kwargs):
        self.cols = cols
        self.missing_values = missing_values
        self.strategy = strategy
        self.kwargs = kwargs

    def fit(self, X, y=None):
        si = SimpleImputer(missing_values=self.missing_values,
                           strategy=self.strategy,
                           **self.kwargs)
        si.fit(X[self.cols])
        self.si = si
        return self

    def transform(self, X, y=None):
        X.loc[:, self.cols] = self.si.transform(X.loc[:, self.cols])
        return X


class PandasVarianceThreshold(BaseEstimator, TransformerMixin):

    def __init__(self, cols=None, threshold=0.0):
        self.cols = cols
        self.threshold = threshold

    def fit(self, X, y=None):
        vt = VarianceThreshold(threshold=self.threshold)
        if self.cols is None:
            self.cols = X.columns.tolist()
        vt.fit(X[self.cols])
        self.vt = vt
        return self

    def transform(self, X, y=None):
        return_cols = [
            col
            for col, sup in zip(self.cols, self.vt.get_support())
            if sup]
        return X[return_cols]


class PandasFeatureUnion(_BaseComposition, TransformerMixin):

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for label, transformer in self.transformer_list:
            transformer.fit(X)
        return self

    def transform(self, X, y=None):
        Xout = pd.DataFrame(index=X.index)
        for label, transformer in self.transformer_list:
            Xout = Xout.join(transformer.transform(X))
        self.feature_names_ = Xout.columns
        return Xout


class CreateFeatures(BaseEstimator, TransformerMixin):
    """Create some (hopefully!) useful features for VRC models"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Time features
        key0 = 'cla_accident_datetime'
        X.loc[:, 'cla_hour'] = X[key0].apply(get_hour)
        X.loc[:, 'cla_day'] = X[key0].apply(get_day)
        X.loc[:, 'cla_daytime'] = X[key0].apply(get_time_of_day)
        X.loc[:, 'cla_weekday'] = X[key0].apply(get_weekday)
        X.loc[:, 'is_weekend'] = X[key0].apply(is_weekend)
        X.loc[:, 'cla_month'] = X[key0].apply(get_month)
        X.loc[:, 'cla_season'] = X[key0].apply(get_season)

        key1 = 'cla_notification_datetime'
        X.loc[:, 'cla_notification_time'] = (X[key1] - X[key0]).apply(get_days)
        X.loc[:, 'cla_long_notification'] = (X[key1] - X[key0]).apply(one_day_more)

        X.loc[:, 'veh_age'] = X[key0].apply(get_year) - X['veh_manufactured_year']

        # remove spurious weights
        X.loc[:, 'veh_weight'] = X['veh_weight'].where(X['veh_weight'] > 500, np.nan)

        # poly features
        X.loc[:, 'veh_engine_size_x_veh_severity_count'] = X['veh_engine_size'] * \
                                                           X['veh_severity_count']

        # Carweb composite features
        X.loc[:, 'carweb_vol'] = X['carweb_LENGTH'] * \
                                 X['carweb_WIDTH'] * \
                                 X['carweb_HEIGHT'] * 1e-9

        X.loc[:, 'carweb_surf'] = 2 * (X['carweb_LENGTH'] * X['carweb_WIDTH'] * 0.5 +
                                       X['carweb_HEIGHT'] * X['carweb_LENGTH'] +
                                       X['carweb_WIDTH'] * X['carweb_HEIGHT']) * 1e-6

        # Glass depreciation
        X.loc[:, 'glass_depreciation'] = X['glass_new_vehicle_price'] - X['glass_pav']

        # Damage Area 2
        X.loc[:, 'veh_DAM_Area_2_binary'] = X['veh_DAM_Area_2'].apply(dam_area_2_nulls)

        return X


class CreateMetaFeatures(BaseEstimator, TransformerMixin):
    """
    Create some (hopefully!) useful features for
    a second layer of the VRC model stack
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X.loc[:, 'sum_predictions'] = X['audatex_Parts_Material_Net_p'] + \
                                      X['audatex_Paint_Material_Net_p'] + \
                                      X['audatex_Parts_Labour_Hours_p'] * X['audatex_Parts_Labour_Rate'] + \
                                      X['audatex_Paint_Labour_Hours_p'] * X['audatex_Paint_Labour_Rate']

        return X


class InflationTransformer(BaseEstimator, TransformerMixin):
    """
    Completely bespoke transformer for tracking how some
    agg features have changed over time
    """

    def __init__(self, col_keys, tkeys):
        self.col_keys = col_keys
        self.tkeys = tkeys

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col_key in self.col_keys:
            X.loc[:, 'infl_' + col_key] = X[col_key + self.tkeys[0]] / X[col_key + self.tkeys[1]]
            X['infl_' + col_key].replace([np.inf, -np.inf], np.nan, inplace=True)
            X['infl_' + col_key].fillna(-1, inplace=True)

        return X

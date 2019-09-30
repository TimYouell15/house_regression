#!/usr/bin/env python3

import sys
sys.path.append(r'/home/tyouell/Documents/Coding/house_price_regression/')

import os

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.externals import joblib

from utils.pipeline_classes import (
    ColumnSelector, DummyTransformer,
    ApplyTransformer, GroupbyTransformer,
    GroupbyImputer, CreateFeatures,
    CreateMetaFeatures, InflationTransformer,
    PandasImputer, PandasFeatureUnion,
    RatesImputer, ColumnConverter
)

from utils.pipeline_functions import (
    MANUAL2Manual, loss_cause_groups,
    vehicle_colour_groups
)

from utils.pipeline_parameters import (
    vrc_dam_keys, vrc_sev_keys
)

# from utils.pipeline_utils import csv_kwargs
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor

SEED = 13456

file = r'/home/tyouell/Documents/Coding/house_price_regression/data//'
# read_kwargs = csv_kwargs()
train = pd.read_csv(file + "train.csv") #, **read_kwargs)
test = pd.read_csv(file + "test.csv")

train.any().describe()
lookup = train.isnull().describe()

describe = train.describe()

train['MSSubClass'].isnull().value_counts()

train['LotFrontage'].dtype
train['LotFrontage'].isnull().value_counts()
train['LotFrontage'].astype(int)

plt.hist(train['LotFrontage'])

train.columns.tolist()


def feature_engineering():

    # vocab_keys = ["driving", "road", "hit", "rear", "pulled", "side",
    #               "right", "parked", "park", "damage", "passenger", "door",
    #               "collided", "roundabout", "front", "junction", "van", "left",
    #               "traffic", "lights", "police", "reversed", "stationary",
    #               "stop", "space", "collision", "bumper", "lorry", "weather",
    #               "scraped"]
    # description_pipeln = Pipeline([
    #     ('selector', ColumnSelector(key='cla_claim_description')),
    #     ('imputer', ApplyTransformer(func=description_imputer)),
    #     ('vec', TfidfVectorizer(vocabulary=vocab_keys))])

    # numeric keys
#    numeric_keys = ["MSSubClass"]
#    numeric_pipeln = Pipeline([
#        ('selector', ColumnSelector(key=numeric_keys)),
#        ('imputer', SimpleImputer(strategy='mean'))])
#
#    numeric_pipeln = Pipeline([
#        ('selector', ColumnSelector(key=numeric_keys)),
#        ('imputer', PandasImputer(cols=numeric_keys))])

    # integer keys
#    integer_keys = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
#                    'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1']
    integer_keys = ['MSSubClass', 'LotArea']
    integer_pipeln = Pipeline([
            ('selector', ColumnSelector(key=integer_keys)),
            ('converter', ColumnConverter(key=integer_keys)),
            ('imputer', SimpleImputer(missing_values=-1,
                                      strategy='most_frequent'))])
    
    
    
#    integer_keys = ['veh_age', 'veh_seats', 'veh_doors', 'veh_damage_count',
#                    'veh_Max_Damage_Severity', 'veh_Mode_Damage_Severity',
#                    'veh_severity_count', 'carweb_NO_CYLINDERS',
#                    'carweb_VALVES_PER_CYLINDER', 'carweb_GROUP_1_50',
#                    'carweb_NCAP_OVERALL', 'carweb_GROUP_1_20']
#    integer_pipeln = Pipeline([
#        ('selector', ColumnSelector(key=integer_keys)),
#        ('imputer', PandasImputer(cols=integer_keys,
#                                  strategy='most_frequent')),
#        ('imputer9', PandasImputer(cols=integer_keys,
#                                   missing_values=999,
#                                   strategy='most_frequent'))])

    # times
#    time_keys = ['cla_hour', 'cla_day', 'cla_daytime', 'cla_weekday',
#                 'is_weekend', 'cla_month', 'cla_season', 'cla_notification_time',
#                 'cla_long_notification']
#    times_pipeln = Pipeline([
#        ('selector', ColumnSelector(key=time_keys))])
#
#    # sev and dam keys
#    sev_keys = vrc_sev_keys()
#    sev_keys = ['veh_DAM_SEV_' + key for key in sev_keys]
#    sev_pipeln = Pipeline([
#        ('selector', ColumnSelector(key=sev_keys)),
#        ('imputer', PandasImputer(cols=sev_keys,
#                                  strategy='constant',
#                                  fill_value=0))])  #,
#        # ('var', PandasVarianceThreshold(cols=sev_keys,
#        #                                 threshold=0.1))])
#
#    dam_keys = vrc_dam_keys()
#    dam_keys = ['veh_DAM_' + key for key in dam_keys]
#    dam_keys += ['veh_DAM_Area_2_binary']
#    dam_pipeln = Pipeline([
#        ('selector', ColumnSelector(key=dam_keys)),
#        ('imputer', PandasImputer(cols=dam_keys,
#                                  strategy='most_frequent'))])  #,
#        # ('var', PandasVarianceThreshold(cols=dam_keys,
#        #                                 threshold=0.1))])
#
#    # short cat keys
#    car_key = ['CAR', 'VAN']
#    car_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='veh_vehicle_wireframe')),
#        ('dummies', DummyTransformer(car_key))])
#
#    fuel_key = ['Diesel', 'Petrol']
#    fuel_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='veh_fuel_type')),
#        ('dummies', DummyTransformer(fuel_key))])
#
#    dam_area_keys = ['Frontal', 'Rear', 'Nearside', 'Offside',
#                     'Offside Front', 'Offside Rear',
#                     'Nearside Front', 'Nearside Rear']
#    dam_area_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='veh_DAM_Area_1')),
#        ('dummies', DummyTransformer(keys=dam_area_keys))])
#
#    veh_cat_keys = ['ESTATE', 'VAN', 'PEOPLE CARRIER', 'COUPE',
#                    'HATCHBACK', 'CONVERTIBLE/CABRIOLET', 'SALOON',
#                    'LIGHT VAN', 'PICK-UP']
#    veh_cat_pipeln = Pipeline([('selector', ColumnSelector(key='veh_vehicle_category')),
#                               # ('filter', ApplyTransformer(func=vehicle_type_groups)),
#                               ('dummies', DummyTransformer(veh_cat_keys))])
#
#    transmission_key = ['Manual']
#    transmission_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='veh_transmission')),
#        ('clean', ApplyTransformer(func=MANUAL2Manual)),
#        ('dummies', DummyTransformer(transmission_key))])
#
#    liability_keys = ['Fault', 'Non-Fault', 'Not Applicable']
#    liability_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='cla_liability')),
#        ('dummies', DummyTransformer(liability_keys))])
#
#    drivetype_keys = ['4X4']
#    drivetype_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_DRIVE_TYPE')),
#        ('dummies', DummyTransformer(drivetype_keys))])
#
#    aspiration_keys = ['TURBO CHARGED', 'NATURALLY ASPIRATED']
#    aspiration_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_ASPIRATION')),
#        ('dummies', DummyTransformer(aspiration_keys))])
#
#    eng_loc_keys = ['FRONT']
#    eng_loc_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_ENGINE_LOCATION')),
#        ('dummies', DummyTransformer(eng_loc_keys))])
#
#    eng_config_keys = ['IN LINE']
#    eng_config_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_ENGINE_CONFIG')),
#        ('dummies', DummyTransformer(eng_config_keys))])
#
#    valve_gear_keys = ['DOHC', 'SOHC']
#    valve_gear_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_VALVE_GEAR')),
#        ('dummies', DummyTransformer(valve_gear_keys))])
#
#    wheelbase_keys = ['SHORT WHEELBASE', 'MEDIUM WHEELBASE', 'LONG WHEELBASE']
#    wheelbase_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_WHEELBASE')),
#        ('dummies', DummyTransformer(wheelbase_keys))])
#
#    market_seg_keys = ['SUPER MINI', 'LOWER MEDIUM', 'DUAL PURPOSE', 'UPPER MEDIUM',
#                       'LCVs (Heavy Vans 2601-3500 Kgs)', 'LCVs (Medium Vans 2001-2600 Kgs)',
#                       'LCVs (Car-Derived/Integral <2000 Kgs)']
#    market_seg_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_MARKET_SEGMENT')),
#        # ('clean', ApplyTransformer(func=clean_market_segments)),
#        ('dummies', DummyTransformer(market_seg_keys,
#                                     **{'prefix': 'MARKET_SEG'}))])
#
#    security_keys = ['T1', 'T2']
#    security_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='carweb_SECURITY_STATUS')),
#        ('dummies', DummyTransformer(security_keys))])
#
#    # long cat keys
#    colour_keys = ['WHITE', 'BLUE', 'BLACK', 'GREY', 'SILVER']
#    colour_pipeln = Pipeline([('selector', ColumnSelector(key='veh_vehicle_colour')),
#                              ('filter', ApplyTransformer(func=vehicle_colour_groups)),
#                              ('dummies', DummyTransformer(colour_keys))])  #,
#                              # ('var', PandasVarianceThreshold(threshold=0.025))])
#
#    loss_causes = ['Impact No Third Party Involved', 'Third Party Hit Insured In Rear']
#    loss_cause_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='cla_loss_cause')),
#        ('filter', ApplyTransformer(func=loss_cause_groups)),
#        ('dummies', DummyTransformer(loss_causes))])  #,
#        # ('var', PandasVarianceThreshold(threshold=0.025))])
#
#    manuf_keys = ['FORD', 'VOLKSWAGEN', 'VAUXHALL']
#    manuf_pipeln = Pipeline([
#        ('selector', ColumnSelector(key='veh_manufacturer')),
#        ('dummies', DummyTransformer(manuf_keys))])  #,
#        # ('var', PandasVarianceThreshold(threshold=0.025))])
#
#    # lookup agg audatex pipeline
#    lookup_keys = ['veh_manufacturer', 'veh_Mode_Damage_Severity']
#    select_keys = ['audatex_Total_Repair_Net', 'audatex_Parts_Material_Net',
#                   'audatex_Paint_Material_Net', 'audatex_Additional_Net',
#                   'audatex_Parts_Labour_Hours', 'audatex_Replace_Operations',
#                   'audatex_Repair_Operations']
#    agg_methods = [('mean', np.mean),
#                   ('std', np.std),
#                   ('max', np.amax)]
#    tkey = 'cla_notification_datetime'
#    tdelays = [('3mo', 91.25),
#               ('6mo', 182.5),
#               ('1yr', 365)]
#    aggs = [('gt_' + key + '_' + agg_str + '_' + td_str,
#             GroupbyTransformer(lookup_keys,
#                                key,
#                                key + '_' + agg_str + '_' + td_str,
#                                agg_func,
#                                tkey=tkey,
#                                tdelay=tdelay))
#            for key in select_keys
#            for agg_str, agg_func in agg_methods
#            for td_str, tdelay in tdelays]
#    col_keys = [agg[1].key3 for agg in aggs]
#
#    base_col_keys = [key.strip('3mo')[:-1] for key in col_keys if key.endswith('3mo')]
#    infl_keys = ['infl_' + key for key in base_col_keys]
#
#    lookup_pipeln = Pipeline(
#        aggs + [('infl', InflationTransformer(base_col_keys, ['_3mo',
#                                                              '_1yr'])),
#                ('selector', ColumnSelector(key=col_keys + infl_keys)),
#                ('imputer', PandasImputer(cols=col_keys + infl_keys,
#                                          strategy='mean'))])
#
#    complex_imputation_vars = ['glass_pav', 'glass_new_vehicle_price',
#                               'glass_depreciation', 'carweb_vol',
#                               'carweb_surf', "veh_engine_size", "veh_weight",
#                               "veh_engine_size_x_veh_severity_count",
#                               "carweb_LENGTH", "carweb_WIDTH",
#                               "carweb_HEIGHT", "carweb_BORE", "carweb_STROKE",
#                               "carweb_ACCEL_0TO100_KPH", "carweb_GVW",
#                               "carweb_KERB_WEIGHT_MIN"]
#    lookup_keys = ['veh_manufacturer',
#                   'veh_short_model',
#                   'veh_manufactured_year']
#    gtis = [('gi_' + var, GroupbyImputer(lookup_keys, var, np.mean))
#            for var in complex_imputation_vars]
#    complex_pav_pipeln = Pipeline(
#        gtis + [('selector', ColumnSelector(key=complex_imputation_vars)),
#                ('simple', PandasImputer(cols=complex_imputation_vars,
#                                         strategy='mean'))])
#
#    rates_pipeln = Pipeline([
#        ('rates_imputer', RatesImputer()),
#        ('selector', ColumnSelector(key=['audatex_Parts_Labour_Rate',
#                                         'audatex_Paint_Labour_Rate']))])
#
#    infl_parts_cols = ['nsf_osf_infl', 'wheel_infl', 'airbag_infl']
#    infl_parts_pipeln = Pipeline([('selector',
#                                   ColumnSelector(infl_parts_cols))])

    return [('integers', integer_pipeln)]


def transformed_xgb():
    xgb_regr = XGBRegressor(n_estimators=500,
                            learning_rate=0.1,
                            max_depth=5,
                            subsample=0.8,
                            booster='gbtree',
                            objective='reg:linear',
                            min_samples_leaf=5,
                            n_jobs=4,
                            random_state=42)

    return TransformedTargetRegressor(
        regressor=xgb_regr,
        transformer=PowerTransformer(method='yeo-johnson',
                                     standardize=True))


def build_model():

    features = FeatureUnion(feature_engineering())
    features = PandasFeatureUnion(feature_engineering())

    model = Pipeline([('create', CreateFeatures()),
                      ('features', features),
                      ('regr', transformed_xgb())])
    return model


features = PandasFeatureUnion(feature_engineering())
print(features)

model = Pipeline([('features', features),
                  ('regr', transformed_xgb())])

print(model)


train_cut = train[['MSSubClass', 'LotArea']]


engineering = model
train_engineered = model.fit_transform(train_cut)








def meta_features():

    keys = ['sum_predictions', 'audatex_Paint_Material_Net_p',
            'audatex_Parts_Material_Net_p', 'audatex_Parts_Labour_Hours_p',
            'audatex_Paint_Labour_Hours_p', 'audatex_Replace_Operations_p']

    meta_features = Pipeline([
        ('selector', ColumnSelector(key=keys))])

    return [('meta', meta_features)]


def build_meta_model():

    enrich_X = Pipeline([('create', CreateFeatures()),
                         ('create_meta', CreateMetaFeatures())])

    first_level_features = feature_engineering()
    second_level_features = meta_features()

    features = FeatureUnion(first_level_features + second_level_features)

    return Pipeline([('enriched_X', enrich_X),
                     ('features', features),
                     ('regr', transformed_xgb())])


if __name__ == "__main__":

    file_dir = os.getcwd()
    train_file_path = os.path.join(file_dir, "train_test_data/CAR_train.csv")

    read_kwargs = read_train_test_csv_kwargs()
    X = pd.read_csv(train_file_path, **read_kwargs, nrows=1000)

    y_key = 'audatex_Total_Repair_Net'

    y = X[y_key].values.reshape(-1, 1)

    model = build_model()
    Xt = model.fit_transform(X)
    import sys; sys.exit()
    import pdb;pdb.set_trace()
    search = False
    if search:
        param_grid = {
            "regr__regressor__learning_rate": [0.05, 0.025],
            "regr__regressor__n_estimators": [1250, 1000],
            "regr__regressor__max_depth": [5, 4],
        }
        cv = KFold(n_splits=4, shuffle=True, random_state=SEED)
        Search = GridSearchCV(model, cv=cv, param_grid=param_grid,
                              verbose=2, n_jobs=1,
                              scoring='neg_mean_squared_error',
                              refit=True)
        Search.fit(X, y)
        model = Search.best_estimator_
        print(Search.best_params_)
    else:
        model.fit(X, y)

    joblib.dump(model, './fitted_models/total_repair_net_model.joblib')




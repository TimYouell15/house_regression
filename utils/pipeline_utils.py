#!/usr/bin/env python

import datetime

from utils.pipeline_parameters import (
    vrc_dam_keys, vrc_sev_keys
)


def datestr_to_datetime(date_str):
    """ Parses strings into datetime objects"""
    date_fmt = "%d%b%Y:%H:%M:%S.%f"
    return datetime.datetime.strptime(date_str, date_fmt)


def data_quality_mask(df):
    """Filters data"""
    mask = ((df.cla_claim_number > 4500000000) &
            df.veh_manufacturer.notnull() &
            df.veh_short_model.notnull() &
            df.veh_model.notnull() &
            df.veh_vehicle_category.notnull() &
            df.veh_manufactured_year.notnull() &
            df.veh_DAM_Airbag_Deployed.notnull() &
            (df.veh_manufactured_year > 1885))

    return mask


def audatex_quality_mask(df):
    """Filters data based on audatex data quality"""

    mask = ((df.audatex_Total_Repair_Net > 100.00) &
            df.copart_CAT.isnull() &
            (((df.audatex_Parts_Labour_Hours > 0) &
              (df.audatex_Paint_Labour_Hours > 0) &
              (df.audatex_Parts_Material_Net > 0) &
              (df.audatex_Paint_Material_Net > 0)) |
             ((df.audatex_Parts_Labour_Hours > 0) &
              (df.audatex_Paint_Labour_Hours == 0) &
              (df.audatex_Parts_Material_Net > 0) &
              (df.audatex_Paint_Material_Net == 0)) |
             ((df.audatex_Parts_Labour_Hours == 0) &
              (df.audatex_Paint_Labour_Hours > 0) &
              (df.audatex_Parts_Material_Net == 0) &
              (df.audatex_Paint_Material_Net > 0))))

    return mask


def read_train_test_csv_kwargs():
    """Provides kwargs for pandas.read_csv"""
    read_dtypes = {
        'cla_claim_number': str,
        'cla_policy_number': str,
        'cla_policy_product_code': str,
        'cla_accident_datetime': str,
        'cla_notification_datetime': str,
        'cla_liability': str,
        'cla_liability_percentage': float,
        'cla_fault_agreed': str,
        'cla_loss_cause_filter': str,
        'cla_loss_cause': str,
        'cla_claim_description': str,
        # 'cla_accident_location': str,
        'cla_accident_postcode': str,
        'cla_excess': float,
        'veh_vrm': str,
        'veh_manufacturer': str,
        'veh_model': str,
        'veh_short_model': str,
        'veh_vehicle_colour': str,
        'veh_vehicle_wireframe': str,
        'veh_vehicle_category': str,
        # 'veh_manufactured_year': str,
        'veh_manufactured_year': int,
        'veh_SPEED': float,
        'veh_mileage': float,
        'veh_passengers': float,
        'veh_engine_size': float,
        'veh_fuel_type': str,
        'veh_transmission': str,
        'veh_doors': float,
        'veh_seats': float,
        'veh_weight': float,
        'veh_tp_vehicle': str,
        'veh_DAM_Area_1': str,
        'veh_DAM_Area_2': str,
        'veh_Max_Damage_Severity': int,
        'veh_Mode_Damage_Severity': int,
        'veh_driver_postcode': str,
        'veh_Damaged_FLag': int,
        'veh_tp_insurer': str,
        # 'veh_age': int,
        'veh_damage_count': int,
        'veh_severity_count': float,
        'veh_liability': str,
        'veh_liability_percentage': float,
        'glass_VRM': str,
        # 'glass_doors': float,
        # 'glass_gears': float,
        # 'glass_fuel_type': str,
        # 'glass_engine_size': float,
        # 'glass_body_type': str,
        # 'glass_transmission': str,
        'glass_insurance_group': float,
        # 'glass_manufactured_year': float,
        'glass_new_vehicle_price': float,
        'glass_pav': float,
        # 'carweb_RANGE_SERIES': str,
        'carweb_VARIANT': str,
        'carweb_ASPIRATION': str,
        'carweb_COUNTRY_OF_ORIGIN': str,
        'carweb_DRIVE_TYPE': str,
        'carweb_ENGINE_LOCATION': str,
        'carweb_DRIVING_AXLE': str,
        'carweb_ENGINE_MAKE': str,
        'carweb_ENGINE_MODEL': str,
        'carweb_ENGINE_CONFIG': str,
        'carweb_FUEL_DELIVERY': str,
        'carweb_BORE': float,
        'carweb_STROKE': float,
        'carweb_NO_CYLINDERS': float,
        'carweb_VALVES_PER_CYLINDER': float,
        'carweb_VALVE_GEAR': str,
        'carweb_POWER_KW': float,
        'carweb_POWER_BHP': float,
        'carweb_POWER_RPM': float,
        'carweb_TORQUE_NM': float,
        'carweb_TORQUE_LBFT': float,
        'carweb_TORQUE_RPM': float,
        'carweb_MAX_SPEED_MPH': float,
        'carweb_MAX_SPD_KPH': float,
        'carweb_ACCEL_0TO100_KPH': float,
        'carweb_URBAN_COLD_MPG': float,
        'carweb_URBAN_COLD_LITRE_100KMS': float,
        'carweb_EXTRA_URBAN_MPG': float,
        'carweb_EXTRA_URBAN_LITRE_100KMS': float,
        'carweb_COMBINED_MPG': float,
        'carweb_COMBINED_LITRE_100KMS': float,
        'carweb_CO2': float,
        'carweb_LENGTH': float,
        'carweb_WIDTH': float,
        'carweb_HEIGHT': float,
        'carweb_WHEELBASE': str,
        'carweb_WHEELBASE_TYPE': str,
        'carweb_GVW': float,
        'carweb_NCAP_OVERALL': float,
        'carweb_GROUP_1_50': float,
        'carweb_GROUP_1_20': float,
        'carweb_MARKET_SEGMENT': str,
        'carweb_KERB_WEIGHT_MIN': float,
        'carweb_SECURITY_STATUS': str
    }

    dam_keys = vrc_dam_keys()
    dam_dict = {'veh_DAM_' + key: int for key in dam_keys}

    sev_keys = vrc_sev_keys()
    sev_dict = {'veh_DAM_SEV_' + key: float for key in sev_keys}

    audatex_dict = {
        'audatex_Parts_Labour_Hours': float,
        'audatex_Parts_Labour_Rate': float,
        'audatex_Parts_Labour_Net': float,
        'audatex_Parts_Labour_Gross': float,
        'audatex_Paint_Labour_Hours': float,
        'audatex_Paint_Labour_Rate': float,
        'audatex_Paint_Labour_Net': float,
        'audatex_Paint_Labour_Gross': float,
        'audatex_Parts_Material_Net': float,
        'audatex_Parts_Material_Gross': float,
        'audatex_Paint_Material_Net': float,
        'audatex_Paint_Material_Gross': float,
        'audatex_Additional_Net': float,
        'audatex_Additional_Gross': float,
        'audatex_Total_Repair_Net': float,
        'audatex_Total_Repair_Gross': float,
        'audatex_Replace_Operations': float,
        'audatex_Repair_Operations': float
    }

    read_dtypes = dict(read_dtypes,
                       **dam_dict,
                       **sev_dict,
                       **audatex_dict)

    return {
        'header': 0,
        'sep': ',',
        'delim_whitespace': False,
        'parse_dates': ['cla_accident_datetime', 'cla_notification_datetime'],
        'dtype': read_dtypes,
        'date_parser': datestr_to_datetime,
        'usecols': read_dtypes.keys(),
        "encoding": "utf-8",
        "engine": "c"
    }


def csv_kwargs():
    """Provides kwargs for pandas.read_csv"""
    read_dtypes = {
        'Id': int,
        'MSSubClass': int,
        'MSZoning': str,
        'LotFrontage': int,
        'LotArea': int,
        'Street': ,
        'Alley',
        'LotShape',
        'LandContour',
        'Utilities',
        'LotConfig',
        'LandSlope',
        'Neighborhood',
        'Condition1',
        'Condition2',
        'BldgType',
        'HouseStyle',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'RoofStyle',
        'RoofMatl',
        'Exterior1st',
        'Exterior2nd',
        'MasVnrType',
        'MasVnrArea',
        'ExterQual',
        'ExterCond',
        'Foundation',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinSF1',
        'BsmtFinType2',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        'Heating',
        'HeatingQC',
        'CentralAir',
        'Electrical',
        '1stFlrSF',
        '2ndFlrSF',
        'LowQualFinSF',
        'GrLivArea',
        'BsmtFullBath',
        'BsmtHalfBath',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'KitchenQual',
        'TotRmsAbvGrd',
        'Functional',
        'Fireplaces',
        'FireplaceQu',
        'GarageType',
        'GarageYrBlt',
        'GarageFinish',
        'GarageCars',
        'GarageArea',
        'GarageQual',
        'GarageCond',
        'PavedDrive',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        '3SsnPorch',
        'ScreenPorch',
        'PoolArea',
        'PoolQC',
        'Fence',
        'MiscFeature',
        'MiscVal',
        'MoSold',
        'YrSold',
        'SaleType',
        'SaleCondition',
        'SalePrice'
        }

    read_dtypes = dict(read_dtypes)

    return {
        'header': 0,
        'sep': ',',
        'delim_whitespace': False,
        'dtype': read_dtypes,
        'usecols': read_dtypes.keys(),
        "encoding": "utf-8",
        "engine": "c"
    }

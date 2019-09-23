#!/usr/bin/env python

"""
Module containing useful helper functions for the VRC
Autoestimation project
"""

import datetime

import numpy as np
import pandas as pd

from .pipeline_parameters import Months


def get_month(x):
    return x.month


def get_day(x):
    return x.day


def get_days(x):
    return x.days


def get_year(x):
    return x.year


def one_day_more(x):
    days = get_days(x)
    if days == 0:
        return 0
    elif days == -1:
        return -1
    else:
        return 1


def get_weekday(x):
    return x.weekday()


def is_weekend(x):
    day = x.weekday()
    if day == 5 or day == 6:
        return 1
    else:
        return 0


def get_time_of_day(x):
    hr = x.hour
    if hr in [0, 1, 2, 3, 4, 5]:
        return 0
    elif hr in [6, 7, 8, 9, 10, 11]:
        return 1
    elif hr in [12, 13, 14, 15, 16, 17]:
        return 2
    else:
        return 3


def get_season(x):
    months = Months()
    month = get_month(x)
    if month in months.winter:
        return 0
    elif month in months.spring:
        return 1
    elif month in months.summer:
        return 2
    elif month in months.autumn:
        return 3


def MANUAL2Manual(x):
    if x == 'MANUAL':
        return 'Manual'
    else:
        return x


def description_imputer(x):
    if not isinstance(x, str):
        x = ''
    return x


def get_hour(x):
    return x.hour


def loss_cause_groups(cla_loss_cause):
    if cla_loss_cause in ["Bridge Strike",
                          "Defective Workmanship (Customer Vehicle)",
                          "Autonomous System Failure",
                          "Insured Vehicle Jack-Knifed",
                          "Mechanical Failure/Blow Out (Insured Vehicle)",
                          "Insured Vehicle Overturned"]:
        calc_cla_loss_cause = "Misc serious"
    elif cla_loss_cause in ["Insured Hit Cyclist",
                            "Insured Hit Pedestrian",
                            "Insured Hit Pothole/Object In Road/Animal"]:
        calc_cla_loss_cause = "Insured hit something minor"
    elif cla_loss_cause in ["Object Fell From Insured Vehicle",
                            "Object Fell From Third Party Vehicle",
                            "Falling Object"]:
        calc_cla_loss_cause = "Falling Object"
    elif cla_loss_cause in ["Third Party Turned Right, Insured Overtaking",
                            "Insured Turned Right, Third Party Overtaking"]:
        calc_cla_loss_cause = "Turned right, other vehicle overtaking"
    elif cla_loss_cause in ["Not Otherwise Catered For",
                            "Not Yet Known"]:
        calc_cla_loss_cause = "Not Yet Known"
    else:
        calc_cla_loss_cause = cla_loss_cause
    return calc_cla_loss_cause


def vehicle_colour_groups(veh_vehicle_colour):
    if veh_vehicle_colour in ['DARK BLUE', 'NAVY BLUE', 'LIGHT BLUE', ' NAVY']:
        calc_vehicle_colour = 'BLUE'
    elif veh_vehicle_colour in ['METALLIC GREY', 'DARK GREY', 'GUN METAL GREY']:
        calc_vehicle_colour = 'GREY'
    elif veh_vehicle_colour in ['MAROON', 'COFFEE']:
        calc_vehicle_colour = 'BROWN'
    else:
        calc_vehicle_colour = veh_vehicle_colour
    return calc_vehicle_colour


def dam_area_2_nulls(x):
    if x is np.nan:
        return 0.0
    else:
        return 1.0

def wheelbase_groups(wheelbase_length):
    if wheelbase_length in ["LWB"]:
        calc_wheelbase_length = "LONG WHEELBASE"
    elif wheelbase_length in ["MWB"]:
        calc_wheelbase_length = "MEDIUM WHEELBASE"
    elif wheelbase_length in ["SWB"]:
        calc_wheelbase_length = "SHORT WHEELBASE"
    else:
        calc_wheelbase_length = wheelbase_length
    return calc_wheelbase_length


#########################################################
########################### END #########################
#########################################################

# def cutter(entry_df):
#     one = entry_df[(entry_df['audatex_Parts_Labour_Hours'] > 0) &
#                    (entry_df['audatex_Paint_Labour_Hours'] > 0) &
#                    (entry_df['audatex_Paint_Material_Net'] > 0) &
#                    (entry_df['audatex_Parts_Material_Net'] > 0)]
#     two = entry_df[(entry_df['audatex_Parts_Labour_Hours'] > 0) &
#                    ((entry_df['audatex_Paint_Labour_Hours'] == 0) &
#                     (entry_df['audatex_Paint_Material_Net'] == 0)) &
#                    (entry_df['audatex_Parts_Material_Net'] > 0)]
#     three = entry_df[(entry_df['audatex_Paint_Labour_Hours'] > 0) &
#                      ((entry_df['audatex_Parts_Labour_Hours'] == 0) &
#                       (entry_df['audatex_Parts_Material_Net'] == 0)) &
#                      (entry_df['audatex_Paint_Material_Net'] > 0)]
#     frames = [one, two, three]
#     new_df = pd.concat(frames)
#     return new_df

# def fill_sevs(data):
#     sev_keys = ['Airbag', 'Bonnet', 'Boot', 'BurntOut', 'ElectricBattery',
#                 'Engine', 'Frontal', 'FrontSuspension', 'Fuel_Tank',
#                 'Interior', 'Lock', 'Left_Handle', 'LSB', 'Mechanical',
#                 'NSF', 'NSFDoor', 'NSMirror', 'NSR', 'NSRDoor', 'NSS', 'OSF',
#                 'OSFDoor', 'OSMirror', 'OSR', 'OSRDoor', 'OSS', 'Other',
#                 'Rear', 'RearDoor', 'RearSuspension', 'Roof', 'Right_Handle',
#                 'RSB', 'Seat', 'SideGlass', 'Underside', 'Immobile', 'Rolled',
#                 'Waterlogged', 'Wheel', 'Windscreen']
#     sev_keys = ['veh_DAM_SEV_' + x for x in sev_keys]
#     data[sev_keys] = data[sev_keys].fillna(0)
#     return data

# def surf_area(x):
#     return 2 * (x[0] * x[1] + x[1] * x[2] + x[0] * x[2])


# def mili_to_m(x):
#     return x/1000

# def surf_area(x):
#     return 2 * (x[0] * x[1] + x[1] * x[2] + x[0] * x[2])

# def upper(x):
#     return str(x).upper()

# def postcode_to_region(x):
#     address_regions = AddressRegions()
#     if x in address_regions.scotland:
#         return 'SCOTLAND'
#     elif x in address_regions.northern_ireland:
#         return 'NI'
#     elif x in address_regions.wales:
#         return 'WALES'
#     elif x in address_regions.london:
#         return 'LONDON'
#     elif x in address_regions.north_east:
#         return 'NORTHEAST'
#     elif x in address_regions.north_west:
#         return 'NORTHWEST'
#     elif x in address_regions.east_mid:
#         return 'EASTMID'
#     elif x in address_regions.west_mid:
#         return 'WESTMID'
#     elif x in address_regions.south_west:
#         return 'SOUTHWEST'
#     elif x in address_regions.south_east:
#         return 'SOUTHEAST'

# def clean_market_segments(x):
#     if  isinstance(x, str):
#         if x.startswith('LCVs'):
#             return 'LCVs'
#         else:
#             return x
#     else:
#         return x


# def fill_glass_veh(data, col, col1):
#     if data[col].dtype == np.float64 or data[col].dtype == np.int64:
#         for idx in data[col][data[col].isnull()].index.tolist():
#             if not math.isnan(data[col1].loc[idx]):
#                 data.loc[idx, col] = data[col1].loc[idx]
#     else:
#         data[col] = data[col].map(str)
#         data[col1] = data[col1].map(str)
#         for idx in data[col][data[col].isnull()].index.tolist():
#             if data[col1].loc[idx] != 'nan':
#                 data.loc[idx, col] = data[col1].loc[idx]
#     return data

# def column_null_filler(df, col1, col2):
#     df[col1].fillna(value=df[col2], inplace = True)
#     return df

# #def get_month(data):
# #    data['calc_month'] = data['cla_notification_datetime'].astype(str).str[5:]
# #    data['calc_month'] = data['calc_month'].astype(str).str[:2]
# #    data['calc_month'] = data['calc_month'].astype(int)
# #    data['calc_month'] = data['calc_month'].apply(lambda x: calendar.month_name[x])
# #    data = pd.get_dummies(data, columns=['calc_month'])
# #    return data


# def add_time_to_notify(data):
#     not_date = 'cla_notification_datetime'
#     acc_date = 'cla_accident_datetime'
#     ttn = 'calc_time_to_notify'
#     data[not_date] = data[not_date].astype(str).str[:9]
#     data[not_date] = pd.to_datetime(data[not_date])
#     data[acc_date] = data[acc_date].astype(str).str[:9]
#     data[acc_date] = pd.to_datetime(data[acc_date])
#     data[ttn] = data[not_date] - data[acc_date]
#     data[ttn] = data[ttn].astype('timedelta64[h]')
#     data['calc_time_to_notify'] = data[ttn]/24
#     return data


# def get_month_whole(data):
#     mon = 'calc_month'
#     data[mon] = data['cla_notification_datetime'].astype(str).str[5:]
#     data[mon] = data[mon].astype(str).str[:2]
#     data[mon] = data[mon].astype(int)
#     data[mon] = data[mon].apply(lambda x: calendar.month_name[x])
#     data = pd.get_dummies(data, columns=[mon])
#     return data


# def replace_numbers(string):
#     a = str(string).replace('0', '').replace('1', '').replace('2', '')
#     a = a.replace('3', '').replace('4', '').replace('5', '')
#     a = a.replace('6', '').replace('7', '').replace('8', '').replace('9', '')
#     return a


# def postcode_to_areacode(x):
#     h, sep, tail = str(x).partition(' ')
#     h = replace_numbers(h)
#     return str(h)


# def region_grouper(area):
#         if area in ['AB', 'DG', 'ML', 'G', 'IV', 'DD', 'KA',
#                     'KY', 'FK', 'PH', 'KW', 'EH', 'PA',
#                     'TD', 'ZE', 'HS']:
#             calc_regions = 'SCOTLAND'
#         elif area == 'BT':
#             calc_regions = 'NI'
#         elif area in ['CF', 'SA', 'NP', 'LL', 'LD', 'SY']:
#             calc_regions = 'WALES'
#         elif area in ['BR', 'RM', 'SM', 'EN', 'UB', 'KT',
#                       'WD', 'NW', 'DA', 'EC', 'TW', 'IG',
#                       'WC', 'CR', 'SE', 'SW', 'HA', 'W',
#                       'N', 'E']:
#             calc_regions = 'LONDON'
#         elif area in ['DH', 'SR', 'HU', 'YO', 'NE',
#                       'HG', 'WF', 'DL', 'TS', 'LS']:
#             calc_regions = 'NORTHEAST'
#         elif area in ['BB', 'LA', 'CA', 'PR', 'FY', 'WN',
#                       'L', 'BL', 'OL', 'CW', 'WA', 'HX',
#                       'BD', 'M', 'CH', 'SK', 'HD']:
#             calc_regions = 'NORTHWEST'
#         elif area in ['CB', 'NG', 'DN', 'S', 'LN', 'DE',
#                       'PE', 'LE', 'CO', 'NR', 'IP']:
#             calc_regions = 'EASTMID'
#         elif area in ['B', 'TF', 'HR', 'WV', 'ST', 'DY',
#                       'WS', 'CV', 'WR', 'NN']:
#             calc_regions = 'WESTMID'
#         elif area in ['BA', 'SN', 'DT', 'TQ', 'PL', 'BS',
#                       'TA', 'TR', 'BH', 'SP', 'EX', 'GL']:
#             calc_regions = 'SOUTHWEST'
#         elif area in ['AL', 'PO', 'CT', 'SG', 'LU', 'SS',
#                       'OX', 'CM', 'RH', 'HP', 'SO', 'MK',
#                       'BN', 'RG', 'GU', 'SL', 'ME', 'TN', 'LO']:
#             calc_regions = 'SOUTHEAST'
#         elif area in ['GY', 'JE', 'IM']:
#             calc_regions = 'CROWNDEPENDENCIES'
#         else:
#             calc_regions = 'UNKNOWN'
#         return calc_regions

# def funcI(x):
#     return x.sum()
        
# def funcII(x):
#     return x.prod()
    
# def lower(x):
#     return str(x).lower()

# def vehicle_type_groups(veh_vehicle_category):
#     if veh_vehicle_category in ['Estate Car/Station Wagon','5 Door Estate','4 Door Estate','3 Door Estate']:
#         calc_veh_vehicle_category = 'Estate'
#     elif veh_vehicle_category in ['Tipper','Hearse','Truck - Flat','Multi Seater','Hearse','Pick-up','Light Van','Pickup - Double Cab','Jeep'
#                                   ,'Box Van','Pickup - Double Cab','Truck - Dropside','Sports Car','Refrigerated/Insulated','Station Wagon'
#                                   ,'Pickup - Single Cab','Luton Van','Truck - Curtainside']:
#         calc_veh_vehicle_category ='Van'
#     elif veh_vehicle_category in ['nan','No Code Available']:
#         calc_veh_vehicle_category = 'Unknown'
#     elif veh_vehicle_category in ['5 Door Hatchback','3 Door Hatchbacl']:
#         calc_veh_vehicle_category ='Hatchback'
#     elif veh_vehicle_category in ['Convertible/Cabriolet','2 Door Convertible','Caravanette','Convertible Sport']:
#         calc_veh_vehicle_category = 'Convertible'
#     elif veh_vehicle_category in ['4 Door Saloon','2 Door Saloon']:
#         calc_veh_vehicle_category = 'Saloon' 
#     else: 
#         calc_veh_vehicle_category = veh_vehicle_category 
#     return calc_veh_vehicle_category

# def calc(data):
#     wpdc = 'calc_weight/dam_count'
#     data[wpdc] = data['veh_weight'] / data['veh_damage_count']
#     return data

# def vol_will(x):
#     return x.values[0] * x.values[1] * x.values[2] *  1e-9

# def mode(x):
#     return scipy.stats.mode(x).mode[0]

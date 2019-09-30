#!/usr/bin/env python


class Months():
    def __init__(self):
        self.winter = [12, 1, 2]
        self.spring = [3, 4, 5]
        self.summer = [6, 7, 8]
        self.autumn = [9, 10, 11]


class AddressRegions():
    def __init__(self):
        self.scotland = ['AB', 'DG', 'ML', 'G', 'IV', 'DD',
                         'KY', 'FK', 'PH', 'KW', 'EH', 'PA',
                         'TD', 'ZE']
        self.northern_ireland = ['BT']
        self.wales = ['CF', 'SA', 'NP', 'LL', 'LD', 'SY']
        self.london = ['BR', 'RM', 'SM', 'EN', 'UB', 'KT',
                       'WD', 'NW', 'DA', 'EC', 'TW', 'IG',
                       'WC', 'CR', 'SE', 'SW', 'HA', 'W',
                       'N']
        self.north_east = ['DH', 'SR', 'HU', 'YO', 'NE',
                           'HG', 'WF', 'DL', 'TS', 'LS']
        self.north_west = ['BB', 'LA', 'CA', 'PR', 'FY', 'WN',
                           'L', 'BL', 'OL', 'CW', 'WA', 'HX',
                           'BD', 'M', 'CH', 'SK', 'HD']
        self.east_mid = ['CB', 'NG', 'DN', 'S', 'LN', 'DE',
                         'PE', 'LE', 'CO', 'NR', 'IP']
        self.west_mid = ['B', 'TF', 'HR', 'WV', 'ST', 'DY',
                         'WS', 'CV', 'WR', 'NN']
        self.south_west = ['BA', 'SN', 'DT', 'TQ', 'PL', 'BS',
                           'TA', 'TR', 'BH', 'SP', 'EX', 'GL']
        self.south_east = ['AL', 'PO', 'CT', 'SG', 'LU', 'SS',
                           'OX', 'CM', 'RH', 'HP', 'SO', 'MK',
                           'BN', 'RG', 'GU', 'SL', 'ME', 'TN']
        

def vrc_dam_keys():
    # return ['Airbag_Deployed', 'Bonnet', 'Boot', 'BurntOut',
    #         'ElectricBattery', 'Engine', 'Frontal', 'FrontSuspension',
    #         'Fuel_Tank', 'Interior', 'Lock', 'Left_handle', 'LSB',
    #         'Mechanical', 'NSF', 'NSFDoor', 'NSMirror', 'NSR', 'NSRDoor',
    #         'NSSide', 'OSF', 'OSFDoor', 'OSMirror', 'OSR', 'OSRDoor',
    #         'OSSide', 'Other', 'Rear', 'RearDoor', 'RearSuspension', 'Roof',
    #         'Right_Handle', 'RSB', 'Seat', 'SideGlass', 'Underside',
    #         'Immobile', 'Rolled', 'Waterlogged', 'Wheel', 'Windscreen']
    return ['Frontal', 'NSF', 'NSR', 'OSF', 'OSR', 'Rear', 'NSSide', 'RearDoor']


def vrc_sev_keys():
    # return ['Airbag', 'Bonnet', 'Boot', 'BurntOut', 'ElectricBattery',
    #         'Engine', 'Frontal', 'FrontSuspension', 'Fuel_Tank',
    #         'Interior', 'Lock', 'Left_Handle', 'LSB', 'Mechanical',
    #         'NSF', 'NSFDoor', 'NSMirror', 'NSR', 'NSRDoor', 'NSS', 'OSF',
    #         'OSFDoor', 'OSMirror', 'OSR', 'OSRDoor', 'OSS', 'Other', 'Rear',
    #         'RearDoor', 'RearSuspension', 'Roof', 'Right_Handle', 'RSB',
    #         'Seat', 'SideGlass', 'Underside', 'Immobile', 'Rolled',
    #         'Waterlogged', 'Wheel', 'Windscreen']
    return ['Bonnet', 'Boot', 'Frontal', 'NSF', 'NSFDoor', 'NSR',
            'NSRDoor', 'OSF', 'OSFDoor', 'OSMirror', 'OSR', 'OSRDoor',
            'Rear', 'Immobile', 'Wheel', 'NSMirror', 'NSS', 'OSS', 'RearDoor']
    
    

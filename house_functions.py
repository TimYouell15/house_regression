#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:44:16 2019

@author: tyouell
"""

def ZoneTransformer(zone):
    if zone in ['C (all)']:
        ms_zoning = 'ALL'
    else:
        ms_zoning = zone
    return ms_zoning

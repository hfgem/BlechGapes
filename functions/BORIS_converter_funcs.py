#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:36:14 2024

@author: Hannah Germaine

A file for functions related to the BORIS converter
"""

import numpy as np

def data_index_calcs(csv_headers):
    media_dur_index = -1
    behavior_index = -1
    modifier_index = -1
    behavior_type_index = -1
    behavior_time_index = -1
    for bt in range(len(csv_headers)):
        if csv_headers[bt] == 'Media duration (s)':
            media_dur_index = bt
        if csv_headers[bt] == 'Behavior':
            behavior_index = bt
        if csv_headers[bt] == 'Modifier #1':
            modifier_index = bt
        if csv_headers[bt] == 'Behavior type':
            behavior_type_index = bt
        if csv_headers[bt] == 'Time':
            behavior_time_index = bt
    
    return media_dur_index, behavior_index, modifier_index, behavior_type_index, behavior_time_index

def data_exists_check(behavior_index,behavior_type_index,behavior_time_index):
    """This function takes in index calculations and determines if the data exists"""
    check = 1
    if behavior_index == -1: #Error check 1
        check = 0
    if behavior_type_index == -1: #Error check 1
        check = 0
    if behavior_time_index == -1: #Error check 1
        check = 0

    return check

def reformat_data(csv_data_list, media_dur_index, behavior_index, modifier_index, 
                  behavior_type_index, behavior_time_index):
    media_durations = []
    behavior_names = []
    behavior_start_times = []
    behavior_end_times = []
    for d_i in np.arange(1,len(csv_data_list)):
        media_durations.extend([float(csv_data_list[d_i][media_dur_index])])
        behavior_name = csv_data_list[d_i][behavior_index]
        if behavior_name == 'mouth or tongue movement':
            behavior_name = csv_data_list[d_i][modifier_index]
        if csv_data_list[d_i][behavior_type_index] == 'START':
            behavior_names.extend([behavior_name])
            behavior_start_times.extend([1000*(float(csv_data_list[d_i][behavior_time_index]))])
        if csv_data_list[d_i][behavior_type_index] == 'STOP':
            behavior_end_times.extend([1000*(float(csv_data_list[d_i][behavior_time_index]))])
            
    return media_durations, behavior_names, behavior_start_times, behavior_end_times

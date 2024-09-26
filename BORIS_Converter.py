#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:07:40 2024

@author: Hannah Germaine

This script is designed to reformat the scoring outputs from the BORIS behavioral
scoring software. Outputs are .csv files that need to be reformatted into
arrays by movement type of onset and offset surrounding taste delivery.
"""

import sys
import os
import csv
import easygui
import numpy as np
import functions.BORIS_converter_funcs as bcf

#Have the user first select the folder where the BORIS output files exist
print("Please select the folder where the BORIS data is stored.")
data_dir = easygui.diropenbox(title='Please select the folder where data is stored.')

#Create a save folder for reformatted data with the same naming and location as original data
new_save_dir = os.path.join(os.path.split(data_dir)[0],os.path.split(data_dir)[1] + '_reformatted')
if not os.path.isdir(new_save_dir):
    os.mkdir(new_save_dir)

#Sort through all the original data files in the folder, keeping only .csv files
data_file_list = os.listdir(data_dir)
csv_file_list = []
for df in data_file_list:
    if df[-4:] == '.csv':
        csv_file_list.append(df)
del df
        
#Now run through each .csv file and reformat
reformat_data = dict()
all_behavior_names = []
all_taste_names = []
all_trial_inds = []
all_media_durations = []
for csv_ind, csv_name in enumerate(csv_file_list):
    #Split name into components
    animal_name = csv_name.split('_')[0]
    taste_name = csv_name.split('_')[-2]
    trial_index = (csv_name.split('_')[-1]).split('.')[0]
    
    #Import rows of .csv
    with open(os.path.join(data_dir,csv_name),newline='') as f:
        reader = csv.reader(f)
        csv_data_list = list(reader)
    
    #csv_data_list will have the first sublist == header names
    csv_headers = csv_data_list[0]
    
    #Pull out indices of data of interest
    media_dur_index, behavior_index, modifier_index, behavior_type_index, behavior_time_index = \
        bcf.data_index_calcs(csv_headers)
    check1 = bcf.data_exists_check(behavior_index,behavior_type_index,behavior_time_index)
    if check1 == 0:
        print("ERROR: Necessary headers not found in .csv file " + csv_name)
        print("Please ensure you're in the correct folder and only have BORIS outputs stored there.")
        quit()
        
    #Now reformat data!
    media_durations, behavior_names, behavior_start_times, behavior_end_times = \
        bcf.reformat_data(csv_data_list, media_dur_index, behavior_index, \
                          modifier_index, behavior_type_index, behavior_time_index)
    
    #Store reformatted data in dictionary
    reformat_data[csv_ind] = dict()
    reformat_data[csv_ind]['animal_name'] = animal_name
    reformat_data[csv_ind]['taste'] = taste_name
    reformat_data[csv_ind]['trial_index'] = trial_index
    reformat_data[csv_ind]['behavior_names'] = behavior_names
    reformat_data[csv_ind]['behavior_start_times'] = behavior_start_times #In ms from taste delivery
    reformat_data[csv_ind]['behavior_end_times'] = behavior_end_times #In ms from taste delivery
    
    #Store all experiment properties
    all_media_durations.extend(media_durations)
    all_behavior_names.extend(behavior_names)
    all_taste_names.extend([taste_name])
    all_trial_inds.extend([int(trial_index)])
    
    del animal_name, taste_name, trial_index, reader, csv_data_list, csv_headers, \
        behavior_index, modifier_index, behavior_type_index, behavior_time_index, \
            behavior_names, behavior_start_times, behavior_end_times
del csv_ind, csv_name
   
#Now that we have organized data in a dict() let's save individual arrays for
#different behavior types and tastes - all binary
unique_behaviors = np.unique(all_behavior_names)
unique_behaviors = [str(i) for i in unique_behaviors]
unique_tastes = np.unique(all_taste_names)
unique_tastes = [str(i) for i in unique_tastes]
max_trials = np.max(all_trial_inds)
pre_taste = 2000
post_taste = np.max(all_media_durations).astype('int')*1000

for ut in unique_tastes:
    ut_combined = ('_').join(ut.split(' '))
    for ub in unique_behaviors:
        ub_combined = ('_').join(ub.split(' '))
        exec(ub_combined + '_' + ut_combined + ' = np.zeros((max_trials,pre_taste+post_taste))')
        
for d_i in range(len(reformat_data)):
    t = reformat_data[d_i]['taste']
    t_combined = ('_').join(t.split(' '))
    b_list = reformat_data[d_i]['behavior_names']
    b_starts = reformat_data[d_i]['behavior_start_times']
    b_stops = reformat_data[d_i]['behavior_end_times']
    trial_ind = int(reformat_data[d_i]['trial_index'])
    num_behaviors = len(b_list)
    for b_i in range(num_behaviors):
        b = b_list[b_i]
        b_combined = ('_').join(b.split(' '))
        b_start = int(np.floor(b_starts[b_i])) + 2000
        b_stop = int(np.ceil(b_stops[b_i])) + 2000
        exec(b_combined + '_' + t_combined + '[' + str(trial_ind) + ',' + \
             str(b_start) + ':' + str(b_stop) + ']' + ' = np.ones(' + \
             str(b_stop) + '-' + str(b_start) + ')')
                    
for ut in unique_tastes:
    ut_combined = ('_').join(ut.split(' '))
    for ub in unique_behaviors:
        ub_combined = ('_').join(ub.split(' '))
        data_array = exec(ub_combined + '_' + ut_combined)
        np.save(os.path.join(new_save_dir,full_name + '.npy'),full_name)

    
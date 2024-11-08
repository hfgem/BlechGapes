#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:38:11 2024

@author: cmazzio + hfgem
"""

import os
import csv
import tables
import json
import numpy as np

def bool_input(prompt):
	"""This function asks a user for a boolean response to a prompt
    and returns either y or n"""
	bool_loop = 1	
	while bool_loop == 1:
		print("Respond with Y/y/N/n:")
		response = input(prompt)
		if (response.lower() != 'y')*(response.lower() != 'n'):
			print("\tERROR: Incorrect data entry, only give Y/y/N/n.")
		else:
			bool_val = response.lower()
			bool_loop = 0
	
	return bool_val

def import_held_units_array(held_save_dir):
    """This function imports held units into an array using the
    held units csv file directory and user input. It returns
    an array of held units, the number of units, and the number of days."""
    file_list = os.listdir(held_save_dir)
    csv_name = ''
    for files in file_list:
        if files[-3:] == 'csv':
            bool_result = bool_input("Is " + files + " the correct held units csv file? ")
            if bool_result == 'y':
                csv_name = files
                break
    held_units = []
    with open(os.path.join(held_save_dir,csv_name), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            held_units.append(row)
    held_units = np.array(held_units)
    num_units, num_days = np.shape(held_units)
    
    return held_units, num_units, num_days

def import_held_units_dict(held_save_dir, num_days):
    """This function imports the directories where the different days of
    held units were pulled from and returns lists of those directories, the
    names of the hdf5_files, and the animal names"""
    
    data_dict_dir = os.path.join(held_save_dir,'data_dict.npy')
    data_dict = np.load(data_dict_dir,allow_pickle=True).item()
    directories = []
    hdf5_names = []
    animal_names = []
    for nd in range(num_days):
        dirname = data_dict[nd]['dir_name']
        directories.extend([dirname])
        hdf5_names.extend([data_dict[nd]['hdf5_name']])
        animal_names.extend([dirname.split('/')[-2]])
    #The animal names should be all the same so we'll just keep 1
    return directories, hdf5_names, animal_names[0]
        
def import_spike_times_dict(hdf5_names,directories):
    """This function loads spike time data for each day of held unit
    analyses and separates into a dictionary by taste with further info."""
    spike_times_dict = dict()
    for ind, hdf5_name in enumerate(hdf5_names):
        hf5 = tables.open_file(os.path.join(directories[ind],hdf5_name), 'r')
        num_neur = len(hf5.root.unit_descriptor[:])
        #Load dig in names from info file
        info_file_name = hdf5_name.split('_repacked.h5')[0]
        with open(os.path.join(directories[ind],info_file_name + '.info'),'r') as info_file:
            info_file = json.load(info_file)
        taste_names = info_file['taste_params']['tastes']
        dig_in_filenames = info_file['taste_params']['filenames']
        delivery_counts = info_file['taste_params']['trial_count']
        dig_in_inds = [fname.split('.dat')[0][-2:] for fname in dig_in_filenames]
        #Load dig in times from hdf5
        all_spike_trains = []
        for t_i in range(len(taste_names)):
            all_spike_trains.append(exec("hf5.root.spike_trains.dig_in_" + dig_in_inds[t_i] + ".spike_array[:]"))
        hf5.close()
        #Store to dictionary
        short_day_name = ('_').join(hdf5_name.split('_')[:2])
        spike_times_dict[short_day_name] = dict()
        spike_times_dict[short_day_name]['num_neur'] = num_neur
        spike_times_dict[short_day_name]['taste_names'] = taste_names
        spike_times_dict[short_day_name]['delivery_counts'] = delivery_counts
        spike_times_dict[short_day_name]['pre_time'] = 2000
        spike_times_dict[short_day_name]['post_time'] = 5000
        for t_i, t_name in enumerate(taste_names):
            spike_times_dict[short_day_name][t_name] = all_spike_trains[t_i]

    return spike_times_dict
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:42:34 2024

@author: hannahgermaine
Test code to try to pull out different mouth movements from emg data
"""


import sys, pickle, easygui, os
sys.path.append('/home/cmazzio/Desktop/blech_clust/')
sys.path.append('/home/cmazzio/Desktop/pytau/')
from matplotlib import cm
from pytau.changepoint_io import FitHandler
import pylab as plt
from pytau.utils import plotting
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from pytau.utils.ephys_data import EphysData
from scipy import stats
from pytau.changepoint_io import DatabaseHandler
fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()
# Get fits for a particular experiment
dframe = fit_database.fit_database
from pytau.changepoint_analysis import PklHandler

#%% Function definition

def int_input(prompt):
    #This function asks a user for an integer input
    int_loop = 1    
    while int_loop == 1:
        response = input(prompt)
        try:
            int_val = int(response)
            int_loop = 0
        except:
            print("\tERROR: Incorrect data entry, please input an integer.")
    
    return int_val

def bool_input(prompt):
    #This function asks a user for an integer input
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

#%% Load all files to be analyzed

desired_states = 4 #which number of states dataset to use

#Prompt user for the number of datasets needed in the analysis
num_files = int_input("How many data files do you need to import for this analysis (integer value)? ")
if num_files >= 1:
    print("Multiple file import selected.")
else:
    print("Single file import selected.")

#Pull all data into a dictionary
emg_data_dict = dict()
for nf in range(num_files):
    #Directory selection
    print("Please select the folder where the data # " + str(nf+1) + " is stored.")
    data_dir = easygui.diropenbox(title='Please select the folder where data is stored.')
    given_name = input("\tHow would you rename " + data_dir.split('/')[-1] + "? ")
    #Import associated emg data
    try:
        emg_filt = np.load(os.path.join(data_dir,'emg_output','emg','emg_filt.npy'))
        emg_data_dict[nf]['emg_filt'] = emg_filt
        print("\tEmg filtered data successfully imported for dataset.")
    except:
        print("\tEmg filtered data not imported successfully.")
        bool_val = bool_input("\tIs there an emg_filt.npy file to import?")
        if bool_val == 'y':
            emg_filt_data_dir = easygui.diropenbox(title='\tPlease select the folder where emg_filt.npy is stored.')
            emg_filt = np.load(os.path.join(emg_filt_data_dir,'emg_filt.npy'))
            emg_data_dict[nf]['emg_filt'] = emg_filt
        else:
            print("\tMoving On.")
    #Import associated gapes
    # print("\tNow import associated first gapes data with this dataset.")
    # gape_data_dir = easygui.diropenbox(title='\tPlease select the folder where data is stored.')
    # #Search for matching file type - ends in _gapes.npy
    # files_in_dir = os.listdir(gape_data_dir)
    # print("There are " + str(len(files_in_dir)) + " gape files in this folder.")
    # for filename in files_in_dir:
    #     if filename[-10:] == '_gapes.npy':
    #         bool_val = bool_input("\tIs " + filename + " the correct associated file with " + given_name + "?")
    #         if bool_val == 'y':
    #             first_gapes =  np.load(os.path.join(gape_data_dir,filename))
    #             emg_data_dict[nf]['first_gapes'] = first_gapes
    #             break
    # try: #Check that something was imported
    #     first_gapes = emg_data_dict[nf]['first_gapes']
    # except:
    #     print('First gapes file not found/selected in given folder. Did you run gape_onset.py before?')
    #     print('You may want to quit this program now - it will break in later code blocks by missing this data.')
    
#Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')

#Save dictionary
dict_save_dir = os.path.join(results_dir,'emg_data_dict.pkl')
f = open(dict_save_dir,"wb")
pickle.dump(emg_data_dict,f)

#%% Take each emg signal and break it into individual groups



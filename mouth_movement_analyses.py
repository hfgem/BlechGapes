#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:42:34 2024

@author: hannahgermaine
Test code to try to pull out different mouth movements from emg data
"""


import sys, pickle, easygui, os
sys.path.append('/home/cmazzio/Desktop/blech_clust/')
from matplotlib import cm
import pylab as plt
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from scipy import stats
from scipy.signal import find_peaks, peak_widths

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
    emg_data_dict[nf] = dict()
    emg_data_dict[nf]['given_name'] = given_name
    #Import associated raw emg data
    try:
        emg_filt = np.load(os.path.join(data_dir,'emg_output','emg','emg_filt.npy')) #(num_tastes,num_trials,time) #2000 pre, 5000 post
        [num_tastes, max_num_trials, num_time] = np.shape(emg_filt)
        emg_data_dict[nf]['emg_filt'] = emg_filt
        emg_data_dict[nf]['num_tastes'] = num_tastes
        emg_data_dict[nf]['max_num_trials'] = max_num_trials
        emg_data_dict[nf]['num_time'] = num_time
        taste_names = []
        for t_i in range(num_tastes):
            taste_names.append(input("What is the name of taste " + str(t_i+1) + ": "))
        emg_data_dict[nf]['taste_names'] = taste_names
        print("\tEmg filtered data successfully imported for dataset.")
    except:
        print("\tEmg filtered data not imported successfully.")
        bool_val = bool_input("\tIs there an emg_filt.npy file to import?")
        if bool_val == 'y':
            emg_filt_data_dir = easygui.diropenbox(title='\tPlease select the folder where emg_filt.npy is stored.')
            emg_filt = np.load(os.path.join(emg_filt_data_dir,'emg_filt.npy'))
            [num_tastes, max_num_trials, num_time] = np.shape(emg_filt)
            emg_data_dict[nf]['emg_filt'] = emg_filt
            emg_data_dict[nf]['num_tastes'] = num_tastes
            emg_data_dict[nf]['max_num_trials'] = max_num_trials
            emg_data_dict[nf]['num_time'] = num_time
            taste_names = []
            for t_i in range(num_tastes):
                taste_names.append(input("What is the name of taste " + str(t_i+1)+ ": "))
            emg_data_dict[nf]['taste_names'] = taste_names
        else:
            print("\tMoving On.")
	#Import associated enveloped emg data
    try:
        env = np.load(os.path.join(data_dir,'emg_output','emg','emg_env.npy')) #(num_tastes,num_trials,time) #2000 pre, 5000 post
        emg_data_dict[nf]['env'] = env
        print("\tEnveloped data successfully imported for dataset.")
    except:
        print("\tEnveloped data not imported successfully.")
        bool_val = bool_input("\tIs there an env.npy file to import?")
        if bool_val == 'y':
            env_data_dir = easygui.diropenbox(title='\tPlease select the folder where env.npy is stored.')
            env = np.load(os.path.join(emg_filt_data_dir,'emg_env.npy'))
            emg_data_dict[nf]['env'] = env
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

#emg_data_dict[dataset_num] contains a dictionary with keys "emg_filt" and "given_name"
#given_name is a string of the dataset name
#emg_filt is an array of size [num_tastes,num_trials,7000 ms] where 7000 ms is 2000 pre and 5000 post

pre_taste = 2000
post_taste = 5000
min_inter_peak_dist = 10 #min ms apart peaks have to be to count
min_gape_band = 4
max_gape_band = 6

for nf in range(len(emg_data_dict)):
	print("Analyzing dataset " + emg_data_dict[nf]['given_name'])
	emg_filt = emg_data_dict[nf]['emg_filt']
	env = emg_data_dict[nf]['env']
	[num_tastes, max_num_trials, num_time] = np.shape(emg_filt)
	emg_gapes = np.zeros((num_tastes,max_num_trials,num_time))
	taste_names = emg_data_dict[nf]['taste_names']
	for t_i in range(num_tastes):
		taste_save_dir = os.path.join(results_dir,taste_names[t_i])
		if not os.path.isdir(taste_save_dir):
			os.mkdir(taste_save_dir)
		for tr_i in range(max_num_trials):
			if not np.isnan(emg_filt[t_i,tr_i,0]): #Make sure a delivery actually happened - nan otherwise
				f, ax = plt.subplots(nrows=3,ncols=2)
				tr_emg = emg_filt[t_i,tr_i,:].flatten() #pre_taste+post_taste length array
				#Plot raw EMG data
				ax[0,0].plot(np.arange(-pre_taste,post_taste),tr_emg)
				ax[0,0].axvline(0,color='k',linestyle='dashed')
				ax[0,0].set_title('Raw EMG')
				#Plot enveloped EMG data
				tr_env = env[t_i,tr_i,:].flatten()
				max_env = np.max(tr_env)
				min_env = np.min(tr_env)
				ax[0,1].plot(np.arange(-pre_taste,post_taste),tr_env)
				ax[0,1].axvline(0,color='k',linestyle='dashed')
				ax[0,1].set_title('Enveloped EMG')
				#Calculate pre-taste enveloped signal mean and standard deviation
				mu_env = np.nanmean(tr_env[:pre_taste])
				sig_env = np.nanstd(tr_env[:pre_taste])
				#Find peaks above 1 std. and with a preset minimum dist between
				[peak_inds, peak_props] = find_peaks(tr_env-mu_env,prominence=sig_env,distance=min_inter_peak_dist,width=0,rel_height=0.99)
				#Find edges of peaks using peak widths function
				[peak_ws,_,_,_] = peak_widths(tr_env-mu_env,peak_inds)
				peak_left = np.array([np.max((peak_inds[p_i] - peak_ws[p_i]/2,0)) for p_i in range(len(peak_inds))])
				peak_right = np.array([np.min((peak_inds[p_i] + peak_ws[p_i]/2,pre_taste+post_taste-1)) for p_i in range(len(peak_inds))])
				#Plot peak heights and widths as a check
				ax[1,0].plot(np.arange(-pre_taste,post_taste),tr_env)
				peak_freq = np.zeros(len(peak_inds)) #Store instantaneous frequency
				peak_amp = np.zeros(len(peak_inds))
				jit = np.random.rand()/10
				for i in range(len(peak_inds)):
				    p_i = peak_inds[i]
				    ax[1,0].axvline(p_i-pre_taste,color='k',linestyle='dashed',alpha=0.5)
				    w_i = peak_props['widths'][i]
				    peak_freq[i] = 1/(w_i/1000) #Calculate instantaneous frequency
				    w_h = peak_props['width_heights'][i]
				    if np.mod(i,2) == 0:
				        ax[1,0].plot([peak_left[i]-pre_taste,peak_right[i]-pre_taste],[w_h + jit,w_h + jit],color='g',linestyle='dashed',alpha=0.5)
				    else:
				        ax[1,0].plot([peak_left[i]-pre_taste,peak_right[i]-pre_taste],[w_h,w_h],color='g',linestyle='dashed',alpha=0.5)
				    peak_amp[i] = tr_env[p_i]
				ax[1,0].set_xlim([0,1000])
				ax[1,0].set_title('Zoom peak loc + wid')
				#Plot instantaneous frequency
				ax[1,1].plot(peak_inds-pre_taste,peak_freq)
				ax[1,1].axhline(min_gape_band,linestyle='dashed',color='g')
				ax[1,1].axhline(max_gape_band,linestyle='dashed',color='g')
				ax[1,1].set_title('Instantaneous Frequency')
				#Plot movement amplitude
				ax[2,0].plot(peak_inds-pre_taste,peak_amp)
				ax[2,0].axhline(mu_env+sig_env,linestyle='dashed',alpha=0.5)
				ax[2,0].set_title('Peak Amplitude')
				#Pull out gape intervals only where amplitude is above cutoff, and frequency in range
				gape_peak_inds = np.where((peak_amp>=mu_env+sig_env)*(min_gape_band<=peak_freq)*(peak_freq<=max_gape_band))[0]
				gape_starts = peak_left[gape_peak_inds]
				gape_ends = peak_right[gape_peak_inds]
				ax[2,1].plot(np.arange(-pre_taste,post_taste),tr_env)
				ax[2,1].axvline(0,color='k',linestyle='dashed')
				for gpi in range(len(gape_peak_inds)):
					x_vals = np.arange(gape_starts[gpi],gape_ends[gpi]) - pre_taste
					ax[2,1].fill_between(x_vals,min_env*np.ones(len(x_vals)),max_env*np.ones(len(x_vals)),color='r',alpha=0.2)
				#ax[2,1].set_xlim([0,2000])
				ax[2,1].set_title('Enveloped EMG Gape Times')
				f.tight_layout()
				f.savefig(os.path.join(taste_save_dir,'emg_gapes_trial_' + str(tr_i) + '.png'))
				f.savefig(os.path.join(taste_save_dir,'emg_gapes_trial_' + str(tr_i) + '.svg'))
				plt.close(f)
				emg_data_dict[nf]['gape_starts'] = gape_starts
				emg_data_dict[nf]['gape_ends'] = gape_ends
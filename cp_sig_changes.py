#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:28:58 2023

@author: Hannah Germaine

Code to test which changepoints have significant activity changes across them
"""

#######################################################
### find neurons with FR changes at each transition ###
#######################################################

import numpy as np
from scipy import stats
import easygui, os

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


#%% Load all files to be analyzed

#Prompt user for the number of datasets needed in the analysis
num_files = int_input("How many data files do you need to import for this analysis (integer value)? ")
if num_files >= 1:
	print("Multiple file import selected.")
else:
	print("Single file import selected.")

#Pull all data into a dictionary
data_dict = dict()
for nf in range(num_files):
	#Directory selection
	print("Please select the folder where the data # " + str(nf+1) + " is stored.")
	data_dir = easygui.diropenbox(title='Please select the folder where data is stored.')
	#Search for matching file type - ends in _snips.npy
	files_in_dir = os.listdir(data_dir)
	matching_files = []
	for filename in files_in_dir:
		if filename[-10:] == '_snips.npy':
			matching_files.append(filename)
	cond_dict = dict() #Dictionary of all files in this folder / this "condition"
	#For each file in dir get a name and import data to dict
	cond_i = 0
	for filename in matching_files:
		cond_dict[cond_i] = dict()
		cond_dict[cond_i]['directory'] = data_dir
		data_array = np.load(os.path.join(data_dir,filename))
		#Name the dataset
		print("Give a more colloquial name to the dataset.")
		cond_dict[cond_i]['filename'] = filename
		given_name = input("How would you rename " + filename + "? ")
		cond_dict[cond_i]['given_name'] = given_name
		num_deliv, num_neur, num_time, num_cp = np.shape(data_array)
		cond_dict[cond_i]['num_deliv'] = num_deliv
		cond_dict[cond_i]['num_neur'] = num_neur
		cond_dict[cond_i]['num_time'] = num_time
		cond_dict[cond_i]['num_cp'] = num_cp
		cond_dict[cond_i]['data'] = data_array
		cond_i += 1
	#Add to the larger dict
	data_dict[nf] = cond_dict

#Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')

#%% Test changepoints with significant difference in activity

for nf in range(len(data_dict)):
	cond_dict = data_dict[nf]
	for cond_i in range(len(cond_dict)):
		data = cond_dict[cond_i]['data']
		num_deliv = cond_dict[cond_i]['num_deliv']
		num_neur = cond_dict[cond_i]['num_neur']
		num_time = cond_dict[cond_i]['num_time']
		num_cp = cond_dict[cond_i]['num_cp']
		#Calculate and store significant activity
		sig_neuron_activity = np.zeros((num_deliv,num_neur,num_cp))
		for taste_index in range(num_deliv):
			test_snip_array = data[taste_index]
			test_snips_before = test_snip_array[:, :int(num_time/2), :]
			test_snips_after = test_snip_array[:, int(num_time/2):, :]
			for cp_index in range(num_cp):
				for neur_index in range(num_neur):
					#using kolmogorov-smirnoff 2-sample test to compare firing rate distributions
					#on either side of a changepoint. This doesn't assume the type of distribution
					#(like if they're gaussian distributed firing rates or not) making it a better
					#test for our firing rate data
					pre_bin_spikes = test_snips_before[neur_index,:,cp_index]
					post_bin_spikes = test_snips_after[neur_index,:,cp_index]
					#use moving bins to get distributions of spikes in a bin on either side of a cp
					fr_bin = 50 #moving bin size
					pre_spike_sum =[np.sum(pre_bin_spikes[i:i+fr_bin]) for i in range(int(num_time/2) - fr_bin)]
					post_spike_sum =[np.sum(post_bin_spikes[i:i+fr_bin]) for i in range(int(num_time/2) - fr_bin)]
					#calculate the significance with the 2-sample KS-Test
					result= stats.ks_2samp(pre_spike_sum,post_spike_sum)
					#check p-value
					if result[1] < 0.05:
						sig_neuron_activity[taste_index,neur_index,cp_index] = 1
		#Store significant activity in dictionary
		data_dict[nf][cond_i]['sig_neuron_activity'] = sig_neuron_activity
		#Fraction of taste deliveries that each neuron is significantly changing activity across cp
		frac_sig_diff = np.sum(sig_neuron_activity,0)/num_deliv
		#Which neurons are generally significantly changing activity by cp
		sig_neur_by_cp = frac_sig_diff > 0.5
		data_dict[nf][cond_i]['sig_neur_by_cp'] = sig_neur_by_cp
		#Save in data folder
		filename = cond_dict[cond_i]['filename']
		sig_neur_filename = (filename.split('_')[:-1]).join('_') + '_sig_neurons.npy'
		np.save(os.path.join(cond_dict[cond_i]['directory'], sig_neur_filename),np.array(sig_neur_by_cp))
		
		
#%% Plot changepoint significance changes across datasets and conditions






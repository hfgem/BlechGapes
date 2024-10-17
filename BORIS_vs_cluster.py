#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:12:21 2024

@author: Hannah Germaine

This script is to be run following both BORIS_Converter.py and 
mouth_movement_analyses.py and is intended to calculate how many of the
gapes detected in mouth_movement_analyses overlap with visually identified
gapes in BORIS. Likewise, it calculates false-positives where detected
gapes overlap with other detected movements in BORIS.
"""

#%% Import necessary packages and functions

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from functions.mouth_movement_funcs import int_input


#%% Import cluster results

# Directory selection
print("Please select the folder where the data is stored.")
data_dir = filedialog.askdirectory()

# Analysis Storage Directory
results_dir = os.path.join(data_dir, 'BlechGapes_analysis')
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

# Check if dictionary previously saved
dict_save_dir = os.path.join(results_dir, 'emg_data_dict.pkl')
try:
    f = open(dict_save_dir, "rb")
    emg_data_dict = pickle.load(f)
    all_taste_gapes = np.load(os.path.join(results_dir, 'emg_clust_results.npy'))
    print("Results of mouth_movement_analyses.py loaded.")
except:
    print("ERROR: Results of mouth_movement_analyses.py NOT loaded.")
    print("Please check your directory / run mouth_movement_analyses.py on your data.")
    quit()

# Check if BORIS data previously saved
dataset_name =  emg_data_dict['given_name']
print("Please select the folder where BORIS reformatted scoring data is stored for " + dataset_name)
boris_dir = filedialog.askdirectory()
#Pull out all file names from directory
boris_file_list = os.listdir(boris_dir)
npy_file_list = [] #All names are stored in the format [behavior]_[taste].npy
for df in boris_file_list:
    if df[-4:] == '.npy':
        npy_file_list.append(df)
del df
if len(npy_file_list) == 0:
    print("ERROR: No BORIS files found.")
    print("Please check your directory / run BORIS_Converter.py.")
    quit()
else:
    print("Results of BORIS_Converter.py loaded.")

#%% Test cluster results against scoring results from BORIS
#This will only work if you've actually run scoring in BORIS and output the
#scoring data to .csv files in a single folder. It also relies on your running
#BORIS_Converter.py prior to running this code-block to ensure the scoring data
#is converted to a useable format for the analysis.

overlap_save_dir = os.path.join(results_dir,'BORIS_vs_cluster')
if not os.path.isdir(overlap_save_dir):
    os.mkdir(overlap_save_dir)

#Parameters
pre_taste = 2000 #how many bins are before delivery
video_FPS = 30 #how many frames per second is the video shot at?
cluster_overlap_buffer = np.ceil(1000/video_FPS).astype('int') #in ms
cluster_overlap_buffer = 100

#Pull out the unique behaviors and tastes in the directory
unique_boris_behaviors = []
unique_boris_tastes = []
for cf in npy_file_list:
    name_only = cf.split('.')[0]
    unique_boris_tastes.extend([name_only.split('_')[-1]])
    unique_boris_behaviors.extend([('_').join(name_only.split('_')[:-1])])
unique_boris_behaviors = [str(ub) for ub in np.unique(unique_boris_behaviors)]
unique_boris_tastes = [str(ut) for ut in np.unique(unique_boris_tastes)]
del cf, name_only
#Determine which index of behaviors is the gape
print("BORIS Index: Behavior")
for bb_i, bb in enumerate(unique_boris_behaviors):
    print(str(bb_i) + ': ' + bb)
del bb_i, bb
print(str(-1) + ": None of the above.")
gape_index = int_input("Which index is the gape index? ")
#Now align tastes in this directory with tastes in the cluster dataset
taste_names = emg_data_dict['taste_names']
boris_index_match = [] #Index of boris taste aligned with data taste (-1 if not available)
for t_n in taste_names:
    print("BORIS Index: Taste")
    for bt_i, bt in enumerate(unique_boris_tastes):
        print(str(bt_i) + ': ' + bt)
    print(str(-1) + ": None of the above.")
    index = int_input("Which index aligns with " + t_n + "? ")
    boris_index_match.extend([index])
del t_n, bt_i, bt, index

#Run through each taste and each delivery and check if there's BORIS data then compare
taste_gape_success = []
for t_i, t_name in enumerate(taste_names):
    bt_i = boris_index_match[t_i]
    if bt_i >= 0: #A BORIS match exists   
        #Import BORIS results for this taste and each behavior
        BORIS_gape_index = -1
        BORIS_behavior_names = []
        BORIS_behavior_data = []
        for bb_i, bb_name in enumerate(unique_boris_behaviors):
            try:
                Bbd = np.load(os.path.join(boris_dir,bb_name + '_' + unique_boris_tastes[bt_i] + '.npy'))
                BORIS_behavior_names.extend([bb_name])
                BORIS_behavior_data.extend([Bbd])
                if bb_i == gape_index:
                    BORIS_gape_index = bb_i
            except:
                print("No BORIS data for combination " + t_name + " " + bb_name)
        del Bbd, bb_i, bb_name
        #Gather cluster results for gapes to this taste
        bin_cluster_gapes = all_taste_gapes[t_i,:,:].squeeze() #trials x time
        _, clust_len = np.shape(bin_cluster_gapes)
        #Pull out BORIS trial indices and behavior start/end times [trial, start, end]
        BORIS_behavior_indices = []
        #Pull out whether there's a clustered gape aligning with each behavior timeframe
        cluster_behavior_overlap = []
        #Pull out the fraction of behaviors with a clustered gape aligned
        cluster_behavior_overlap_fracs = []
        for bb_i, bb_name in enumerate(BORIS_behavior_names):
            #Collect [trial, start, end] data for this particular behavior
            bb_indices = []
            bb_data = BORIS_behavior_data[bb_i]
            _, bb_len = np.shape(bb_data)
            min_behav_len = np.min([clust_len, bb_len])
            bb_data[:,0] = 0
            bb_data[:,-1] = 0
            tr_num, bb_len = np.shape(bb_data)
            for tr_i in range(tr_num):
                if np.sum(bb_data[tr_i,:]) > 0:
                    bb_starts = np.where(np.diff(bb_data[tr_i,:]) == 1)[0] + 1
                    bb_ends = np.where(np.diff(bb_data[tr_i,:]) == -1)[0] + 1
                    for bb_s_i, bb_s in enumerate(bb_starts):
                        if (bb_s < min_behav_len): #Only keep if within interval of time from both BORIS and clustering
                            bb_end = np.min([bb_ends[bb_s_i],min_behav_len])
                            bb_indices.append([tr_i,bb_s,bb_end])
                    del bb_starts, bb_ends, bb_s_i, bb_s
            del tr_i
            BORIS_behavior_indices.append(np.array(bb_indices))
            #Now for each check if there's an overlap with clustered gape data
            clust_overlap = np.zeros(len(bb_indices))
            for bb_i_i in range(len(bb_indices)):
                tr_i = bb_indices[bb_i_i][0]
                tr_bb_start = np.max([bb_indices[bb_i_i][1] - cluster_overlap_buffer,0])
                tr_bb_end = np.min([bb_indices[bb_i_i][2] + cluster_overlap_buffer,min_behav_len])
                bb_i_i_clust_data = bin_cluster_gapes[tr_i,tr_bb_start:tr_bb_end]
                if np.sum(bb_i_i_clust_data) > 0:
                    clust_overlap[bb_i_i] = 1
            cluster_behavior_overlap.append(clust_overlap)
            #Calculate fraction of overlap
            cluster_behavior_overlap_fracs.extend([np.sum(clust_overlap)/len(clust_overlap)])
            del bb_indices, bb_data, tr_num, bb_len
        del bb_i, bb_name
        
        #Plot true positive, false positive, true negative, and false negative rates by behavior
        f_gape_rates = plt.figure(figsize=(5,5))
        not_gape_inds = np.setdiff1d(np.arange(len(BORIS_behavior_names)),BORIS_gape_index*np.ones(1))
        true_positive = np.sum(cluster_behavior_overlap[BORIS_gape_index])
        false_negative = len(cluster_behavior_overlap[BORIS_gape_index]) - true_positive
        false_positive = np.sum([np.sum(cluster_behavior_overlap[ngi]) for ngi in not_gape_inds])
        true_negative = np.sum([len(cluster_behavior_overlap[i]) for i in range(len(cluster_behavior_overlap))]) - (true_positive + false_negative + false_positive)
        plt.pie([true_positive,false_negative,false_positive,true_negative], \
                labels=['true cluster gapes','missed cluster gapes','false cluster gapes (other behavior)','true cluster not gape'], \
                    autopct='%1.1f%%')
        plt.title(t_name + ' all behavior rates')
        f_gape_rates.savefig(os.path.join(overlap_save_dir,t_name+'_all_behavior_rates.png'))
        f_gape_rates.savefig(os.path.join(overlap_save_dir,t_name+'_all_behavior_rates.svg'))
        plt.close(f_gape_rates)
        
        f_just_gapes = plt.figure(figsize=(5,5))
        plt.pie([true_positive, false_negative], labels=['true cluster gapes', \
                'missed cluster gapes'],autopct='%1.1f%%')
        plt.title(t_name + ' just gape rates')
        f_just_gapes.savefig(os.path.join(overlap_save_dir,t_name+'_gape_rates.png'))
        f_just_gapes.savefig(os.path.join(overlap_save_dir,t_name+'_gape_rates.svg'))
        plt.close(f_just_gapes)
        
        taste_gape_success.extend([true_positive/(true_positive+false_negative)])
        
    del bt_i
del t_i, t_name

f_taste_success = plt.figure(figsize=(5,5))
plt.plot(np.arange(len(taste_names)),100*np.array(taste_gape_success),label='_')
mean = np.nanmean(100*np.array(taste_gape_success))
plt.axhline(mean,linestyle='dashed',\
            color='k',label='mean = ' + str(np.round(mean,2)))
plt.legend()
plt.ylim([0,100])
plt.xticks(np.arange(len(taste_names)),taste_names)
plt.ylabel('% Successfully Clustered Gapes')
plt.xlabel('Taste')
plt.title('Successfully Clustered Gapes (vs. BORIS)')
plt.tight_layout()
f_taste_success.savefig(os.path.join(overlap_save_dir,'taste_success_rates.png'))
f_taste_success.savefig(os.path.join(overlap_save_dir,'taste_success_rates.svg'))
plt.close(f_taste_success)


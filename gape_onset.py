#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:48:26 2023

This code plots gape onset

@authors: Christina Mazzio and Hannah Germaine
"""

# Import stuff!
import numpy as np
import tables, easygui, sys, os, glob, json, pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.stats import mannwhitneyu, f_oneway
from itertools import combinations
# Necessary blech_clust modules
#sys.path.append('/home/cmazzio/Desktop/blech_clust/')
#sys.path.append('Users/hannahgermaine/Documents/GitHub/blech_clust/')
sys.path.append('/home/cmazzio/Desktop/blech_clust/utils')
from blech_utils import imp_metadata

"""Changes to make:
    - Ask for user to select folder with data rather than manual write-in
    - Ask for user input on which index in "all_a1_gapes" associates with which tastant
    - Ask for user input on this_start and this_end - this_start gape onset and length restrictions
    - If there are 2 separate directories for the two tastes, handle (lower priority)
"""

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


def bw_plot(dataset,xlabels,all_pairs,sig_vals,ylabel,anim_name,title,savename,save_dir):
    #This function plots the results as box-and-whisker plots
    f_box = plt.figure(figsize=(8,8))
    for d_i in range(len(dataset)):
        dataset_i = np.array(dataset[d_i])
        no_nan_data = dataset_i[~np.isnan(dataset_i)]
        plt.boxplot([list(no_nan_data)],positions=[d_i+1],sym='',showmeans=False,meanline=False,medianprops=dict(linestyle='None',linewidth=0))
        plt.plot([d_i+1-0.1,d_i+1+0.1],[np.nanmean(no_nan_data),np.nanmean(no_nan_data)],label=xlabels[d_i]+' mean')
        plt.scatter(np.random.normal(d_i+1,0.04,size=len(no_nan_data)),no_nan_data,color='g',alpha=0.2,label='Individual Trials')
    plt.legend()
    xtick_vals = plt.xticks()[0]
    plt.xticks(ticks=xtick_vals,labels=xlabels)
    ytick_vals = plt.yticks()[0]
    max_ytick = np.max(ytick_vals)
    frac_ytick = (0.2*(len(xlabels)))*max(ytick_vals) #Grab fraction of the total height to add on top for sig bars and stars
    if len(all_pairs) > 0:
        sig_height_step = frac_ytick/len(all_pairs)
        sig_height_label_step = sig_height_step/2
        sig_height = max_ytick + sig_height_step
        for ap_i in range(len(all_pairs)):
            ap = all_pairs[ap_i]
            plt.plot([xtick_vals[ap[0]],xtick_vals[ap[1]]],[sig_height,sig_height],'k')
            if sig_vals[ap_i] == 0:
                plt.annotate('n.s.',((xtick_vals[ap[0]]+xtick_vals[ap[1]])/2,sig_height + sig_height_label_step))
            else:
                plt.annotate('*',((xtick_vals[ap[0]]+xtick_vals[ap[1]])/2,sig_height + sig_height_label_step))
            sig_height += sig_height_step
        plt.ylim((0,sig_height))
    plt.ylabel(ylabel)
    title_text = anim_name + '\n' + title
    plt.title(title_text)
    plt.tight_layout()
    fig_name = anim_name + savename
    fig_save_dir = os.path.join(save_dir,fig_name)
    f_box.savefig(fig_save_dir + '.png')
    f_box.savefig(fig_save_dir + '.svg')
    plt.close(f_box)
    
def hist_plot(dataset,data_labels,xlabel,anim_name,title,savename,save_dir):
    #This function plots the results as cumulative distribution plots
    data_lens = [len(dataset[i]) for i in range(len(dataset))]
    f_box = plt.figure(figsize=(8,8))
    plt.hist(dataset,bins=max(data_lens),density=True,cumulative=True,label=data_labels,histtype='step')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Density')
    title_text = anim_name + '\n' + title
    plt.title(title_text)
    plt.tight_layout()
    fig_name = anim_name + savename
    fig_save_dir = os.path.join(save_dir,fig_name)
    f_box.savefig(fig_save_dir + '.png')
    f_box.savefig(fig_save_dir + '.svg')
    plt.close(f_box)

def scatt_line_plot(dataset,xlabels,all_pairs,ylabel,anim_name,title,savename,save_dir):
    #This function plots the mean results as line/scatter plots showing how data changes across conditions
    num_datapoints = len(dataset)
    cm_subsection = np.linspace(0,1,num_datapoints)
    cmap = [cm.gist_rainbow(x) for x in cm_subsection]
    x_vals = np.arange(num_datapoints)
    mean_vals = [np.nanmean(dataset[i]) for i in range(num_datapoints)]
    std_vals = [np.nanstd(dataset[i]) for i in range(num_datapoints)]
    f_box = plt.figure(figsize=(8,8))
    for i in range(num_datapoints):
        x_i = x_vals[i]
        mu_i = mean_vals[i]
        std_i = std_vals[i]
        plt.scatter(x_i,mu_i,color=cmap[i])
        plt.plot([x_i,x_i],[mu_i-std_i,mu_i+std_i],color=cmap[i])
    plt.plot(x_vals,mean_vals,alpha=0.3,color='k',linestyle='dashed')
    plt.xticks(ticks=np.arange(num_datapoints),labels=xlabels)
    plt.ylabel(ylabel)
    title_text = anim_name + '\n' + title
    plt.title(title_text)
    plt.tight_layout()
    fig_name = anim_name + savename
    fig_save_dir = os.path.join(save_dir,fig_name)
    f_box.savefig(fig_save_dir + '.png')
    f_box.savefig(fig_save_dir + '.svg')
    plt.close(f_box)

#%% Import data

#Prompt user for the number of datasets needed in the analysis
num_anim = int_input("How many data files do you need to import for this analysis (integer value)? ")
if num_anim >= 1:
    print("Multiple file import selected.")
else:
    print("Single file import selected.")
    
#Prompt user which data type is being analyzed: BSA gapes or cluster gapes
#BSA = 1, Cluster = 2
data_type = int_input("Which gape dataset do you wish to analyze: BSA = 1, Cluster = 2 (enter integer value)? ")
if data_type == 1:
    type_name = 'bsa_gape_onset'
elif data_type == 2:
    type_name = 'cluster_gape_onset'
    print("Please select the folder where the clustering gape data is stored. (Likely BlechGapes_analysis)")
    clust_gape_dir = easygui.diropenbox(title='Please select the folder where data is stored.')
else:
    raise Exception

#Pull all data into a dictionary
data_dict = dict()
for na in range(num_anim):
    #Directory selection
    animal_dict = dict()
    print("Please select the folder where the data # " + str(na+1) + " is stored.")
    animal_dir = easygui.diropenbox(title='Please select the folder where data is stored.')
    animal_dict['dir'] = animal_dir
    #Name separation
    test_name = animal_dir.split('/')[-1]
    animal_dict['name'] = test_name
    print("Give a more colloquial name to the dataset.")
    given_name = input("How would you rename " + test_name + "? ")
    animal_dict['given_name'] = given_name
    #Gape data pull
    metadata_handler = imp_metadata([[], animal_dir])
    dir_name1 = metadata_handler.dir_name
    os.chdir(dir_name1)
    if data_type == 1:
        # Open the hdf5 file
        hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
        #extract data from HDf5 and get rid of dimensions with one value
        all_gapes = np.squeeze(hf5.root.emg_BSA_results.gapes[:])
        hf5.close()
    else: #Import cluster data
        all_gapes = np.load(os.path.join(clust_gape_dir,'emg_clust_results.npy'))
    animal_gape_data = []
    num_tastes = np.shape(all_gapes)[0]
    animal_taste_names = []
    #Prompt user to select the dimension of each taste and the name
    print("There is gape data in this directory for " + str(num_tastes) + " tastes.")
    print("Please select which tastes to keep in the analysis from this list of indices: " + str(np.arange(num_tastes)))
    tastes_keep = input("Provide the indices, comma-separated integers with no spaces: ")
    taste_keep_list = tastes_keep.split(',')
    print("For each taste, please provide the associated name.")
    for t_k in taste_keep_list:
        t_i = int(t_k)
        taste_name = input("\tName of tastant with index " + str(t_i) + ": ")
        animal_taste_names.append(taste_name)
        animal_gape_data.append(all_gapes[t_i,:,:])
    animal_dict['taste_names'] = animal_taste_names
    animal_dict['gape_data'] = animal_gape_data
    data_dict[na] = animal_dict

#Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')
results_dir = os.path.join(results_dir,type_name)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

#Save dictionary
dict_save_dir = os.path.join(results_dir,'gape_onset_dict.pkl')
f = open(dict_save_dir,"wb")
pickle.dump(data_dict,f)
#with open(dict_save_dir, "rb") as pickle_file:
#    data_dict = pickle.load(pickle_file)

#%% Process Gape Data for Analyses
pre_time = 2000 #pre-taste time in data (ms) to subtract

gape_data_lengths = []
num_anim = len(data_dict)
for na in range(num_anim):
    anim_gape_data = data_dict[na]['gape_data']
    num_tastes = len(data_dict[na]['taste_names'])
    anim_bool_gape_data = []
    for nt in range(num_tastes):
        taste_gape = anim_gape_data[nt]
        taste_gape_bool = (taste_gape > 0.5)*1
        taste_gape_bool[:,0] = 0
        taste_gape_bool[:,-1] = 0
        gape_data_lengths.extend(np.shape(taste_gape_bool[1]))
        anim_bool_gape_data.append(taste_gape_bool)
    data_dict[na]['bool_gape_data'] = anim_bool_gape_data
gape_data_lengths = np.array(gape_data_lengths)

#Get user input on the start time of a gape and length
print("To analyze gape data, please select the gape limits.")
gape_start_min = int_input("\tHow long after taste delivery (ms) do you want gape detection to begin? ")
gape_start_max = int_input("\tHow long after taste delivery (ms) do you want gape detection to end? ")
if data_type == 1:
    gape_end = int_input("\tWhat is the min length of a gape to consider (must be less than " + str(min(gape_data_lengths)) + ")? ")
else:
    gape_end = 0
    inter_gape_interval = int_input("\tWhat is the maximum interval (ms) between gapes to consider as part of a bout? ")
#Pull out gape times
for na in range(num_anim):
    anim_bool_gape_data = data_dict[na]['bool_gape_data'] #num tastes x [num_trials,num_time]
    anim_gape_times = []
    for taste in range(len(anim_bool_gape_data)):
        taste_gape_data = anim_bool_gape_data[taste] #[num_trials,num_time]
        taste_gape_times = [] #will be length 
        for trial in range(np.shape(taste_gape_data)[0]):
            trial_gape_data = taste_gape_data[trial,:]
            trial_diff = np.diff(trial_gape_data)
            trial_gape_starts = np.where(trial_diff == 1)[0] - pre_time +1 #starts of gapes converted to ms from taste delivery
            trial_gape_ends = np.where(trial_diff == -1)[0] - pre_time +1 #ends of gapes converted to ms from taste delivery
            #weed out gapes outside of selected gape-detection-interval and weed out gapes longer than given gape length max
            keep_inds = np.where((trial_gape_starts > gape_start_min)*(trial_gape_starts < gape_start_max)*(trial_gape_ends - trial_gape_starts >= gape_end))[0]
            trial_gape_starts = trial_gape_starts[keep_inds]
            trial_gape_ends = trial_gape_ends[keep_inds]
            num_trial_gapes = len(trial_gape_starts)
            if num_trial_gapes > 0:
                trial_gape_times = np.zeros((num_trial_gapes,2))
                trial_gape_times[:,0] = np.expand_dims(trial_gape_starts,0)
                trial_gape_times[:,1] = np.expand_dims(trial_gape_ends,0)
            else:
                trial_gape_times = np.nan*np.ones((1,2))
            taste_gape_times.append(trial_gape_times)
        anim_gape_times.append(taste_gape_times)
    data_dict[na]['gape_times'] = anim_gape_times
        
#%% Analyze First Gapes

#Create a storage folder
first_single_gapes_dir = os.path.join(results_dir,'first_single_gapes')
if os.path.isdir(first_single_gapes_dir) == False:
    os.mkdir(first_single_gapes_dir)

#Collect first gape start and end times and plot stats
first_single_gapes = []
first_single_gape_bout_lengths = []
first_data_names = []
first_bouts = []
first_bout_lengths = []
#_____Plot within-animal/dataset stats_____
for na in range(num_anim):
    anim_dir = data_dict[na]['dir']
    anim_name = data_dict[na]['given_name']
    anim_taste_names = data_dict[na]['taste_names']
    joint_names = [anim_name+'_'+anim_taste_name for anim_taste_name in anim_taste_names]
    anim_gape_times = data_dict[na]['gape_times']
    num_tastes = len(anim_taste_names)
    anim_first_single_gapes = []
    anim_first_single_gape_bout_lengths = [] #If clustering, save the bout length of consecutive gapes, if BSA save the length of the first BSA interval
    anim_first_bouts = []
    anim_first_bout_lengths = [] #If BSA save the length of the gape interval
    for t_i in range(num_tastes):
        taste_gape_times = anim_gape_times[t_i]
        num_trials = len(taste_gape_times)
        #If going of first gape onset only
        taste_first_single_gapes = []
        taste_first_single_gape_times = []
        taste_first_single_gape_bout_lengths = []
        #If going off first bout of gaping (cluster results only differ)
        taste_first_bout_onset = []
        taste_first_bout_times = []
        taste_first_bout_lengths = []       
        
        for trial in range(num_trials):
            trial_gape_times = taste_gape_times[trial]
            if ~np.isnan(trial_gape_times[0][0]):
                #Convert all gaping to individual bout start and end times as well as number of gapes per bout
                trial_gape_bouts = []
                trial_gape_bout_counts = []
                last_gape_start = trial_gape_times[0][0]
                last_gape_end = trial_gape_times[0][1]
                last_bout_count = 1
                for gp_i in range(len(trial_gape_times)-1):
                    new_gape_start = trial_gape_times[gp_i+1][0]
                    new_gape_end = trial_gape_times[gp_i+1][1]
                    if new_gape_start - last_gape_end <= inter_gape_interval:
                        #Part of last bout
                        last_gape_end = new_gape_end
                        last_bout_count += 1
                    else:
                        #Start of new bout
                        trial_gape_bouts.append([last_gape_start,last_gape_end])
                        trial_gape_bout_counts.append(last_bout_count)
                        last_gape_start = new_gape_start
                        last_gape_end = new_gape_end
                        last_bout_count = 1
                if (trial_gape_bouts[-1][0] != last_gape_start)*(trial_gape_bouts[-1][1] != last_gape_end):
                    trial_gape_bouts.append([last_gape_start,last_gape_end])
                    trial_gape_bout_counts.extend([last_bout_count])
                
                #Pull out gape data if looking at just the first detected gape as onset
                taste_first_single_gape_times.append(trial_gape_times[0])
                taste_first_single_gapes.extend([trial_gape_times[0][0]])
                taste_first_single_gape_bout_lengths.extend([trial_gape_bouts[0][1]-trial_gape_bouts[0][0]])
                    
                #Pull out gape data if looking at minimum bout length of > 1 (only for cluster results)
                if data_type == 1: #BSA
                    taste_first_bout_onset.append(trial_gape_times[0][0])
                    taste_first_bout_lengths.extend([trial_gape_times[0][1]-trial_gape_times[0][0]])
                else: #Cluster
                    first_bout_ind = np.where(trial_gape_bout_counts > 1)[0]
                    if ~np.isempty(first_bout_ind):
                        taste_first_bout_onset.append(trial_gape_bouts[first_bout_ind[0]][0])
                        taste_first_bout_times.append(trial_gape_bouts[first_bout_ind[0]])
                        taste_first_bout_lengths.append(trial_gape_bouts[first_bout_ind[0]][1] - trial_gape_bouts[first_bout_ind[0]][0])
            else:
                taste_first_single_gape_times.append([np.nan,np.nan])
                taste_first_single_gapes.extend([np.nan])
                taste_first_single_gape_bout_lengths.extend([np.nan])
                taste_first_bout_times.append([np.nan,np.nan])
                taste_first_bout_onset.extend([np.nan])
                taste_first_bout_lengths.extend([np.nan])
        #Save first gapes to numpy array
        #    SINGLE GAPE RESULTS
        taste_first_single_gape_times = np.array(taste_first_single_gape_times)
        gape_onset_dir = os.path.join(anim_dir,'gape_onset_plots')
        if os.path.isdir(gape_onset_dir) == False:
            os.mkdir(gape_onset_dir)
        gape_onset_type_dir = os.path.join(gape_onset_dir,type_name)
        if os.path.isdir(gape_onset_type_dir) == False:
            os.mkdir(gape_onset_type_dir)
        first_gape_save_dir = os.path.join(gape_onset_type_dir,anim_name + '_' + anim_taste_names[t_i] + '_first_single_gapes.npy')
        np.save(first_gape_save_dir, taste_first_single_gape_times)
        #    BOUT GAPE RESULTS
        taste_first_bout_times = np.array(taste_first_bout_times)
        first_bout_save_dir = os.path.join(gape_onset_type_dir,anim_name + '_' + anim_taste_names[t_i] + '_first_gape_bouts.npy')
        np.save(first_bout_save_dir, taste_first_bout_times)
        
        #Add to dictionary
        anim_first_single_gapes.append(taste_first_single_gapes)
        anim_first_single_gape_bout_lengths.append(taste_first_single_gape_bout_lengths)
        anim_first_bouts.append(taste_first_bout_onset)
        anim_first_bout_lengths.append(taste_first_bout_lengths)
        
    first_single_gapes.extend(anim_first_single_gapes)
    first_data_names.extend(joint_names)
    first_single_gape_bout_lengths.extend(anim_first_single_gape_bout_lengths)
    first_bouts.extend(anim_first_bouts)
    first_bout_lengths.extend(anim_first_bout_lengths)
    #Calculate statistical difference between taste pairs
    all_pairs = list(combinations(np.arange(num_tastes),2))
    #    SINGLE GAPE ANALYSIS
    sig_pairs_first_single_gapes = np.zeros(len(all_pairs))
    sig_pairs_first_single_gape_lengths = np.zeros(len(all_pairs))
    sig_pairs_first_single_gape_bout_lengths = np.zeros(len(all_pairs))
    for ap_i in range(len(all_pairs)):
        ap = all_pairs[ap_i]
        stat_first, p_first = mannwhitneyu(anim_first_single_gapes[ap[0]], anim_first_single_gapes[ap[1]])
        if p_first <= 0.05:
            sig_pairs_first_single_gapes[ap_i] = 1
        stat_first_len, p_first_len = mannwhitneyu(anim_first_single_gape_bout_lengths[ap[0]], anim_first_single_gape_bout_lengths[ap[1]])
        if p_first_len <= 0.05:
            sig_pairs_first_single_gape_bout_lengths[ap_i] = 1
    #    BOUT ANALYSIS
    sig_pairs_first_bouts = np.zeros(len(all_pairs))
    sig_pairs_first_bout_lengths = np.zeros(len(all_pairs))
    for ap_i in range(len(all_pairs)):
        ap = all_pairs[ap_i]
        stat_first, p_first = mannwhitneyu(anim_first_bouts[ap[0]], anim_first_bouts[ap[1]])
        if p_first <= 0.05:
            sig_pairs_first_bouts[ap_i] = 1
        
        stat_first_len, p_first_len = mannwhitneyu(anim_first_bout_lengths[ap[0]], anim_first_bout_lengths[ap[1]])
        if p_first_len <= 0.05:
            sig_pairs_first_bout_lengths[ap_i] = 1
        
    #Plot box-and-whisker plots of first gapes
    bw_plot(anim_first_single_gapes,anim_taste_names,all_pairs,sig_pairs_first_single_gapes,\
         'Time to First Gape (ms)',anim_name,'Time to First Gape',\
             '_time_to_first_gape_bw',first_single_gapes_dir)
    #Plot scatter/line trends across tastants of first gapes
    scatt_line_plot(anim_first_single_gapes,anim_taste_names,all_pairs,'Time to First Gape (ms)',\
                 anim_name,'Time to First Gape','_time_to_first_gape_scat',first_single_gapes_dir)
    #Plot cumulative histogram of first gapes
    hist_plot(anim_first_single_gapes,anim_taste_names,'Time to First Gape (ms)',\
            anim_name,'Time to First Gape','_time_to_first_gape_cumhist',first_single_gapes_dir)
    if data_type == 1: #BSA
        #Plot box-and-whisker plots of first gape lengths
        bw_plot(anim_first_single_gape_lengths,anim_taste_names,all_pairs,sig_pairs_first_single_gape_lengths,\
             'First Gape Length (ms)',anim_name,'First Gape Length',\
                 '_first_single_gape_lengths_bw',first_single_gapes_dir)
        #Plot cumulative histogram of first gape lengths
        hist_plot(anim_first_single_gape_lengths,anim_taste_names,'First Gape Length (ms)',\
                anim_name,'First Gape Length','_first_single_gape_lengths_cumhist',first_single_gapes_dir)
        #Plot scatter/line trends across tastants of first gape lengths
        scatt_line_plot(anim_first_single_gape_lengths,anim_taste_names,all_pairs,'First Gape Length (ms)',\
                anim_name,'First Gape Length','_first_single_gape_lengths_scat',first_single_gapes_dir)
    else: #Clusters
        #Plot box-and-whisker plots of first gape lengths
        bw_plot(anim_first_single_gape_bout_lengths,anim_taste_names,all_pairs,sig_pairs_first_single_gape_bout_lengths,\
             'First Gape Bout Length (ms)',anim_name,'First Gape Bout Length',\
                 '_first_single_gape_bout_lengths_bw',first_single_gapes_dir)
        #Plot cumulative histogram of first gape lengths
        hist_plot(anim_first_single_gape_bout_lengths,anim_taste_names,'First Gape Bout Length (ms)',\
                anim_name,'First Gape Bout Length','_first_single_gape_bout_lengths_cumhist',first_single_gapes_dir)
        #Plot scatter/line trends across tastants of first gape lengths
        scatt_line_plot(anim_first_single_gape_bout_lengths,anim_taste_names,all_pairs,'First Gape Bout Length (ms)',\
                anim_name,'First Gape Bout Length','_first_single_gape_bout_lengths_scat',first_single_gapes_dir)

#_____Calculate across-animal/dataset stats_____        
all_pairs = list(combinations(np.arange(len(first_data_names)),2))
sig_pairs_first_single_gapes = np.zeros(len(all_pairs))
sig_pairs_first_single_gape_lengths = np.zeros(len(all_pairs))
sig_pairs_first_single_gape_bout_lengths = np.zeros(len(all_pairs))
for ap_i in range(len(all_pairs)):
    ap = all_pairs[ap_i]
    stat_first, p_first = mannwhitneyu(first_single_gapes[ap[0]], first_single_gapes[ap[1]])
    if p_first <= 0.05:
        sig_pairs_first_single_gapes[ap_i] = 1
    if data_type == 1: #BSA
        stat_first_len, p_first_len = mannwhitneyu(first_single_gape_lengths[ap[0]], first_single_gape_lengths[ap[1]])
        if p_first_len <= 0.05:
            sig_pairs_first_single_gape_lengths[ap_i] = 1
    else: #Clustering
        stat_first_len, p_first_len = mannwhitneyu(first_single_gape_bout_lengths[ap[0]], first_single_gape_bout_lengths[ap[1]])
        if p_first_len <= 0.05:
            sig_pairs_first_single_gape_bout_lengths[ap_i] = 1
        
#_____Plot across-animal/dataset stats_____
#Plot box-and-whisker plots of first gapes
bw_plot(first_single_gapes,first_data_names,all_pairs,sig_pairs_first_single_gapes,\
     'Time to First Gape (ms)','all','Time to First Gape',\
         '_time_to_first_gape_bw',first_single_gapes_dir)
#Plot cumulative histogram of first gapes
hist_plot(first_single_gapes,first_data_names,'Time to First Gape (ms)',\
        'all','Time to First Gape','_time_to_first_gape_cumhist',first_single_gapes_dir)
if data_type == 1: #BSA
    #Plot box-and-whisker plots of first gape lengths
    bw_plot(first_single_gape_lengths,first_data_names,all_pairs,sig_pairs_first_single_gape_lengths,\
         'First Gape Length (ms)','all','First Gape Length',\
             '_first_single_gape_lengths_bw',first_single_gapes_dir)
    #Plot cumulative histogram of first gape lengths
    hist_plot(first_single_gape_lengths,first_data_names,'First Gape Length (ms)',\
            'all','First Gape Length','_first_single_gape_lengths_cumhist',first_single_gapes_dir)
else: #Clustering
    #Plot box-and-whisker plots of first gape lengths
    bw_plot(first_single_gape_bout_lengths,first_data_names,all_pairs,sig_pairs_first_single_gape_bout_lengths,\
         'First Gape Bout Length (ms)','all','First Gape Bout Length',\
             '_first_single_gape_bout_lengths_bw',first_single_gapes_dir)
    #Plot cumulative histogram of first gape lengths
    hist_plot(first_single_gape_bout_lengths,first_data_names,'First Gape Bout Length (ms)',\
            'all','First Gape Bout Length','_first_single_gape_bout_lengths_cumhist',first_single_gapes_dir)

#%% Compare across animals the same tastes (must be imported in the same order too)

f_compare_first = plt.figure()
f_compare_length = plt.figure()
cm_subsection = np.linspace(0,1,num_anim)
cmap = [cm.gist_rainbow(x) for x in cm_subsection]
taste_names = []
#_____Plot within-animal/dataset stats_____
for na in range(num_anim):
    anim_name = data_dict[na]['given_name']
    anim_taste_names = data_dict[na]['taste_names']
    joint_names = [anim_name+'_'+anim_taste_name for anim_taste_name in anim_taste_names]
    anim_gape_times = data_dict[na]['gape_times']
    num_tastes = len(anim_taste_names)
    if na == 0:
        taste_names = anim_taste_names
    first_gape_means = []
    first_gape_std = []
    first_gape_length_means = []
    first_gape_length_stds = []
    for t_i in range(num_tastes):
        taste_gape_times = anim_gape_times[t_i]
        num_trials = len(taste_gape_times)
        taste_first_single_gapes = []
        taste_first_single_gape_lengths = []
        for trial in range(num_trials):
            trial_gape_times = taste_gape_times[trial]
            if ~np.isnan(trial_gape_times[0][0]):
                taste_first_single_gapes.extend([trial_gape_times[0][0]])
                taste_first_single_gape_lengths.extend([trial_gape_times[0][1]-trial_gape_times[0][0]])
        first_gape_means.extend([np.nanmean(taste_first_single_gapes)])
        first_gape_std.extend([np.nanstd(taste_first_single_gapes)])
        first_gape_length_means.extend([np.nanmean(taste_first_single_gape_lengths)])
        first_gape_length_stds.extend([np.nanstd(taste_first_single_gape_lengths)])
    plt.figure(1)
    plt.scatter(np.arange(num_tastes),first_gape_means,color=cmap[na],label=anim_name)
    for t_i in range(num_tastes):
        plt.plot([t_i,t_i],[first_gape_means[t_i]-first_gape_std[t_i],first_gape_means[t_i]+first_gape_std[t_i]],color=cmap[na],alpha=0.5,label='_nolegend_')
    plt.plot(np.arange(num_tastes),first_gape_means,color=cmap[na],alpha=0.5,label='_nolegend_')
    plt.figure(2)
    plt.scatter(np.arange(num_tastes),first_gape_length_means,color=cmap[na],label=anim_name)
    for t_i in range(num_tastes):
        plt.plot([t_i,t_i],[first_gape_length_means[t_i]-first_gape_length_stds[t_i],first_gape_length_means[t_i]+first_gape_length_stds[t_i]],color=cmap[na],alpha=0.5,label='_nolegend_')
    plt.plot(np.arange(num_tastes),first_gape_length_means,color=cmap[na],alpha=0.5,label='_nolegend_')




#%% PLOTTING MEAN GAPE ONSET FOR EACH ANIMAL

# mean_array = [mean_gape_onset_sac_a1, mean_gape_onset_sac_a2, mean_gape_onset_sac_a3, mean_gape_onset_qhcl_a1, mean_gape_onset_qhcl_a2, mean_gape_onset_qhcl_a3]
# yy = np.vstack([mean_array[[0,3]], mean_array[[1,4]], mean_array[[2,5]]])

#animals with one train session only (TG11, TG13, CM18)
fig = plt.figure(figsize=(10,8))
plt.scatter(1, mean_gape_onset_sac_a1, s= 200, color = 'blue')
plt.scatter(1, mean_gape_onset_sac_a2, s= 200, color = 'orange')
plt.scatter(1, mean_gape_onset_sac_a3, s= 200, color = 'green')
plt.scatter(2, mean_gape_onset_qhcl_a1, s = 200, color = 'blue')
plt.scatter(2, mean_gape_onset_qhcl_a2, s = 200, color = 'orange')
plt.scatter(2, mean_gape_onset_qhcl_a3, s = 200, color = 'green')
plt.plot([1,2],[mean_gape_onset_sac_a1, mean_gape_onset_qhcl_a1], color = 'blue')
plt.plot([1,2],[mean_gape_onset_sac_a2, mean_gape_onset_qhcl_a2], color = 'orange')
plt.plot([1,2],[mean_gape_onset_sac_a3, mean_gape_onset_qhcl_a3], color = 'green')

plt.xticks([1,2], ['Saccharin', 'Quinine'], fontsize = 20)
plt.xlabel('Taste', fontsize=20, fontweight= 'bold')
plt.ylabel('Mean Gape Onset', fontsize=20, fontweight = 'bold')
plt.ylim([0, 800])
plt.xlim([0.8, 2.2])
plt.yticks(fontsize = 20)

#adding CM26 to TG11, TG13, and CM18  (all with 100ms as min gape onset time)
CM26_sac_mean_test = 435.15384615384613
CM26_qhcl_mean_test =  1044.0833333333333

fig = plt.figure(figsize=(10,8))

plt.scatter(1, CM26_sac_mean_test, s= 200, color = 'mediumpurple')
plt.scatter(1, mean_gape_onset_sac_a1, s= 200, color = 'blue')
plt.scatter(1, mean_gape_onset_sac_a2, s= 200, color = 'orange')
plt.scatter(1, mean_gape_onset_sac_a3, s= 200, color = 'green')
plt.scatter(2, mean_gape_onset_qhcl_a1, s = 100, color = 'blue')
plt.scatter(2, mean_gape_onset_qhcl_a2, s = 100, color = 'red')
plt.scatter(2, mean_gape_onset_qhcl_a3, s = 100, color = 'green')
plt.scatter(2, CM26_qhcl_mean_test, s= 200, color = 'mediumpurple')
plt.plot([1,2],[mean_gape_onset_sac_a1, mean_gape_onset_qhcl_a1], color = 'blue')
plt.plot([1,2],[mean_gape_onset_sac_a2, mean_gape_onset_qhcl_a2], color = 'red')
plt.plot([1,2],[mean_gape_onset_sac_a3, mean_gape_onset_qhcl_a3], color = 'green')
plt.plot([1,2],[CM26_sac_mean, CM26_qhcl_mean], color = 'mediumpurple')

plt.xticks([1,2], ['Saccharin', 'Quinine'], fontsize = 20)
plt.xlabel('Taste', fontsize=20, fontweight= 'bold')
plt.ylabel('Mean Gape Onset', fontsize=20, fontweight = 'bold')
plt.ylim([0, 1100])
plt.yticks(fontsize = 20)
plt.xlim([0.8, 2.2])


CM26_sac_mean_train2 = 524.2727272727273
CM26_qhcl_mean_train2 =  1044.0833333333333

fig = plt.figure(figsize=(10,8))
plt.scatter(1, CM26_sac_mean_test, s= 100, color = 'orange')
plt.scatter(1, CM26_sac_mean_train2, s= 100, color = 'red')
plt.scatter(2, CM26_qhcl_mean_test, s= 100, color = 'orange')
plt.scatter(2, CM26_qhcl_mean_train2, s= 100, color = 'red')
plt.plot([1,2],[CM26_sac_mean_test, CM26_qhcl_mean_test], color = 'orange')
plt.plot([1,2],[CM26_sac_mean_train2, CM26_qhcl_mean_train2], color = 'red')

plt.xticks([1,2], ['Saccharin', 'Quinine'], fontsize = 20)
plt.xlabel('Taste', fontsize=20, fontweight= 'bold')
plt.ylabel('Mean Gape Onset', fontsize=20, fontweight = 'bold')
plt.ylim([0, 1100])
plt.yticks(fontsize = 20)
plt.xlim([0.8, 2.2])

        
stat, p = mannwhitneyu(qhcl_gape_array[:,1], sac_gape_array_test1[:,1])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')         
            
            
#%% Comparing train2 day saccharin and test2 day qhcl

train2_dir = '/media/cmazzio/large_data/CM26/CM26_CTATrain2_sac_h2o_230818_104128/'


metadata_handler = imp_metadata([[], train2_dir])
dir_name3 = metadata_handler.dir_name
os.chdir(dir_name3)
 
# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

#extract test day 1 data from HDf5 and get rid of dimensions with one value
all_train2_gapes = np.squeeze(hf5.root.emg_BSA_results.gapes[:])

hf5.close()

#pulls out probability of gaping frequency at each time point across all trials of sac
sac_gapes_train2 = all_train2_gapes[1,:,:]


#turns array into boolean in which 1 means gape is occurring and -1 means gape is not occurring.
sac_gapes_train2_boolean = (sac_gapes_train2 > 0.5)*1

#first and last time points all have 0s (meaning gaping is not occurring at these time points)
sac_gapes_train2_boolean[:,0] = 0
sac_gapes_train2_boolean[:,-1] = 0

first_single_gapes_sac_train2 = []
for trial in range(sac_gapes_train2_boolean.shape[0]):
    for time in range(sac_gapes_train2_boolean.shape[1]):
        this_data = sac_gapes_train2_boolean[trial][time]
        previous_data = sac_gapes_train2_boolean[trial][time-1]
        if this_data == 1 and previous_data == 0:
            this_start = time - 2000
        if this_data == 0 and previous_data == 1:
            this_end = time - 2000
            this_save_inds = [trial, this_start, this_end]
            if this_start > 0 and this_start <2000 and this_end - this_start >= 500:
                first_single_gapes_sac_train2.append(this_save_inds)
                break

sac_gape_array_train2 = np.array(first_single_gapes_sac_train2)
sac_gape_onset_vector_train2 = sac_gape_array_train2[:,1]
mean_gape_onset_sac_train2 = np.mean(sac_gape_onset_vector_train2)           
sd_mean_gape_onset_sac_train2 = np.std(sac_gape_onset_vector_train2)

se_sac_train2 = sd_mean_gape_onset_sac_train2/np.sqrt(len(first_single_gapes_sac_train2[1]))

# fig = plt.figure(figsize=(10,8))
# plt.ylabel('Mean onset time of gape bouts across trials (ms)')
# #plot each taste avg onset across animals
# plt.bar(2, mean_gape_onset_sac, yerr=se_sac, color = 'orange')

np.save('/media/cmazzio/large_data/CM26/CM26_CTATrain2_sac_h2o_230818_104128/gape_onset_plots/CM26_CTATrain2_first_sac_gapes.npy', np.array(first_single_gapes_sac_train2))


fig = plt.figure(figsize=(10,8))
plt.ylabel('Mean onset time of gape bouts across trials (ms)')
#plot each taste avg onset across animals
plt.bar(1, mean_gape_onset_sac_train2, yerr=se_sac_train2, color = 'mediumpurple')
plt.bar(2, mean_gape_onset_qhcl, yerr= se_qhcl , color = 'coral')
# plt.vlines(1, ci_taste2_all_animals[0], ci_taste2_all_animals[1])
plt.xticks([1,2], ['saccharin', 'quinine'], fontsize =20, fontweight = 'bold')
plt.yticks(fontsize=20)
plt.ylabel('Mean onset of gaping across trials (ms)', fontsize=20, fontweight = 'bold')
plt.xlabel('Tastes', fontsize=20, fontweight = 'bold')           
            
image_name = 'avg_gape_onset_CM26_train2sac_qhcl.svg'

plt.savefig('/media/cmazzio/large_data/CM26/CM26_Figures/' + image_name, dpi=300)             
            
#run non-parametric t test 
stat, p = mannwhitneyu(qhcl_gape_array[:,1], sac_gape_array_train2[:,1])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')        
    
    
    
#%% older code - might still be useful 
###################################################################################################33
# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
# Necessary blech_clust modules
sys.path.append('/home/cmazzio/Desktop/blech_clust/')
from utils.blech_utils import (
        imp_metadata,
        )
#'/media/cmazzio/large_data/CM26/CM26_CTATest_h2o_sac_nacl_qhcl_230819_101513'
#'/media/cmazzio/large_data/CM26/CM26_CTATest2_h2o_sac_ca_qhcl_230820_121050'

#animal_dir = ['CM26_CTA_Test2','media/cmazzio/large_data/CM26/CM26_Test2_h2o_sac_ca_qhcl_230820_121050']
animal_dir = ['/media/cmazzio/large_data/CM26/CM26_CTATest2_h2o_sac_ca_qhcl_230820_121050']

for animal_ind in range(len(animal_dir)):
    this_dir= animal_dir[animal_ind]
    this_name = animal_dir[animal_ind].split('/')[-1]
    # Ask for the directory where the Train day 1 hdf5 file sits, and change to that directory
    # Get name of directory with the data files
    metadata_handler = imp_metadata([[],this_dir])
    dir_name1 = metadata_handler.dir_name
    # info_dict1 = metadata_handler.info_dict
    # params_dict1 = metadata_handler.params_dict
    os.chdir(dir_name1)
    
    
    # Open the hdf5 file
    hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')
    
    #extract Train day 1 data from HDf5 and get rid of dimensions with one value
    all_gapes = np.squeeze(hf5.root.emg_BSA_results.gapes[:])
    
    hf5.close()
    
    #pulls out probability of gaping frequency at each time point across all trials of qhcl and sac
    qhcl_gapes = all_gapes[3,:,:]
    sac_gapes = all_gapes[1,:,:]
    
    
    #turns array into boolean in which 1 means gape is occurring and -1 means gape is not occurring.
    qhcl_gapes_boolean = (qhcl_gapes > 0.5)*1
    sac_gapes_boolean = (sac_gapes > 0.5)*1
    
    qhcl_gapes_boolean[:,0] = 0
    sac_gapes_boolean[:,0] = 0
    
    qhcl_gapes_boolean[:,-1] = 0
    sac_gapes_boolean[:,-1] = 0
    # =============================================================================
    # delta = np.diff(a1_qhcl_gapes_test2, axis=-1)
    # 
    # starts = np.where(delta==1)
    # ends = np.where(delta==-1)
    # 
    # fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
    # ax[0].imshow(a1_qhcl_gapes_test2, aspect='auto', interpolation='none',
    #              origin = 'lower')
    # ax[1].scatter(*starts[::-1], marker = '|', color = 'b')
    # ax[1].scatter(*ends[::-1], marker = '|', color = 'r')
    # =============================================================================



    #FIND FIRST QHCL GAPES#
    #NM's cutoff for minimum length of gape bout was 300 ms. Grill & Norgren paper found avg gape duration was 166ms- NM's cutoff asssumes at least two gapes are occurring together. 
    #cycles through every trial and time point and looks for gaping (1 if gape, -1 if not gape). Adds gape to "first gapes" if it is a bout lasting at least 300 ms
    first_single_gapes_qhcl = []
    for trial in range(qhcl_gapes_boolean.shape[0]):
        for time in range(qhcl_gapes_boolean.shape[1]):
            this_data = qhcl_gapes_boolean[trial][time]
            previous_data = qhcl_gapes_boolean[trial][time-1]
            if this_data == 1 and previous_data == 0:
                this_start = time - 2000
            if this_data == 0 and previous_data == 1:
                this_end = time - 2000
                this_save_inds = [trial, this_start, this_end]
                if this_start < 2000 and this_start > 100 and this_end - this_start >= 300:
                    first_single_gapes_qhcl.append(this_save_inds)
                    break
                
    np.save('media/cmazzio/large_data/CM26/CM26_Test2_h2o_sac_ca_qhcl_230820_121050/gape_onset_plots/CM26_Test2_h2o_sac_ca_qhcl_gapes_qhcl', first_single_gapes_qhcl)
    


    #FIND FIRST SAC GAPES#
    #NM's cutoff for minimum length of gape bout was 300 ms. Grill & Norgren paper found avg gape duration was 166ms- NM's cutoff asssumes at least two gapes are occurring together. 
    #cycles through every trial and time point and looks for gaping (1 if gape, -1 if not gape). Adds gape to "first gapes" if it is a bout lasting at least 300 ms
    first_single_gapes_sac = []
    for trial in range(sac_gapes_boolean.shape[0]):
        for time in range(sac_gapes_boolean.shape[1]):
            this_data = sac_gapes_boolean[trial][time]
            previous_data = sac_gapes_boolean[trial][time-1]
            if this_data == 1 and previous_data == 0:
                this_start = time - 2000
            if this_data == 0 and previous_data == 1:
                this_end = time - 2000
                this_save_inds = [trial, this_start, this_end]
                if this_start < 2000 and this_start > 100 and this_end - this_start >= 300:
                    first_single_gapes_sac.append(this_save_inds)
                    break
    
    np.save('media/cmazzio/large_data/CM26/CM26_Test2_h2o_sac_ca_qhcl_230820_121050/gape_onset_plots/CM26_Test2_h2o_sac_ca_qhcl_gapes_sac', first_single_gapes_sac)
        
                
    mean_gape_onset_qhcl =  np.mean(first_single_gapes_qhcl[1], axis=0)
    mean_gape_onset_sac = np.mean(first_single_gapes_sac[1], axis=0)
    
    sd_mean_gape_onset_qhcl = np.std(first_single_gapes_qhcl[1], axis=0)
    sd_mean_gape_onset_sac = np.std(first_single_gapes_sac[1], axis = 0)
    
    se_qhcl = sd_mean_gape_onset_qhcl/np.sqrt(len(first_single_gapes_qhcl[1]))
    se_sac = sd_mean_gape_onset_sac/np.sqrt(len(first_single_gapes_sac[1]))
    
    fig = plt.figure(figsize=(10,8))
    plt.ylabel('Mean onset time of gape bouts across trials (ms)')
    #plot each taste avg onset across animals
    plt.bar(1, mean_gape_onset_qhcl, yerr= se_qhcl , color = 'orchid')
    # plt.vlines(1, ci_taste2_all_animals[0], ci_taste2_all_animals[1])
    #plt.bar(2, mean_gape_onset_sac, yerr=se_sac, color = 'orange')
    plt.xticks([1,2], ['quinine', 'saccharin'], fontsize =20, fontweight = 'bold')
    plt.yticks(fontsize=20)
    plt.ylabel('Mean onset of gaping across trials (ms)', fontsize=20, fontweight = 'bold')
    plt.xlabel('Tastes', fontsize=20, fontweight = 'bold')
    #fig.savefig('media/cmazzio/large_data/CM26/CM26_Test2_h2o_sac_ca_qhcl_230820_121050/gape_onset_plots', mean_gape_onset + str(this_name))



#%% PLOTTING COMPARISON OF MEAN SACCHARIN GAPE ONSET ACROSS ANIMALS

#sac train 2 information from gape_onset_CM26_CM.py

#arrays with gape onsets for each animal for stats
stat,p = f_oneway(sac_gape_onset_vector_train2, sac_gape_onset_vector_a1, sac_gape_onset_vector_a2, sac_gape_onset_vector_a3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')        


fig = plt.figure(figsize=(10,8))
plt.ylabel('Mean onset time of gape bouts across trials (ms)')
#plot each taste avg onset across animals
plt.bar(1, mean_gape_onset_sac_train2, yerr= se_sac_train2 , color = 'mediumpurple')
plt.bar(2, mean_gape_onset_sac_a1, yerr= se_sac_a1 , color = 'blue')
plt.bar(3, mean_gape_onset_sac_a2, yerr= se_sac_a2 , color = 'orange')
plt.bar(4, mean_gape_onset_sac_a3, yerr= se_sac_a3 , color = 'green')
# plt.vlines(1, ci_taste2_all_animals[0], ci_taste2_all_animals[1])
#plt.bar(2, mean_gape_onset_sac, yerr=se_sac, color = 'orange')
plt.xticks([1,2,3,4], ['1', '2', '3', '4'], fontsize =20, fontweight = 'bold')
plt.yticks(fontsize=20)
plt.ylabel('Mean onset of saccharin gaping (ms)', fontsize=20, fontweight = 'bold')
plt.xlabel('Animal number', fontsize=20, fontweight = 'bold')
#fig.savefig('media/cmazzio/large_data/CM26/CM26_Test2_h2o_sac_ca_qhcl_230820_121050/gape_onset_plots', mean_gape_onset + str(this_name))   
        



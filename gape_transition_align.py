#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:51:31 2023

@author: cmazzio + hgerm
"""

###############################################################################################
### find neurons with FR changes at each transition for CM26 quinine & saccharin on Test day 1###
###############################################################################################
import sys, pickle, easygui, os
from matplotlib import cm
from pytau.changepoint_io import FitHandler
import pylab as plt
from pytau.utils import plotting
import numpy as np
from matplotlib import pyplot as plt
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
tau_data_dict = dict()
for nf in range(num_files):
	#Directory selection
	print("Please select the folder where the data # " + str(nf+1) + " is stored.")
	data_dir = easygui.diropenbox(title='Please select the folder where data is stored.')
	#Import individual trial changepoint data
	name_bool = dframe['data.data_dir'].isin([data_dir])
	wanted_frame = dframe[name_bool]
	state_bool = wanted_frame['model.states'] == desired_states
	wanted_frame = wanted_frame[state_bool]
	name_list = wanted_frame['data.basename']
	taste_list = wanted_frame['data.taste_num']
	print("There are " + str(len(taste_list)) + " tastes available.")
	taste_bool = int_input("Which taste do you want (index starting from 0)? ")
	wanted_frame = wanted_frame[taste_bool]
	taste_list = wanted_frame['data.taste_num']
	pkl_path_list = wanted_frame['exp.save_path']
	this_handler = PklHandler(pkl_path_list[0])
	#Import changepoints for each delivery
	scaled_mode_tau = this_handler.tau.scaled_mode_tau #num trials x num cp
	#Import spikes following each taste delivery
	spike_train = this_handler.firing.raw_spikes
	#Store changepoint and spike data in dictionary
	tau_data_dict[nf] = dict()
	tau_data_dict[nf]['data_dir'] = data_dir
	print("Give a more colloquial name to the dataset.")
	given_name = input("How would you rename " + name_list + "? ")
	tau_data_dict[nf]['given_name'] = given_name
	tau_data_dict[nf]['taste_list'] = taste_list
	tau_data_dict[nf]['states'] = desired_states
	tau_data_dict[nf]['scaled_mode_tau'] = scaled_mode_tau
	tau_data_dict[nf]['spike_train'] = spike_train
	#Import associated gapes
	print("Now import associated first gapes data with this dataset.")
	gape_data_dir = easygui.diropenbox(title='Please select the folder where data is stored.')
	#Search for matching file type - ends in _gapes.npy
	files_in_dir = os.listdir(gape_data_dir)
	for filename in files_in_dir:
		if filename[-10:] == '_gapes.npy':
			bool_val = bool_input("Is " + filename + " the correct associated file with " + given_name + "?")
			if bool_val == 'y':
				first_gapes =  np.load(os.path.join(gape_data_dir,filename))
				tau_data_dict[nf]['first_gapes'] = first_gapes
	try: #Check that something was imported
		first_gapes = tau_data_dict[nf]['first_gapes']
	except:
		'First gapes file not found in given folder. Program closing - try again.'
		quit()
	
#Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')

#Save dictionary
dict_save_dir = os.path.join(results_dir,'tau_dict.pkl')
f = open(dict_save_dir,"wb")
pickle.dump(tau_data_dict,f)
#with open(dict_save_dir, "rb") as pickle_file:
#    tau_data_dict = pickle.load(pickle_file)

#%% ALIGNING GAPE ONSET TO A TRANSITION

pre_taste_time = 2000

#save folders
gape_align_cp_dir = os.path.join(results_dir,'gape_align_plots')
if os.path.isdir(gape_align_cp_dir) == False:
	os.mkdir(gape_align_cp_dir)
	
cp_stats_dir = os.path.join(results_dir,'cp_stats_plots')
if os.path.isdir(cp_stats_dir) == False:
	os.mkdir(cp_stats_dir)

#tau for all trials of each dataset 
tau_data = []
tau_data_names = []
first_gapes_data = []
preceding_transitions = []
for nf in range(len(tau_data_dict)):
	given_name = tau_data_dict[nf]['given_name']
	#load changepoint information
	scaled_mode_tau = tau_data_dict[nf]['scaled_mode_tau']
	taste_list = tau_data_dict[nf]['taste_list']
	tau_data.append(scaled_mode_tau - pre_taste_time)
	#load first gapes information 
	first_gapes = tau_data_dict[nf]['first_gapes']
	first_gapes_data.append(first_gapes)
	#transition preceding each trial's first gape
	pre_cp_i = np.zeros(np.shape(first_gapes)[0])
	for fg_i, fg_times in enumerate(first_gapes):
		trial_tau = scaled_mode_tau[fg_i] - pre_taste_time
		trial_gape_onset = fg_times[0]
		pre_cp = np.where(trial_gape_onset - trial_tau > 0)[0][0]
		pre_cp_i[fg_i] = pre_cp
	preceding_transitions.append(pre_cp_i)
	f_pre = plt.figure()
	plt.hist(pre_cp_i)
	plt.title('Changepoint Index Preceding First Gape')
	plt.xlabel('Changepoint Index')
	plt.ylabel('Number of Trials')
	plt.savefig(f_pre,cp_stats_dir + given_name + '_cp_preceding_first_gape.png')
	plt.savefig(f_pre,cp_stats_dir + given_name + '_cp_preceding_first_gape.svg')
	#spike trains for trial
	spike_trains = tau_data_dict[nf]['spike_train']
	#plot spike train with overlaid changepoints and gape times
	for t_i, train in spike_trains:
		if ~np.isnan(first_gapes[t_i][0]):
			f_i = plt.figure()
			plt.eventplot(train,alpha=0.5,color='k')
			x_ticks = plt.xticks()
			x_tick_labels = x_ticks - pre_taste_time
			plt.xticks(x_ticks,x_tick_labels)
			for cp in scaled_mode_tau:
				plt.axvline(cp,color='r')
			plt.fill_between(np.arange(first_gapes[t_i][0],first_gapes[t_i][1]),min(x_ticks),max(x_ticks),alpha=0.3,color='y')
			plt.title('Trial ' + str(t_i))
			plt.xlabel('Time from Taste Delivery (ms)')
			plt.ylabel('Neuron Index')
			plt.savefig(f_i,gape_align_cp_dir + given_name + '_trial_' + str(t_i) + '.png')
			plt.savefig(f_i,gape_align_cp_dir + 'trial_' + str(t_i) + '.svg')

#%% Plot changes in changepoint onsets across animals

tau_onsets, ax_onsets = plt.subplots(np.shape(tau_data[0][1]),figsize=(8,8))
cm_subsection = np.linspace(0,1,len(tau_data))
cmap = [cm.gist_rainbow(x) for x in cm_subsection]
for nf in range(len(tau_data)):
	nf_tau = tau_data[nf]
	for cp_i in range(np.shape(nf_tau)[1]):
		mu_tau = np.nanmean(nf_tau[:,cp_i])
		sig_tau = np.nanstd(nf_tau[:,cp_i])
		ax_onsets[cp_i].scatter(nf,mu_tau,color=cmap[nf])
		ax_onsets[cp_i].plot([nf,nf],[mu_tau-sig_tau,mu_tau+sig_tau],color=cmap[nf])
for ax_i in range(len(ax_onsets)):
	ax_onsets[ax_i].ylabel('Mean Onset (ms)')
	x_ticks = ax_onsets[ax_i].xticks()
	ax_onsets[ax_i].xticks(x_ticks,tau_data_names)
	ax_onsets[ax_i].x_label('Dataset')
	ax_onsets[ax_i].title('Changepoint ' + str(ax_i))
	ax_onsets[ax_i].legend()
tau_onsets.tight_layout()
plt.savefig(tau_onsets,cp_stats_dir + 'cp_onsets.png')
plt.savefig(tau_onsets,cp_stats_dir + 'cp_onsets.svg')
		


#%%

#first_sac_gapes: trial x time (onset and offset times of first gapes in each trial)
#first_gape_save_dir = os.path.join(anim_dir,'gape_onset_plots',anim_name + '_' + anim_taste_names[t_i] + '.npy')

first_sac_gapes = np.load('/media/cmazzio/large_data/CM26/CM26_CTATest_h2o_sac_nacl_qhcl_230819_101513/gape_onset_plots/CM26_CTATest_first_sac_gapes.npy')
first_qhcl_gapes = np.load('/media/cmazzio/large_data/CM26/CM26_CTATest2_h2o_sac_ca_qhcl_230820_121050/gape_onset_plots/CM26_CTATest2_first_qhcl_gapes.npy')

#pick out trials in which gaping occurred for each taste
sac_gape_trials = first_sac_gapes[:, 0]
qhcl_gape_trials = first_qhcl_gapes[:, 0]

#trial in which gaping occurred
sac_gape_onset = first_sac_gapes[:,1]
qhcl_gape_onset = first_qhcl_gapes[:,1]

#make tau array for only gape trials for each taste
sac_gape_tau = np.zeros(first_sac_gapes.shape)
num_trials = len(sac_gape_onset)
num_cp = len(sac_tau[0])
for trial_ind, trial in enumerate(sac_gape_trials):
    for cp in range(num_cp):
        this_tau = sac_tau[trial,cp]
        sac_gape_tau[trial_ind, cp] = this_tau
    
qhcl_gape_tau = np.zeros(first_qhcl_gapes.shape)
num_trials = len(qhcl_gape_onset)
num_cp = len(qhcl_tau[0])
for trial_ind, trial in enumerate(qhcl_gape_trials):
    for cp in range(num_cp):
        this_tau = qhcl_tau[trial,cp]
        qhcl_gape_tau[trial_ind, cp] = this_tau    


#find mean onset of each changepoint FOR ONLY GAPE TRIALS and find which of those changepoints 
#the average onset to gaping for saccharin and quinine best aligns

#find mean onset of saccharin gaping from day 1  
mean_sac_gape_onset = np.mean(sac_gape_onset)

num_cp = len(sac_tau[0])
mean_sac_cp_onsets = np.zeros(num_cp)
sac_diffs = np.zeros(num_cp)
sac_cp_diffs = np.zeros(num_cp)
for cp_ind, cp in enumerate(range(num_cp)):
    this_onset_mean = np.mean(sac_gape_tau[:,cp])
    mean_sac_cp_onsets[cp_ind] = this_onset_mean
    this_diff = abs(this_onset_mean - mean_sac_gape_onset)
    sac_diffs[cp_ind] = this_diff
# find changepoint time that is closest to mean onset of gaping
sac_gape_cp = np.argmin(sac_diffs)
print('Transition aligned with sac gape onset: ' + str(sac_gape_cp))

mean_qhcl_gape_onset = np.mean(qhcl_gape_onset)

num_cp = len(qhcl_tau[0])
mean_qhcl_cp_onsets = np.zeros(num_cp)
qhcl_diffs = np.zeros(num_cp)
qhcl_cp_diffs = np.zeros(num_cp)
for cp_ind, cp in enumerate(range(num_cp)):
    this_onset_mean = np.mean(qhcl_gape_tau[:,cp])
    mean_qhcl_cp_onsets[cp_ind] = this_onset_mean
    this_diff = abs(this_onset_mean - mean_qhcl_gape_onset)
    qhcl_diffs[cp_ind] = this_diff

qhcl_gape_cp = np.argmin(qhcl_diffs)
print('Transition aligned with qhcl gape onset: ' + str(qhcl_gape_cp))

#for plotting the gape trials below

#pull out spike trains for only gape trials
d1 = len(sac_gape_trials)
d2 = day1_spike_train.shape[1]
d3 = day1_spike_train.shape[2]
day1_gape_spike_train = np.zeros(shape = (d1,d2,d3))
for trial_ind, trial in enumerate(sac_gape_trials):
    this_spike_train = day1_spike_train[trial]
    day1_gape_spike_train[trial_ind] = this_spike_train


d1 = len(qhcl_gape_trials)
d2 = day2_spike_train.shape[1]
d3 = day2_spike_train.shape[2]
day2_gape_spike_train = np.zeros(shape = (d1,d2,d3))
for trial_ind, trial in enumerate(qhcl_gape_trials):
    this_spike_train = day2_spike_train[trial]
    day2_gape_spike_train[trial_ind] = this_spike_train


#find mean onset of each changepoint FOR ALL TRIALS and find which of those changepoints 
#the average onset to gaping for saccharin and quinine best aligns

#find the mean onset of each changepoint across all day 1 taste trials 
# num_cp = len(sac_tau[0])
# mean_sac_cp_onsets = np.zeros(num_cp)
# sac_diffs = np.zeros(num_cp)
# sac_cp_diffs = np.zeros(num_cp)
# for cp_ind, cp in enumerate(range(num_cp)):
#     this_onset_mean = np.mean(sac_tau[:,cp])
#     mean_sac_cp_onsets[cp_ind] = this_onset_mean
#     this_diff = abs(this_onset_mean - mean_sac_gape_onset)
#     sac_diffs[cp_ind] = this_diff

# #find changepoint time that is closest to mean onset of gaping
# sac_gape_cp = np.argmin(sac_diffs)
# print('Transition aligned with sac gape onset: ' + str(sac_gape_cp))

#find mean onset of quinine gaping from day 2
# mean_qhcl_gape_onset = np.mean(qhcl_gape_onset)

# #find the mean onset of each changepoint across all day 1 taste trials 
# num_cp = len(qhcl_tau[0])
# mean_qhcl_cp_onsets = np.zeros(num_cp)
# qhcl_diffs = np.zeros(num_cp)
# qhcl_cp_diffs = np.zeros(num_cp)
# for cp_ind, cp in enumerate(range(num_cp)):
#     this_onset_mean = np.mean(qhcl_tau[:,cp])
#     mean_qhcl_cp_onsets[cp_ind] = this_onset_mean
#     this_diff = abs(this_onset_mean - mean_qhcl_gape_onset)
#     qhcl_diffs[cp_ind] = this_diff

# qhcl_gape_cp = np.argmin(qhcl_diffs)
# print('Transition aligned with qhcl gape onset: ' + str(qhcl_gape_cp))

#%%PLOTTING CHANGEPOINT MODEL

#plotting for all trials
fig, ax = plotting.plot_changepoint_raster(day1_spike_train, sac_tau+2000, [1500, 4000])
plt.show()

fig, ax = plotting.plot_changepoint_raster(day2_spike_train, qhcl_tau+2000, [1500, 4000])
plt.show()

fig, ax = plotting.plot_state_firing_rates(day1_spike_train, sac_tau+2000)
plt.show()

fig, ax = plotting.plot_state_firing_rates(day2_spike_train, qhcl_tau+2000)
plt.show()


# plotting for gape trials only
fig, ax = plotting.plot_changepoint_raster(day1_gape_spike_train, sac_gape_tau+2000, [1500, 4000])
plt.show()

fig, ax = plotting.plot_changepoint_raster(day2_gape_spike_train, qhcl_gape_tau+2000, [1500, 4000])
plt.show()

fig, ax = plotting.plot_state_firing_rates(day1_gape_spike_train, sac_gape_tau+2000)
plt.show()

fig, ax = plotting.plot_state_firing_rates(day2_gape_spike_train, qhcl_gape_tau+2000)
plt.show()



fig, ax = plotting.plot_changepoint_overview(sac_gape_tau, [1500, 4000])
plt.show()

fig, ax = plotting.plot_aligned_state_firing(day1_gape_spike_train, sac_gape_tau, 300)
plt.show()

#%% PLOTTING CHANGEPOINT & EMG OVERLAY

#load emg filt data
day1_emg_filt = np.load('/media/cmazzio/large_data/CM26/CM26_CTATest_h2o_sac_nacl_qhcl_230819_101513/emg_output/emg/emg_filt.npy')

#get emg filt data for only sac trials
sac_emg_filt = day1_emg_filt[1,:,:]

#get emg filt data for only sac gape trials
num_wanted_trials = len(sac_gape_trials)
times = len(sac_emg_filt[1])
sac_gape_emg_filt = np.zeros(shape = (num_wanted_trials, sac_emg_filt.shape[1]))
for trial_ind, trial in enumerate(sac_gape_trials):
    for time_ind, time in enumerate(range(times)):
        this_gape_emg_filt = sac_emg_filt[trial,time]
        sac_gape_emg_filt[trial_ind,time_ind] = this_gape_emg_filt

#make axes
fig,ax = plt.subplots(sac_emg_filt.shape[0],1, 
                       sharex=True, sharey=True,
                       figsize = (10, sac_emg_filt.shape[0]))
#plot tau over emg traces
i =0
for this_dat, this_ax, this_tau in zip(sac_emg_filt, ax.flatten(), sac_tau+2000):
    this_ax.plot(this_dat)
    for x in this_tau:
        
        this_ax.axvline(x, color = 'red')
    if i in sac_gape_trials:
        index_ = np.where(sac_gape_trials == i)[0][0]
        this_ax.scatter(sac_gape_onset[index_]+2000, 800, s=50, c='green')
    i = i+1

#for quinine

#load emg filt data
day2_emg_filt = np.load('/media/cmazzio/large_data/CM26/CM26_CTATest2_h2o_sac_ca_qhcl_230820_121050/emg_output/emg/emg_filt.npy')

#get emg filt data for only qhcl trials
qhcl_emg_filt = day2_emg_filt[3,:,:]

#get emg filt data for only qhcl gape trials
num_wanted_trials = len(qhcl_gape_trials)
times = len(qhcl_emg_filt[1])
qhcl_gape_emg_filt = np.zeros(shape = (num_wanted_trials, qhcl_emg_filt.shape[1]))
for trial_ind, trial in enumerate(qhcl_gape_trials):
    for time_ind, time in enumerate(range(times)):
        this_gape_emg_filt = qhcl_emg_filt[trial,time]
        qhcl_gape_emg_filt[trial_ind,time_ind] = this_gape_emg_filt

#make axes
fig,ax = plt.subplots(qhcl_emg_filt.shape[0],1, 
                       sharex=True, sharey=True,
                       figsize = (10, qhcl_emg_filt.shape[0]))
#plot tau over emg traces
i =0
for this_dat, this_ax, this_tau in zip(qhcl_emg_filt, ax.flatten(), qhcl_tau+2000):
    this_ax.plot(this_dat)
    for x in this_tau:
        
        this_ax.axvline(x, color = 'red')
    if i in qhcl_gape_trials:
        index_ = np.where(qhcl_gape_trials == i)[0][0]
        this_ax.scatter(qhcl_gape_onset[index_]+2000, 800, s=50, c='green')
    i = i+1


#%%
###############################################################################
###plotting example trials for presentation
####specific trials for saccharin
plot_trials_sac = [1,2,4,6]
time_vec_sac = np.arange(1500,4001)
new_sac_gape_tau = sac_gape_tau +2000

color_list = ['lightcoral','teal', 'red']

fig,ax = plt.subplots(len(plot_trials_sac),1,
        sharey=True, sharex=True, figsize = (15,20))

for trial in range(len(plot_trials_sac)):
    ax[trial].plot(time_vec, sac_emg_filt[plot_trials_sac[trial], time_vec], color ='black')


#for overlaying epochs on EMG traces    
    for cp in range(3): 
        start = new_sac_gape_tau[trial,cp]
        if cp < 2:
            end = new_sac_gape_tau[trial,cp+1]
        else:
            end = time_vec[-1]
        ax[trial].axvspan(start, end, color = color_list[cp], alpha = 0.7)

    #ax[trial].set_ylim([1500, 4000])
    ax[trial].tick_params(axis="y", labelsize=20)
    ax[trial].axvline(2000, color='gray', linewidth = 4, linestyle = 'dashed')


#quinine    
plot_trials_qhcl = [1,2,9,12]  
time_vec_qhcl = np.arange(1500,4001)
new_sac_gape_tau = sac_gape_tau +2000  

fig,ax = plt.subplots(len(plot_trials_qhcl),1,
        sharey=True, sharex=True, figsize = (15,20))

for trial in range(len(plot_trials_qhcl)):
    ax[trial].plot(time_vec, qhcl_emg_filt[plot_trials_qhcl[trial], time_vec], color ='black')








    
 






#%%
#inds = list(np.ndindex(ax.shape))
for trial in range(len(plot_trials_sac)):
    #plot emg signal
    ax[trial].plot(time_vec, cut_emg_filt[taste_index,wanted_trials[trial]].flatten(), color ='black')
    #plot changepoints
    for tau_i in range(model_parameters['states']-2):
        start = emg_tau[wanted_trials[trial]][tau_i]
        if tau_i < model_parameters['states']-1:
            end = emg_tau[wanted_trials[trial]][tau_i+1]
        else:
            end = time_vec[-1]
        ax[trial].axvspan(start, end,
            color= color_list[tau_i],
            alpha = 0.7)
    #set axis range
    ax[trial].set_ylim([-1000, 1000])
    ax[trial].tick_params(axis="y", labelsize=25)
    ax[trial].axvline(0, color='gray', linewidth = 4, linestyle = 'dashed')
 
#    ax[trial].plt.x_ticks(fontsize=25)
#    ax[trial].tick_params(axis='y', which='minor', labelsize=20 )
#    ax[trial].set_ylabel('hi')
    if trial == 0:
        this_taste = tastes[taste_index]
#        ax[trial].set_title(this_taste)
    if trial == emg_filt.shape[1]-1:
        ax[trial].set_xlabel('Time post-stim (ms)')
#plt.suptitle('Red --> Not significant, Blue --> Significant')
plt.subplots_adjust(top = 0.95)
#plt.yticks(fontsize = 20)
#plt.ylabel('EMG signal', fontweight = 'bold')
plt.xlabel('Time from stimulus delivery (ms)', fontsize=30, fontweight= 'bold')
ax[1].set_ylabel('EMG signal (microvolts)', fontsize=30, fontweight = 'bold')
plt.xticks(fontsize=25)
plt.show()
fig.savefig('emg_filtered_plots_with_changepoint_select_trials' + str(tastes[taste_index]) + '.png', bbox_inches = 'tight')
#plt.close(fig)









# #make axes
# fig,ax = plt.subplots(qhcl_gape_emg_filt.shape[0],1, 
#                        sharex=True, sharey=True,
#                        figsize = (10, qhcl_gape_emg_filt.shape[0]))
# #plot tau over emg traces
# for this_dat, this_ax, this_tau in zip(qhcl_gape_emg_filt, ax.flatten(), qhcl_gape_tau+2000):
#     this_ax.plot(this_dat)
#     for x in this_tau:
#         this_ax.axvline(x, color = 'red')






# fig,ax = plt.subplots(sac_emg_filt.shape[0],1, 
#                        sharex=True, sharey=True,
#                        figsize = (10, sac_emg_filt.shape[0]))














#%% OLD PLOTTING

plt.figure(figsize =(10,8))
plt.bar(1, mean_test_sac_diff[wanted_test_sac_transition], yerr=se_sac_transition , width = 0.4, color = 'orchid')
plt.bar(2, mean_test_qhcl_diff[wanted_test_qhcl_transition], yerr=se_qhcl_transition, width = 0.4, color = 'orange')
plt.xticks([1,2], ['saccharin', 'quinine'], fontsize =20, fontweight = 'bold')
plt.yticks(fontsize=20)
plt.xlabel('Tastes', fontsize = 20, fontweight = 'bold')
plt.ylabel('Mean transition 1 onset time (ms)', fontsize = 20, fontweight = 'bold')

#run non-parametric t test 
from scipy.stats import mannwhitneyu
stat, p = mannwhitneyu(qhcl_tau_gape_trials, sac_tau_gape_trials)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


#find correlation between gape onset time and transition 1 time for sac
# Import libraries
import pandas as pd
from scipy.stats import pearsonr
 
sac_qhcl_tau_gape_trials= sac_tau_gape_trials + qhcl_tau_gape_trials
sac_qhcl_gape_onsets_gape_trials= sac_gape_onsets_gape_trials + qhcl_gape_onsets_gape_trials

#calculate correlation between gape onset time and transition 1 time for qhcl and sac together 
corr, p = pearsonr(sac_qhcl_tau_gape_trials, sac_qhcl_gape_onsets_gape_trials)

# =============================================================================
# inds = np.array(sac_qhcl_gape_onsets_gape_trials) < 750
# pearsonr(np.array(sac_qhcl_tau_gape_trials)[inds], 
#          np.array(sac_qhcl_gape_onsets_gape_trials)[inds])
# =============================================================================

print(f'Pearsons correlation sac & qhcl: {round(corr, 3)}')
print(f'p-value sac & qhcl: {round(p,3)}')

#plot 
plt.figure(figsize=(10,8))
plt.scatter(sac_qhcl_tau_gape_trials,sac_qhcl_gape_onsets_gape_trials, color='black')
plt.plot([0,2000], [0,2000], color='gray', linestyle='dashed')
plt.xlabel('Frist transition start time (ms)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=20)
plt.ylabel('First gape onset time (ms)', fontsize=20, fontweight ='bold')
plt.yticks(fontsize=20)



# sac correlation of gape onset times and transition 1 times
corr, p = pearsonr(sac_tau_gape_trials, sac_gape_onsets_gape_trials)
print(f'Pearsons correlation sac: {round(corr, 3)}')
print(f'p-value sac: {round(p,3)}')

#quinine correlation of gape onset times and transition 1 times
# corr, p = pearsonr(qhcl_tau_gape_trials, qhcl_gape_onsets_gape_trials)
# print(f'Pearsons correlation qhcl: {round(corr, 3)}')
# print(f'p-value qhcl: {round(p,3)}')

plt.figure(figsize=(10,8))
plt.scatter(sac_tau_gape_trials,sac_gape_onsets_gape_trials, color='black')
#plt.scatter(qhcl_tau_gape_trials,qhcl_gape_onsets_gape_trials, color='black')
plt.plot([0,2000], [0,2000], color='gray', linestyle='dashed')
plt.xlabel('Frist transition start time (ms)', fontsize=20, fontweight='bold')
plt.xticks(fontsize=20)
plt.ylabel('First gape onset time (ms)', fontsize=20, fontweight ='bold')
plt.yticks(fontsize=20)


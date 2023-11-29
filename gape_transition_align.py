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
sys.path.append('/home/cmazzio/Desktop/blech_clust/')
sys.path.append('/home/cmazzio/Desktop/pytau/')
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
	print("\tThere are " + str(len(taste_list)) + " tastes available.")
	print("\tAvailable taste indices:")
	print(list(taste_list))
	taste_ind = int_input("\tWhich taste do you want from this list? ")
	wanted_frame = wanted_frame[wanted_frame['data.taste_num'] == taste_ind]
	taste_list = wanted_frame['data.taste_num']
	pkl_path_list = wanted_frame['exp.save_path']
	this_handler = PklHandler(pkl_path_list[0])
	#Import changepoints for each delivery
	scaled_mode_tau = this_handler.tau.scaled_mode_tau #num trials x num cp
	#Import Hannah's CPs for each delivery
	hannah_tau_dir = os.path.join(data_dir,'Changepoint_Calculations','All_Taste_CPs','pop')
	hannah_tau_dir_files = os.listdir(hannah_tau_dir)
	for filename in hannah_tau_dir_files:
		if filename[-4:] == '.npy':
			bool_val = bool_input('\tIs ' + filename + ' the correct associated cp file?')
			if bool_val == 'y':
				hannah_cp =  np.load(os.path.join(hannah_tau_dir,filename))
				break
	#Import spikes following each taste delivery
	spike_train = this_handler.firing.raw_spikes
	#Store changepoint and spike data in dictionary
	tau_data_dict[nf] = dict()
	tau_data_dict[nf]['data_dir'] = data_dir
	name_list = wanted_frame['data.basename']
	print("\tGive a more colloquial name to the dataset.")
	given_name = input("\tHow would you rename " + list(name_list)[0] + "? ")
	tau_data_dict[nf]['given_name'] = given_name
	tau_data_dict[nf]['taste_list'] = taste_list
	tau_data_dict[nf]['states'] = desired_states
	tau_data_dict[nf]['scaled_mode_tau'] = scaled_mode_tau
	try:
		tau_data_dict[nf]['hannah_cp'] = hannah_cp
	except:
		"no hannah cp data found"
	tau_data_dict[nf]['spike_train'] = spike_train
	#Import associated emg data
	try:
		emg_filt = np.load(os.path.join(data_dir,'emg_output','emg','emg_filt.npy'))
		tau_data_dict[nf]['emg_filt'] = emg_filt
		print("\tEmg filtered data successfully imported for dataset.")
	except:
		print("\tEmg filtered data not imported successfully.")
		bool_val = bool_input("\tIs there an emg_filt.npy file to import?")
		if bool_val == 'y':
			emg_filt_data_dir = easygui.diropenbox(title='\tPlease select the folder where emg_filt.npy is stored.')
			emg_filt = np.load(os.path.join(emg_filt_data_dir,'emg_filt.npy'))
			tau_data_dict[nf]['emg_filt'] = emg_filt
		else:
			print("\tMoving On.")
	#Import associated gapes
	print("\tNow import associated first gapes data with this dataset.")
	gape_data_dir = easygui.diropenbox(title='\tPlease select the folder where data is stored.')
	#Search for matching file type - ends in _gapes.npy
	files_in_dir = os.listdir(gape_data_dir)
	print("There are " + str(len(files_in_dir)) + " gape files in this folder.")
	for filename in files_in_dir:
		if filename[-10:] == '_gapes.npy':
			bool_val = bool_input("\tIs " + filename + " the correct associated file with " + given_name + "?")
			if bool_val == 'y':
				first_gapes =  np.load(os.path.join(gape_data_dir,filename))
				tau_data_dict[nf]['first_gapes'] = first_gapes
				break
	try: #Check that something was imported
		first_gapes = tau_data_dict[nf]['first_gapes']
	except:
		print('First gapes file not found/selected in given folder. Did you run gape_onset.py before?')
		print('You may want to quit this program now - it will break in later code blocks by missing this data.')
	
#Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')

#Save dictionary
dict_save_dir = os.path.join(results_dir,'tau_dict.pkl')
f = open(dict_save_dir,"wb")
pickle.dump(tau_data_dict,f)
#with open(dict_save_dir, "rb") as pickle_file:
#	tau_data_dict = pickle.load(pickle_file)

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
max_num_tau = 0
tau_data = []
tau_data_names = []
first_gapes_data = []
preceding_transitions = []
preceding_transitions_hannah = []
for nf in range(len(tau_data_dict)):
	given_name = tau_data_dict[nf]['given_name']
	tau_data_names.extend([given_name])
	#load changepoint information
	scaled_mode_tau = tau_data_dict[nf]['scaled_mode_tau']
	hannah_cp = tau_data_dict[nf]['hannah_cp']
	#Plot the two sets of changepoints side by side
	hannah_cp_scaled = hannah_cp[:,1:] - np.expand_dims(hannah_cp[:,0],1)
	f_cp = plt.figure(figsize=(8,8))
	num_cp_plot = np.shape(scaled_mode_tau)[1]
	for cp_i in range(num_cp_plot):
		plt.subplot(num_cp_plot,1,cp_i + 1)
		plt.hist(scaled_mode_tau[:,cp_i] - pre_taste_time,color='blue',alpha=0.5,label='Tau')
		plt.hist(hannah_cp_scaled[:,cp_i],color='green',alpha=0.5,label='CP')
		plt.legend()
		plt.title('CP ' + str(cp_i))
	plt.tight_layout()
	f_cp.savefig(os.path.join(cp_stats_dir,given_name + '_cp_distributions.png'))
	f_cp.savefig(os.path.join(cp_stats_dir,given_name + '_cp_distributions.svg'))
	plt.close(f_cp)
	#Calculate cp preceding gape
	taste_list = tau_data_dict[nf]['taste_list']
	tau_data.append(scaled_mode_tau - pre_taste_time)
	#load first gapes information 
	first_gapes = tau_data_dict[nf]['first_gapes']
	first_gapes_data.append(first_gapes)
	#transition preceding each trial's first gape
	pre_cp_i = np.zeros(np.shape(first_gapes)[0]) #From scaled mode tau
	pre_cp_i_hannah = np.zeros(np.shape(first_gapes)[0]) #From scaled mode tau
	for fg_i, fg_times in enumerate(first_gapes):
		#scaled mode tau
		trial_tau = scaled_mode_tau[fg_i] - pre_taste_time
		if len(trial_tau) > max_num_tau:
			max_num_tau = len(trial_tau)
		trial_gape_onset = fg_times[0]
		if ~np.isnan(trial_gape_onset):
			try:
				pre_cp = np.where(trial_gape_onset - trial_tau > 0)[0][-1]
				pre_cp_i[fg_i] = pre_cp + 1
			except:
				try:
					pre_cp = np.where(trial_gape_onset > 0)[0][-1]
					pre_cp_i[fg_i] = 0
				except:
					pre_cp_i[fg_i] = np.nan
		else:
			pre_cp_i[fg_i] = np.nan
		#hannah cp
		trial_cp = hannah_cp[fg_i,1:] - hannah_cp[fg_i,0]
		if len(trial_cp) > max_num_tau:
			max_num_tau = len(trial_tau)
		if ~np.isnan(trial_gape_onset):
			try:
				pre_cp = np.where(trial_gape_onset - trial_cp > 0)[0][-1]
				pre_cp_i_hannah[fg_i] = pre_cp + 1
			except:
				try:
					pre_cp = np.where(trial_gape_onset > 0)[0][-1]
					pre_cp_i_hannah[fg_i] = 0
				except:
					pre_cp_i_hannah[fg_i] = np.nan
		else:
			pre_cp_i_hannah[fg_i] = np.nan
	preceding_transitions.append(pre_cp_i)
	preceding_transitions_hannah.append(pre_cp_i_hannah)
	f_pre = plt.figure()
	plt.subplot(1,2,1)
	plt.hist(pre_cp_i)
	plt.title('Tau Index Pre-Gape')
	plt.xlabel('Changepoint Index')
	plt.ylabel('Number of Trials')
	plt.subplot(1,2,2)
	plt.hist(pre_cp_i_hannah)
	plt.title('CP Index Pre-Gape')
	plt.xlabel('Changepoint Index')
	plt.ylabel('Number of Trials')
	plt.suptitle('Changepoint Index Preceding First Gape')
	f_pre.savefig(os.path.join(cp_stats_dir,given_name + '_cp_preceding_first_gape.png'))
	f_pre.savefig(os.path.join(cp_stats_dir,given_name + '_cp_preceding_first_gape.svg'))
	plt.close(f_pre)
	#spike trains for trial
	spike_trains = tau_data_dict[nf]['spike_train']
	#plot spike train with overlaid changepoints and gape times
	for t_i, train in enumerate(spike_trains):
		if ~np.isnan(first_gapes[t_i][0]): #Only plot if gape occurs
			num_neur = np.shape(train)[0]
			train_indices = [list(np.where(train[n_i] == 1)[0]) for n_i in range(num_neur)]
			f_i = plt.figure()
			plt.eventplot(train_indices,alpha=0.5,color='k')
			x_ticks = plt.xticks()
			x_tick_labels = x_ticks[0] - pre_taste_time
			plt.xticks(x_ticks[0],x_tick_labels)
			for cp in scaled_mode_tau[t_i]:
				plt.axvline(cp,color='r')
			plt.fill_between(np.arange(first_gapes[t_i][0] + pre_taste_time,first_gapes[t_i][1] + pre_taste_time),0,num_neur,alpha=0.3,color='y')
			plt.title('Trial ' + str(t_i))
			plt.xlabel('Time from Taste Delivery (ms)')
			plt.ylabel('Neuron Index')
			f_i.savefig(os.path.join(gape_align_cp_dir,given_name + '_trial_' + str(t_i) + '.png'))
			f_i.savefig(os.path.join(gape_align_cp_dir,given_name + '_trial_' + str(t_i) + '.svg'))
			plt.close(f_i)
	#plot spike train with overlaid hannah changepoints and gape times
	for t_i, train in enumerate(spike_trains):
		if ~np.isnan(first_gapes[t_i][0]): #Only plot if gape occurs
			num_neur = np.shape(train)[0]
			train_indices = [list(np.where(train[n_i] == 1)[0]) for n_i in range(num_neur)]
			f_i = plt.figure()
			plt.eventplot(train_indices,alpha=0.5,color='k')
			x_ticks = plt.xticks()
			x_tick_labels = x_ticks[0] - pre_taste_time
			plt.xticks(x_ticks[0],x_tick_labels)
			trial_cp = hannah_cp[t_i,1:] - hannah_cp[t_i,0]
			for cp in trial_cp:
				plt.axvline(cp + pre_taste_time,color='r')
			plt.fill_between(np.arange(first_gapes[t_i][0] + pre_taste_time,first_gapes[t_i][1] + pre_taste_time),0,num_neur,alpha=0.3,color='y')
			plt.title('Trial ' + str(t_i))
			plt.xlabel('Time from Taste Delivery (ms)')
			plt.ylabel('Neuron Index')
			f_i.savefig(os.path.join(gape_align_cp_dir,given_name + '_trial_' + str(t_i) + '_hannah_cp.png'))
			f_i.savefig(os.path.join(gape_align_cp_dir,given_name + '_trial_' + str(t_i) + '_hannah_cp.svg'))
			plt.close(f_i)

#%% Plot changes in changepoint onsets across animals

tau_onsets, ax_onsets = plt.subplots(max_num_tau,figsize=(8,8))
for nf in range(len(tau_data)):
	nf_tau = tau_data[nf]
	for cp_i in range(np.shape(nf_tau)[1]):
		mu_tau = np.nanmean(nf_tau[:,cp_i])
		sig_tau = np.nanstd(nf_tau[:,cp_i])
		ax_onsets[cp_i].scatter(nf,mu_tau,color='k')
		ax_onsets[cp_i].plot([nf,nf],[mu_tau-sig_tau,mu_tau+sig_tau],color='k')
for ax_i in range(len(ax_onsets)):
	ax_onsets[ax_i].set_ylabel('Mean Onset (ms)')
	plt.figure(ax_i+1)
	ax_onsets[ax_i].set_xticks(np.arange(len(tau_data_names)))
	ax_onsets[ax_i].set_xticklabels(labels=tau_data_names)
	ax_onsets[ax_i].set_xlabel('Dataset')
	ax_onsets[ax_i].set_title('Changepoint ' + str(ax_i))
tau_onsets.tight_layout()
tau_onsets.savefig(os.path.join(cp_stats_dir,'cp_onsets.png'))
tau_onsets.savefig(os.path.join(cp_stats_dir,'cp_onsets.svg'))

#%% Plot preceding transition stats

cp_labels = ['CP ' + str(i) for i in range(max_num_tau+1)]

#Histogram
plt.figure()
n = plt.hist(preceding_transitions,label=tau_data_names)
plt.title('Transition Immediately Preceding Gape Onset')
plt.legend()
plt.xticks(np.arange(max_num_tau+1),labels=cp_labels)
plt.xlabel('Changepoint')
plt.ylabel('Number of Gape Trials')
plt.tight_layout()
plt.savefig(os.path.join(cp_stats_dir,'preceding_tau_hist.png'))
plt.savefig(os.path.join(cp_stats_dir,'preceding_tau_hist.svg'))

#Histogram
plt.figure()
n = plt.hist(preceding_transitions_hannah,label=tau_data_names)
plt.title('Transition Immediately Preceding Gape Onset')
plt.legend()
plt.xticks(np.arange(max_num_tau+1),labels=cp_labels)
plt.xlabel('Changepoint')
plt.ylabel('Number of Gape Trials')
plt.tight_layout()
plt.savefig(os.path.join(cp_stats_dir,'preceding_cp_hist.png'))
plt.savefig(os.path.join(cp_stats_dir,'preceding_cp_hist.svg'))


#Pie charts
f_pie, ax_pie = plt.subplots(len(preceding_transitions), figsize=(10,10))
for nf in range(len(preceding_transitions)):
	nf_pt = np.array(preceding_transitions[nf])
	nf_pt_nan = np.isnan(nf_pt)
	nf_pt_nonan = nf_pt[np.where(~nf_pt_nan)[0]]
	cp_counts = np.zeros(max_num_tau+1)
	for cp_i in range(max_num_tau+1):
		cp_counts[cp_i] = len(np.where(nf_pt_nonan == cp_i)[0])
	ax_pie[nf].pie(cp_counts,labels=cp_labels,autopct='%1.1f%%')
	ax_pie[nf].set_title(tau_data_names[nf])
plt.tight_layout()
plt.savefig(os.path.join(cp_stats_dir,'preceding_tau_pie.png'))
plt.savefig(os.path.join(cp_stats_dir,'preceding_tau_pie.svg'))

#Pie charts
f_pie, ax_pie = plt.subplots(len(preceding_transitions_hannah), figsize=(10,10))
for nf in range(len(preceding_transitions_hannah)):
	nf_pt = np.array(preceding_transitions_hannah[nf])
	nf_pt_nan = np.isnan(nf_pt)
	nf_pt_nonan = nf_pt[np.where(~nf_pt_nan)[0]]
	cp_counts = np.zeros(max_num_tau+1)
	for cp_i in range(max_num_tau+1):
		cp_counts[cp_i] = len(np.where(nf_pt_nonan == cp_i)[0])
	ax_pie[nf].pie(cp_counts,labels=cp_labels,autopct='%1.1f%%')
	ax_pie[nf].set_title(tau_data_names[nf])
plt.tight_layout()
plt.savefig(os.path.join(cp_stats_dir,'preceding_cp_pie.png'))
plt.savefig(os.path.join(cp_stats_dir,'preceding_cp_pie.svg'))

#Add plots on fraction of lower-cp index onsets than higher (somehow)

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
 
#	ax[trial].plt.x_ticks(fontsize=25)
#	ax[trial].tick_params(axis='y', which='minor', labelsize=20 )
#	ax[trial].set_ylabel('hi')
	if trial == 0:
		this_taste = tastes[taste_index]
#		ax[trial].set_title(this_taste)
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
#						sharex=True, sharey=True,
#						figsize = (10, qhcl_gape_emg_filt.shape[0]))
# #plot tau over emg traces
# for this_dat, this_ax, this_tau in zip(qhcl_gape_emg_filt, ax.flatten(), qhcl_gape_tau+2000):
#	 this_ax.plot(this_dat)
#	 for x in this_tau:
#		 this_ax.axvline(x, color = 'red')






# fig,ax = plt.subplots(sac_emg_filt.shape[0],1, 
#						sharex=True, sharey=True,
#						figsize = (10, sac_emg_filt.shape[0]))














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
#		  np.array(sac_qhcl_gape_onsets_gape_trials)[inds])
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:42:34 2024

@author: hannahgermaine
Test code to try to pull out different mouth movements from emg data
"""



import sys, pickle, easygui, os, tqdm, gzip
sys.path.append('/home/cmazzio/Desktop/blech_clust/')
from matplotlib import cm
import pylab as plt
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from scipy import stats, interpolate
from scipy.signal import find_peaks, peak_widths
from scipy.fft import fft, fftfreq
from functions.mouth_movement_funcs import *

#%% Load all files to be analyzed

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
	#	 if filename[-10:] == '_gapes.npy':
	#		 bool_val = bool_input("\tIs " + filename + " the correct associated file with " + given_name + "?")
	#		 if bool_val == 'y':
	#			 first_gapes =  np.load(os.path.join(gape_data_dir,filename))
	#			 emg_data_dict[nf]['first_gapes'] = first_gapes
	#			 break
	# try: #Check that something was imported
	#	 first_gapes = emg_data_dict[nf]['first_gapes']
	# except:
	#	 print('First gapes file not found/selected in given folder. Did you run gape_onset.py before?')
	#	 print('You may want to quit this program now - it will break in later code blocks by missing this data.')
	
#Analysis Storage Directory
print('Please select a directory to save all results from this set of analyses.')
results_dir = easygui.diropenbox(title='Please select the storage folder.')

#Save dictionary
dict_save_dir = os.path.join(results_dir,'emg_data_dict.pkl')
f = open(dict_save_dir,"wb")
pickle.dump(emg_data_dict,f)

#%% Take each enveloped emg signal and pull out individual movements to test

#emg_data_dict[dataset_num] contains a dictionary with keys "emg_filt" and "given_name"
#given_name is a string of the dataset name
#emg_filt is an array of size [num_tastes,num_trials,7000 ms] where 7000 ms is 2000 pre and 5000 post

pre_taste = 2000
post_taste = 5000
plot_post_taste = 2000
min_inter_peak_dist = 50 #min ms apart peaks have to be to count
min_gape_band = 4
max_gape_band = 6

for nf in range(len(emg_data_dict)):
	dataset_name =  emg_data_dict[nf]['given_name']
	try:
		all_taste_gapes = emg_data_dict[nf]['taste_gapes']
		print("Dataset " + dataset_name + " previously analyzed for gapes. ")
		re_run = bool_input("Would you like to re-run the analysis? ")
		if re_run == 'y':
			prev_run = 0
		else:
			prev_run = 1
			print("Continuing to next dataset.")
	except:
		prev_run = 0
	if prev_run == 0:
		print("Analyzing dataset " + dataset_name)
		emg_filt = emg_data_dict[nf]['emg_filt']
		env = emg_data_dict[nf]['env']
		[num_tastes, max_num_trials, num_time] = np.shape(emg_filt)
		emg_gapes = np.zeros((num_tastes,max_num_trials,num_time))
		taste_names = emg_data_dict[nf]['taste_names']
		all_taste_gapes = []
		for t_i in range(num_tastes):
			print("\t Taste " + taste_names[t_i])
			taste_save_dir = os.path.join(results_dir,taste_names[t_i])
			if not os.path.isdir(taste_save_dir):
				os.mkdir(taste_save_dir)
			taste_gapes = np.zeros((max_num_trials,pre_taste+post_taste))
			for tr_i in tqdm.tqdm(range(max_num_trials)):
				if not np.isnan(emg_filt[t_i,tr_i,0]): #Make sure a delivery actually happened - nan otherwise
					f, ax = plt.subplots(nrows=5,ncols=2,figsize=(10,10))
					gs = ax[4, 0].get_gridspec()
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
					#___Find edges of peaks using peak widths function
					[peak_ws,_,_,_] = peak_widths(tr_env-mu_env,peak_inds,rel_height=0.5) #This is half-height width so double for ~full width
					peak_left = np.array([np.max((peak_inds[p_i] - peak_ws[p_i],0)) for p_i in range(len(peak_inds))])
					peak_right = np.array([np.min((peak_inds[p_i] + peak_ws[p_i],pre_taste+post_taste-1)) for p_i in range(len(peak_inds))])
					#___Find frequency using FFT on 3-peak fit
					peak_freq_fft = np.zeros(len(peak_inds)) #Store instantaneous fft frequency
					for p_i in np.arange(1,len(peak_inds)-1):
						last_peak = peak_inds[p_i-1]
						next_peak = peak_inds[p_i+1]
						peak_form = np.expand_dims(tr_env[last_peak:next_peak],0)
						repeat_peak = peak_form
						num_repeat = np.ceil(1000/np.shape(repeat_peak)[1]).astype('int')
						for rp_i in range(num_repeat):
							if np.mod(rp_i,2) == 0:
								repeat_peak = np.concatenate((repeat_peak,np.fliplr(peak_form)),1)
							else:
								repeat_peak = np.concatenate((repeat_peak,peak_form),1)
						repeat_peak = repeat_peak.flatten()
						fft_repeat = fft(repeat_peak)
						fft_repeat_fr = fftfreq(len(fft_repeat),1/1000)
						fr_sort = np.argsort(fft_repeat_fr)
						fft_repeat_fr_sort = fft_repeat_fr[fr_sort]
						fft_sig_sort = np.abs(fft_repeat[fr_sort])
						fft_pos_fr = fft_repeat_fr_sort[fft_repeat_fr_sort>0]
						fft_pos_sig = fft_sig_sort[fft_repeat_fr_sort>0]
						freq_max = fft_pos_fr[np.argmax(fft_pos_sig)]
						peak_freq_fft[p_i] = freq_max
					peak_left = np.floor(peak_left).astype('int')
					peak_right = np.ceil(peak_right).astype('int')
					#Plot peak heights and widths as a check
					ax[1,0].plot(np.arange(-pre_taste,post_taste),tr_env)
					peak_freq = np.zeros(len(peak_inds)) #Store instantaneous frequency
					peak_amp = np.zeros(len(peak_inds))
					jit = np.random.rand()
					for i in range(len(peak_inds)):
						p_i = peak_inds[i]
						ax[1,0].axvline(p_i-pre_taste,color='k',linestyle='dashed',alpha=0.5)
						#w_i = peak_props['widths'][i]
						w_i = peak_right[i] - peak_left[i]
						peak_freq[i] = 1/(w_i/1000) #Calculate instantaneous frequency
						w_h = peak_props['width_heights'][i]
						if np.mod(i,2) == 0:
							ax[1,0].plot([peak_left[i]-pre_taste,peak_right[i]-pre_taste],[w_h + jit,w_h + jit],color='g',linestyle='dashed',alpha=0.5)
						else:
							ax[1,0].plot([peak_left[i]-pre_taste,peak_right[i]-pre_taste],[w_h,w_h],color='r',linestyle='dashed',alpha=0.5)
						peak_amp[i] = tr_env[p_i]
					ax[1,0].set_xlim([0,1000])
					ax[1,0].set_title('Zoom peak loc + wid')
					#Plot movement amplitude
					ax[1,1].plot(peak_inds-pre_taste,peak_amp)
					ax[1,1].axhline(mu_env+3*sig_env,linestyle='dashed',alpha=0.5)
					ax[1,1].set_title('Peak Amplitude')
					#Plot instantaneous frequency
					ax[2,0].plot(peak_inds-pre_taste,peak_freq)
					ax[2,0].axhline(min_gape_band,linestyle='dashed',color='g')
					ax[2,0].axhline(max_gape_band,linestyle='dashed',color='g')
					ax[2,0].set_title('Instantaneous Frequency')
					#Pull out gape intervals only where amplitude is above cutoff, and frequency in range
					gape_peak_inds = np.where((peak_amp>=mu_env+3*sig_env)*(min_gape_band<=peak_freq)*(peak_freq<=max_gape_band))[0]
					gape_starts = peak_left[gape_peak_inds]
					gape_ends = peak_right[gape_peak_inds]
					ax[2,1].plot(np.arange(-pre_taste,post_taste),tr_env)
					ax[2,1].axvline(0,color='k',linestyle='dashed')
					for gpi in range(len(gape_peak_inds)):
						x_vals = np.arange(gape_starts[gpi],gape_ends[gpi]) - pre_taste
						ax[2,1].fill_between(x_vals,min_env*np.ones(len(x_vals)),max_env*np.ones(len(x_vals)),color='r',alpha=0.2)
					ax[2,1].set_title('Enveloped EMG Gape Times')
					#Plot fast fourier transform frequencies
					gape_peak_inds_fft = np.where((peak_amp>=mu_env+3*sig_env)*(min_gape_band<=peak_freq_fft)*(peak_freq_fft<=max_gape_band))[0]
					gape_times_fft = peak_inds[gape_peak_inds_fft]
					gape_starts_fft = peak_left[gape_peak_inds_fft]
					gape_ends_fft = peak_right[gape_peak_inds_fft]
					for gtf in range(len(gape_times_fft)):
						taste_gapes[tr_i,gape_starts_fft[gtf]:gape_ends_fft[gtf]] = 1
					ax[3,0].plot(peak_inds-pre_taste,peak_freq_fft)
					for gtf in gape_times_fft:
						ax[3,0].scatter(gtf-pre_taste,(max_gape_band+min_gape_band)/2,marker='*',color='k',alpha=0.5)
					ax[3,0].axhline(min_gape_band,linestyle='dashed',color='g')
					ax[3,0].axhline(max_gape_band,linestyle='dashed',color='g')
					ax[3,0].set_title('Instantaneous FFT Frequency')
					#Plot FFT Gape Times
					ax[3,1].plot(np.arange(-pre_taste,post_taste),tr_env)
					ax[3,1].axvline(0,color='k',linestyle='dashed')
					for gpi in range(len(gape_peak_inds_fft)):
	 					x_vals = np.arange(gape_starts_fft[gpi],gape_ends_fft[gpi]) - pre_taste
	 					ax[3,1].fill_between(x_vals,min_env*np.ones(len(x_vals)),max_env*np.ones(len(x_vals)),color='r',alpha=0.2)
					ax[3,1].set_title('Enveloped EMG Gape Times')
					#Plot zoom in of gapes 2 seconds after taste delivery
					for ax_i in ax[-1,:]:
						ax_i.remove() #remove the underlying axes
					axbig = f.add_subplot(gs[-1,:])
					axbig.plot(np.arange(0,pre_taste),tr_env[pre_taste:pre_taste+pre_taste])
					for gpi in range(len(gape_times_fft)):
						if (gape_times_fft[gpi] >= pre_taste)*(gape_times_fft[gpi] <= pre_taste+pre_taste):
							x_vals = np.arange(gape_starts_fft[gpi],gape_ends_fft[gpi]) - pre_taste
							axbig.fill_between(x_vals,min_env*np.ones(len(x_vals)),max_env*np.ones(len(x_vals)),color='r',alpha=0.2)
							axbig.axvline(gape_times_fft[gpi]-pre_taste,linestyle='dashed',color='k',alpha=0.2)
					#Clean up and save
					f.tight_layout()
					f.savefig(os.path.join(taste_save_dir,'emg_gapes_trial_' + str(tr_i) + '.png'))
					f.savefig(os.path.join(taste_save_dir,'emg_gapes_trial_' + str(tr_i) + '.svg'))
					plt.close(f)
			all_taste_gapes.append(taste_gapes)	
		emg_data_dict[nf]['taste_gapes'] = all_taste_gapes
		
		#Plot all taste gapes as image
		xticks = np.arange(0,pre_taste+post_taste,1000)
		f, ax = plt.subplots(nrows=1,ncols=num_tastes,figsize=(3*num_tastes,3))
		for t_i in range(num_tastes):
			ax[t_i].imshow(all_taste_gapes[t_i],aspect='auto')
			ax[t_i].axvline(pre_taste,color='w',linestyle='dashed')
			ax[t_i].set_xticks(np.array(xticks))
			ax[t_i].set_xticklabels(list(np.round(xticks-pre_taste,2).astype('str')),rotation=45)
			ax[t_i].set_title(taste_names[t_i])
			ax[t_i].set_xlim([0,pre_taste+plot_post_taste])
		f.tight_layout()
		f.savefig(os.path.join(results_dir,'emg_gapes_fft.png'))
		f.savefig(os.path.join(results_dir,'emg_gapes_fft.svg'))
		plt.close(f)
	
#Save updated dictionary
f = open(dict_save_dir,"wb")
pickle.dump(emg_data_dict,f)	
	
#%% Cluster and plot all the fft gapes			
				
norm_width = 500 #normalized width of gapes for clustering
gape_time_cutoff = 2000 #Number of ms after taste delivery to analyze gapes up to

for nf in range(len(emg_data_dict)):
	dataset_name =  emg_data_dict[nf]['given_name']
	emg_filt = emg_data_dict[nf]['emg_filt']
	env = emg_data_dict[nf]['env']
	[num_tastes, max_num_trials, num_time] = np.shape(emg_filt)
	taste_names = emg_data_dict[nf]['taste_names']
	all_taste_gapes = emg_data_dict[nf]['taste_gapes']
	#_____Collect gape waveform information_____
	#Store waveform tastes
	gape_tastes = []
	#Store waveform start times
	gape_start_times = [] #in ms from taste delivery
	#Store original emg waveforms
	norm_width_gape_storage = [] #store width-normalized indiv gape waveforms
	norm_full_gape_storage = [] #store fully-normalized indiv gape waveforms
	#Store enveloped waveforms
	norm_width_env_gape_storage = [] #store width-normalized indiv gape waveforms
	norm_full_env_gape_storage = [] #store fully-normalized indiv gape waveforms
	for t_i in range(num_tastes):
		taste_gapes = all_taste_gapes[t_i]
		for tr_i in range(max_num_trials):
			trial_gapes_bin = taste_gapes[tr_i,:]
			trial_gapes_diff = np.diff(trial_gapes_bin)
			gape_starts = np.where(trial_gapes_diff == 1)[0] + 1
			gape_ends = np.where(trial_gapes_diff == -1)[0] + 1
			for gs in gape_starts:
				try:
					ge = gape_ends[np.where((gape_ends-gs > 0))[0][0]]
				except:
					pass
				if gs-pre_taste <= gape_time_cutoff:
					gape_tastes.extend([t_i])            
					gape_start_times.extend([gs-pre_taste]) #store in ms from taste delivery
    				#Get original waveforms
					gape_emg = list(emg_filt[t_i,tr_i,gs:ge])
					x_gape = np.arange(len(gape_emg))
					gape_env = list(env[t_i,tr_i,gs:ge])
    				#Normalize length waveforms
					bin_centers = np.linspace(0,len(gape_emg),norm_width)
					fit_gape_emg_norm_len = interpolate.CubicSpline(x_gape,gape_emg)
					gape_emg_norm_len = fit_gape_emg_norm_len(bin_centers)
					norm_width_gape_storage.append(gape_emg_norm_len)
					fit_gape_env_norm_len = interpolate.CubicSpline(x_gape,gape_env)
					gape_env_norm_len = fit_gape_env_norm_len(bin_centers)
					norm_width_env_gape_storage.append(gape_env_norm_len)
    				#Normalize height and length waveforms
					gape_emg_full_norm = gape_emg_norm_len/np.max(gape_emg_norm_len)
					norm_full_gape_storage.append(gape_emg_full_norm)
					gape_env_full_norm = gape_env_norm_len/np.max(gape_env_norm_len)
					norm_full_env_gape_storage.append(gape_env_full_norm)
	emg_data_dict[nf]['gape_start_times'] = gape_start_times
	emg_data_dict[nf]['gape_tastes'] = gape_tastes
	emg_data_dict[nf]['norm_full_env_gape_storage'] = norm_full_env_gape_storage
	emg_data_dict[nf]['norm_width_env_gape_storage'] = norm_width_env_gape_storage
	
	clust_save_dir = os.path.join(results_dir,'Clustering')
	if not os.path.isdir(clust_save_dir):
		os.mkdir(clust_save_dir)

	#Run cluster analysis with UMAP
	#Normalized width clustering
	umap_env_norm_width_n_clusters,umap_env_norm_width_clust_centers,umap_env_norm_width_labels,\
	umap_env_norm_width_2D,umap_env_norm_width_clust_centers_2D \
		= umap_cluster_waveforms(np.array(norm_width_env_gape_storage))
	#Normalized width/height clustering
	umap_env_norm_full_n_clusters,umap_env_norm_full_clust_centers,umap_env_norm_full_labels,\
	umap_env_norm_full_2D,umap_env_norm_full_clust_centers_2D \
		= umap_cluster_waveforms(np.array(norm_full_env_gape_storage))
		
	#_____Cluster plot grouping_____
	gape_tastes_labels = [taste_names[gp] for gp in gape_tastes]
	gape_start_times_min = np.min(gape_start_times)
	gape_start_times_max = np.max(gape_start_times)
	gape_start_times_group_options = np.arange(np.floor(gape_start_times_min/100).astype('int')*100,np.ceil(gape_start_times_max/100).astype('int')*100,100) #Step from min to max in 100 ms bins
	gape_start_times_group_labels = gape_start_times_group_options/100
	gape_start_times_group = np.array([np.argmin(np.abs(gst-gape_start_times_group_options)) for gst in gape_start_times])
	gape_start_times_labels = [gape_start_times_group_labels[gstg] for gstg in gape_start_times_group]
		
	#Pass the clusters from UMAP to regular cluster plotting
	run_n_clustering(umap_env_norm_width_n_clusters,norm_width_env_gape_storage,
						 umap_env_norm_full_n_clusters,norm_full_env_gape_storage,
						 gape_tastes,gape_tastes_labels,gape_start_times_group,
						 gape_start_times_labels,clust_save_dir,nf,emg_data_dict,
						 dict_save_dir)
	
	#Reload emg_data_dict
	with gzip.open(dict_save_dir, 'rb') as f:
		emg_data_dict = pickle.load(f)
	
	#Calculate cluster statistics
	cluster_stats(nf,emg_data_dict,clust_save_dir,dict_save_dir)
		
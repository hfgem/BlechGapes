#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:16:31 2024

@author: hannahgermaine
Functions used by mouth_movement_analyses.py
"""

from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, umap, pickle, gzip
from scipy.fft import fft

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

def fourier_emg(X):
	
	num_freq = np.shape(X)[1]
	X_fft = fft(X,axis=1)
	X_fft_real = np.real(X_fft)
	X_fourier = X_fft_real[:,:np.ceil(num_freq/2).astype('int')]
	
	return X_fourier

def cluster_waveforms(X,nc):

	km = KMeans(n_clusters = nc)
	km.fit(X)
	labels=km.labels_
	cluster_centers=km.cluster_centers_
	labels_unique=np.unique(labels)
	n_clusters = len(labels_unique)
	
	#Re-dimensionalize data
	p = PCA(n_components=2)
	p.fit(X)
	X_redim = p.transform(X)
	center_redim = p.transform(cluster_centers)
	
	return n_clusters,cluster_centers,labels,X_redim,center_redim

def umap_cluster_waveforms(X):
# =============================================================================
#  	"Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py"
#  	ms = MeanShift()
#  	ms.fit(X)
#  	labels = ms.labels_
#  	cluster_centers = ms.cluster_centers_
#  	labels_unique = np.unique(labels)
#  	n_clusters = len(labels_unique)
# =============================================================================

	reducer = umap.UMAP()
	embedding = reducer.fit_transform(X)
	
	#Cluster embeddings using meanshift
	ms = MeanShift()
	ms.fit(embedding)
	labels = ms.labels_
	labels_unique = np.unique(labels)
	n_clusters = len(labels_unique)
	center_redim = ms.cluster_centers_
	cluster_centers = reducer.inverse_transform(center_redim)

	#Re-dimensionalize data
	p = PCA(n_components=2)
	p.fit(X)
	center_redim = p.transform(cluster_centers)
	
	return n_clusters,cluster_centers,labels,embedding,center_redim

def plot_cluster_results(save_dir,save_name,n_clusters,cluster_centers,labels,X_redim,center_redim,groups,group_labels=[]):
	cluster_colors = cm.viridis(np.linspace(0,1,n_clusters))
	cluster_wav_colors = np.array([cluster_colors[nc,:] for nc in labels])
	
	unique_groups = np.unique(groups)
	unique_group_labels = [group_labels[np.where(groups==ug)[0][0]] for ug in np.unique(groups)]
	group_colors = cm.jet(np.linspace(0,1,len(unique_groups)))
	group_scatter_colors = np.array([group_colors[np.where(unique_groups==g)[0],:] for g in groups])
	
	f,ax = plt.subplots(nrows=3,ncols=2,figsize=(8,8))
	gs = ax[1, 0].get_gridspec()
	#Cluster Centers
	ax[0,0].scatter(center_redim[:,0],center_redim[:,1],c=cluster_colors,edgecolor='r')
	ax[0,0].set_title('Average cluster center PCA 2D')
	#Average Cluster Waveform
	for c_i in range(n_clusters):
		ax[0,1].plot(cluster_centers[c_i,:],alpha=0.5,color=cluster_colors[c_i,:])
	ax[0,1].set_title('Average cluster form for ' + str(n_clusters) + ' clusters')
	#Indiv waveforms 2D
	for ax_i in ax[1,:]:
		ax_i.remove() #remove the underlying axes
	axbig = f.add_subplot(gs[1,:])
	axbig.scatter(X_redim[:,0],X_redim[:,1],c=cluster_wav_colors,alpha=0.4)
	#axbig.scatter(center_redim[:,0],center_redim[:,1],c=cluster_colors,alpha=1,edgecolor='r')
	if n_clusters < 10:
		cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),ax=axbig,ticks=np.linspace(0,1,n_clusters))
		cbar.ax.set_yticklabels(np.arange(n_clusters))
	else:
		group_subsample = np.floor(np.linspace(0,n_clusters-1,5)).astype('int')
		cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'),ax=axbig,ticks=np.linspace(0,1,5))
		cbar.ax.set_yticklabels(np.linspace(0,n_clusters-1,5).astype('int'))
	axbig.set_title('cluster colors')
	#Indiv waveforms 2D with group coloring
	for ax_i in ax[2,:]:
		ax_i.remove() #remove the underlying axes
	axbig2 = f.add_subplot(gs[2,:])
	axbig2.scatter(X_redim[:,0],X_redim[:,1],c=group_scatter_colors,alpha=0.4)
	if len(unique_groups) <= 5:
		cbar = plt.colorbar(cm.ScalarMappable(cmap='jet'),ax=axbig2,ticks=np.linspace(0,1,len(unique_groups)))
		cbar.ax.set_yticklabels(unique_group_labels)
	else:
		group_subsample = np.floor(np.linspace(0,len(unique_groups)-1,5)).astype('int')
		cbar = plt.colorbar(cm.ScalarMappable(cmap='jet'),ax=axbig2,ticks=np.linspace(0,1,5))
		cbar.ax.set_yticklabels(np.array(unique_group_labels)[group_subsample])
	save_name_split = (' ').join(save_name.split('_'))
	axbig2.set_title(save_name_split + ' colors')
	f.suptitle('Clustering Results')
	f.tight_layout()
	f.savefig(os.path.join(save_dir,save_name+'.png'))
	f.savefig(os.path.join(save_dir,save_name+'.png'))
	plt.close(f)
	
def run_n_clustering(norm_width_nc,norm_width_env_gape_storage,
					 norm_full_nc,norm_full_env_gape_storage,
					 gape_tastes,gape_tastes_labels,gape_start_times_group,
					 gape_start_times_labels,clust_save_dir,nf,emg_data_dict,
					 dict_save_dir):
	#_____Cluster gape waveform information_____
	#Normalized width clustering
	env_norm_width_n_clusters,env_norm_width_clust_centers,env_norm_width_labels,env_norm_width_2D,env_norm_width_clust_centers_2D \
		= cluster_waveforms(np.array(norm_width_env_gape_storage),norm_width_nc)
	norm_width_dict = dict()
	norm_width_dict['n_clusters'] = env_norm_width_n_clusters
	norm_width_dict['clust_centers'] = env_norm_width_clust_centers
	norm_width_dict['labels'] = env_norm_width_labels
	norm_width_dict['data_redim'] = env_norm_width_2D
	norm_width_dict['clust_centers_redim'] = env_norm_width_clust_centers_2D
	
	#Normalized width/height clustering
	env_norm_full_n_clusters,env_norm_full_clust_centers,env_norm_full_labels,env_norm_full_2D,env_norm_full_clust_centers_2D \
		= cluster_waveforms(np.array(norm_full_env_gape_storage),norm_full_nc)
	norm_full_dict = dict()
	norm_full_dict['n_clusters'] = env_norm_full_n_clusters
	norm_full_dict['clust_centers'] = env_norm_full_clust_centers
	norm_full_dict['labels'] = env_norm_full_labels
	norm_full_dict['data_redim'] = env_norm_full_2D
	norm_full_dict['clust_centers_redim'] = env_norm_full_clust_centers_2D
		
	#Normalized width
	env_wid_save_dir = os.path.join(clust_save_dir,'Envelope_norm_width')
	if not os.path.isdir(env_wid_save_dir):
		os.mkdir(env_wid_save_dir)
	save_name = 'gape_tastes'
	plot_cluster_results(env_wid_save_dir,save_name,env_norm_width_n_clusters,\
					  env_norm_width_clust_centers,env_norm_width_labels,\
						  env_norm_width_2D,env_norm_width_clust_centers_2D,\
							  gape_tastes,gape_tastes_labels)
	
	save_name = 'gape_start_times'
	plot_cluster_results(env_wid_save_dir,save_name,env_norm_width_n_clusters,\
					  env_norm_width_clust_centers,env_norm_width_labels,\
						  env_norm_width_2D,env_norm_width_clust_centers_2D,\
							  gape_start_times_group,gape_start_times_labels)
		
	#Normalized full
	env_full_save_dir = os.path.join(clust_save_dir,'Envelope_norm_full')
	if not os.path.isdir(env_full_save_dir):
		os.mkdir(env_full_save_dir)
	save_name = 'gape_tastes'
	plot_cluster_results(env_full_save_dir,save_name,env_norm_full_n_clusters,\
					  env_norm_full_clust_centers,env_norm_full_labels,\
						  env_norm_full_2D,env_norm_full_clust_centers_2D,\
							  gape_tastes,gape_tastes_labels)
	
	save_name = 'gape_start_times'
	plot_cluster_results(env_full_save_dir,save_name,env_norm_full_n_clusters,\
					  env_norm_full_clust_centers,env_norm_full_labels,\
						  env_norm_full_2D,env_norm_full_clust_centers_2D,\
							  gape_start_times_group,gape_start_times_labels)
		
	emg_data_dict[nf]['env_norm_width_n_clusters'] = env_norm_width_n_clusters
	emg_data_dict[nf]['env_norm_full_n_clusters'] = env_norm_full_n_clusters
	emg_data_dict[nf]['norm_width_dict'] = norm_width_dict
	emg_data_dict[nf]['norm_full_dict'] = norm_full_dict
	
	#Save updated dictionary
	with gzip.open(dict_save_dir, 'wb') as f:
		pickle.dump(emg_data_dict, f)
	
def cluster_stats(nf,emg_data_dict,clust_save_dir,dict_save_dir):
	
	#____Grab relevant data____
	env_data = emg_data_dict[nf]['env']
	gape_start_times = emg_data_dict[nf]['gape_start_times']
	num_gapes = len(gape_start_times)
	gape_tastes = emg_data_dict[nf]['gape_tastes']
	nc_width = emg_data_dict[nf]['env_norm_width_n_clusters']
	nc_full = emg_data_dict[nf]['env_norm_full_n_clusters']
	norm_width_labels = emg_data_dict[nf]['norm_width_dict']['labels']
	norm_full_labels = emg_data_dict[nf]['norm_full_dict']['labels']
	taste_names = emg_data_dict[nf]['taste_names']
	norm_full_env_gape_storage = emg_data_dict[nf]['norm_full_env_gape_storage']
	norm_width_env_gape_storage = emg_data_dict[nf]['norm_width_env_gape_storage']
	
	#____Plot Settings____
	cluster_colors_norm_width = cm.viridis(np.linspace(0,1,nc_width))
	cluster_colors_norm_full = cm.viridis(np.linspace(0,1,nc_full))
	
	#____Normalized Width Analyses____
	env_wid_save_dir = os.path.join(clust_save_dir,'Envelope_norm_width')
	norm_width_clust_gapes = dict()
	#For each cluster calculate the taste representation ratios
	for nc_i in range(nc_width):
		norm_width_clust_gapes[nc_i] = dict()
		
		cli = np.where(norm_width_labels == nc_i)[0]
		clust_gapes = np.array(norm_width_env_gape_storage)[cli]
		clust_start_times = np.array(gape_start_times)[cli]
		clust_tastes = np.array(gape_tastes)[cli]
		clust_taste_labels = [taste_names[cti] for cti in clust_tastes]
		
		#Calculate stats
		clust_taste_ratios = np.array([len(np.where(clust_tastes==t_i)[0]) for t_i in range(len(taste_names))])/num_gapes
		mean_clust_start_time = np.mean(clust_start_times)
		std_clust_start_time = np.std(clust_start_times)
		
		#Save data
		norm_width_clust_gapes[nc_i]['clust_gapes'] = clust_gapes
		norm_width_clust_gapes[nc_i]['clust_start_times'] = clust_start_times
		norm_width_clust_gapes[nc_i]['clust_tastes'] = clust_tastes
		norm_width_clust_gapes[nc_i]['clust_taste_ratios'] = clust_taste_ratios
		norm_width_clust_gapes[nc_i]['mean_clust_start_time'] = mean_clust_start_time
		norm_width_clust_gapes[nc_i]['std_clust_start_time'] = std_clust_start_time
		
		#___Plot results___
		clust_plot_name = 'norm_width_cluster_' + str(nc_i) + '_stats'
		f,ax = plt.subplots(nrows=3,ncols=1,figsize=(3,9))
		#Plot cluster waveforms overlaid
		ax[0].plot(clust_gapes.T,alpha=0.2,color='b')
		ax[0].set_title('Waveforms Overlaid')
		#Plot taste representation ratios
		ax[1].pie(clust_taste_ratios,labels=taste_names)
		ax[1].set_title('Taste Ratios')
		#Plot histogram of onset times w/ mean and std overlaid
		ax[2].hist(clust_start_times)
		ax[2].axvline(mean_clust_start_time,color='blue',label='Mean = '+str(round(mean_clust_start_time,2)))
		ax[2].axvline(mean_clust_start_time+std_clust_start_time,linestyle='dashed',color='blue',label='Std = ' + str(round(std_clust_start_time,2)))
		ax[2].axvline(mean_clust_start_time-std_clust_start_time,linestyle='dashed',color='blue',label='_')
		ax[2].set_xlabel('ms from taste delivery')
		ax[2].set_title('Movement Onset Time')
		ax[2].legend()
		#Figure tweaks
		plt.suptitle('Normalized Width Cluster ' + str(nc_i))
		plt.tight_layout()
		#Save plot
		f.savefig(os.path.join(env_wid_save_dir,clust_plot_name + '.png'))
		f.savefig(os.path.join(env_wid_save_dir,clust_plot_name + '.svg'))
		plt.close(f)
	
	#Joint Cluster Plot
	f,ax = plt.subplots(nrows=3,ncols=1,figsize=(3,9))
	#Plot cluster waveforms overlaid
	for nc_i in range(nc_width):
		ax[0].plot(norm_width_clust_gapes[nc_i]['clust_gapes'].T,alpha=0.2,color=cluster_colors_norm_width[nc_i,:])
	ax[0].set_title('Waveforms Overlaid')
	#Plot taste representation ratios
	for nc_i in range(nc_width):
		ax[1].plot(np.arange(len(taste_names)),norm_width_clust_gapes[nc_i]['clust_taste_ratios'],label=str(nc_i),color=cluster_colors_norm_width[nc_i,:])
	ax[1].set_xticks(np.arange(len(taste_names)))
	ax[1].set_xticklabels(taste_names)
	ax[1].legend()
	ax[1].set_ylabel('Fraction of Gapes')
	ax[1].set_title('Taste Ratios')
	#Plot histogram of onset times w/ mean and std overlaid
	for nc_i in range(nc_width):
		ax[2].hist(norm_width_clust_gapes[nc_i]['clust_start_times'],density='True',alpha=0.5,color=cluster_colors_norm_width[nc_i,:],label=str(nc_i))
	ax[2].legend()
	ax[2].set_xlabel('ms from taste delivery')
	ax[2].set_ylabel('Density of Distribution')
	ax[2].set_title('Movement Onset Time')
	#Figure tweaks
	plt.suptitle('Normalized Width Clusters')
	plt.tight_layout()
	#Save plot
	f.savefig(os.path.join(env_wid_save_dir,'norm_width_all_cluster_stats.png'))
	f.savefig(os.path.join(env_wid_save_dir,'norm_width_all_cluster_stats.svg'))
	plt.close(f)
	
	
	#____Normalized Width/Height Analyses____
	env_full_save_dir = os.path.join(clust_save_dir,'Envelope_norm_full')
	norm_full_clust_gapes = dict()
	#For each cluster calculate the taste representation ratios
	for nc_i in range(nc_full):
		norm_full_clust_gapes[nc_i] = dict()
		cli = np.where(norm_full_labels == nc_i)[0]
		clust_gapes = np.array(norm_full_env_gape_storage)[cli]
		clust_start_times = np.array(gape_start_times)[cli]
		clust_tastes = np.array(gape_tastes)[cli]
		
		#Calculate stats
		clust_taste_ratios = np.array([len(np.where(clust_tastes==t_i)[0]) for t_i in range(len(taste_names))])/num_gapes
		mean_clust_start_time = np.mean(clust_start_times)
		std_clust_start_time = np.std(clust_start_times)
		
		#Save data
		norm_full_clust_gapes[nc_i]['clust_gapes'] = clust_gapes
		norm_full_clust_gapes[nc_i]['clust_start_times'] = clust_start_times
		norm_full_clust_gapes[nc_i]['clust_tastes'] = clust_tastes
		norm_full_clust_gapes[nc_i]['clust_taste_ratios'] = clust_taste_ratios
		norm_full_clust_gapes[nc_i]['mean_clust_start_time'] = mean_clust_start_time
		norm_full_clust_gapes[nc_i]['std_clust_start_time'] = std_clust_start_time
		
		#___Plot results___
		clust_plot_name = 'norm_full_cluster_' + str(nc_i) + '_stats'
		f,ax = plt.subplots(nrows=3,ncols=1,figsize=(3,9))
		#Plot cluster waveforms overlaid
		ax[0].plot(clust_gapes.T,alpha=0.2,color='b')
		ax[0].set_title('Waveforms Overlaid')
		#Plot taste representation ratios
		ax[1].pie(clust_taste_ratios,labels=taste_names)
		ax[1].set_title('Taste Ratios')
		#Plot histogram of onset times w/ mean and std overlaid
		ax[2].hist(clust_start_times)
		ax[2].axvline(mean_clust_start_time,color='blue',label='Mean = '+str(round(mean_clust_start_time,2)))
		ax[2].axvline(mean_clust_start_time+std_clust_start_time,linestyle='dashed',color='blue',label='Std = ' + str(round(std_clust_start_time,2)))
		ax[2].axvline(mean_clust_start_time-std_clust_start_time,linestyle='dashed',color='blue',label='_')
		ax[2].set_xlabel('ms from taste delivery')
		ax[2].set_title('Movement Onset Time')
		#Figure tweaks
		plt.suptitle('Normalized Full Cluster ' + str(nc_i))
		plt.tight_layout()
		#Save plot
		f.savefig(os.path.join(env_full_save_dir,clust_plot_name + '.png'))
		f.savefig(os.path.join(env_full_save_dir,clust_plot_name + '.svg'))
		plt.close(f)
	
	#Joint Cluster Plot
	f,ax = plt.subplots(nrows=3,ncols=1,figsize=(3,9))
	#Plot cluster waveforms overlaid
	for nc_i in range(nc_full):
		ax[0].plot(norm_full_clust_gapes[nc_i]['clust_gapes'].T,alpha=0.2,color=cluster_colors_norm_full[nc_i,:])
	ax[0].set_title('Waveforms Overlaid')
	#Plot taste representation ratios
	for nc_i in range(nc_full):
		ax[1].plot(np.arange(len(taste_names)),norm_full_clust_gapes[nc_i]['clust_taste_ratios'],label=str(nc_i),color=cluster_colors_norm_full[nc_i,:])
	ax[1].set_xticks(np.arange(len(taste_names)))
	ax[1].set_xticklabels(taste_names)
	ax[1].legend()
	ax[1].set_ylabel('Fraction of Gapes')
	ax[1].set_title('Taste Ratios')
	#Plot histogram of onset times w/ mean and std overlaid
	for nc_i in range(nc_full):
		ax[2].hist(norm_full_clust_gapes[nc_i]['clust_start_times'],density='True',alpha=0.5,color=cluster_colors_norm_full[nc_i,:],label=str(nc_i))
	ax[2].legend()
	ax[2].set_xlabel('ms from taste delivery')
	ax[2].set_ylabel('Density of Distribution')
	ax[2].set_title('Movement Onset Time')
	#Figure tweaks
	plt.suptitle('Normalized Width Clusters')
	plt.tight_layout()
	#Save plot
	f.savefig(os.path.join(env_full_save_dir,'norm_full_all_cluster_stats.png'))
	f.savefig(os.path.join(env_full_save_dir,'norm_full_all_cluster_stats.svg'))
	plt.close(f)
	
	#____Save____
	#Save results to dict
	emg_data_dict[nf]['norm_width_clust_gapes'] = norm_width_clust_gapes
	emg_data_dict[nf]['norm_full_clust_gapes'] = norm_full_clust_gapes
	#Save updated dictionary
	f = open(dict_save_dir,"wb")
	pickle.dump(emg_data_dict,f)	
	
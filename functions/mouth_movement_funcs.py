#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:16:31 2024

@author: hannahgermaine
Functions used by mouth_movement_analyses.py
"""

from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

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

def cluster_waveforms(X,nc):
# =============================================================================
#  	"Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py"
#  	ms = MeanShift()
#  	ms.fit(X)
#  	labels = ms.labels_
#  	cluster_centers = ms.cluster_centers_
#  	labels_unique = np.unique(labels)
#  	n_clusters = len(labels_unique)
# =============================================================================

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
	axbig.scatter(center_redim[:,0],center_redim[:,1],c=cluster_colors,alpha=1,edgecolor='r')
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
# Import stuff!
import sys
import os

current_path = os.path.realpath(__file__)
os.chdir(current_path)

import numpy as np
import tables
import easygui
import itertools
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from blech_held_units_funcs import *
    
# Ask the user for the number of days to be compared
num_days = int_input("How many days-worth of data are you comparing for held units (integer)? ")

# Ask the user for the percentile criterion to use to determine held units
percent_criterion = easygui.multenterbox(msg = 'What percentile of the intra unit J3 distribution do you want to use to pull out held units?', fields = ['Percentile criterion (1-100) - lower is more conservative (e.g., 95)'])
percent_criterion = float(percent_criterion[0])

# Ask the user for the percentile criterion to use to determine held units
wf_type = easygui.multenterbox('Which types of waveforms to be used for held_unit analysis',
                               'Type either "raw_CAR_waveform" or "norm_waveform"',
                               ['waveform type'],
                               ['norm_waveform'])[0]

data_dict = dict() #Store all the different days' data in a dictionary
all_neur_inds = [] #Store all neuron indices to calculate cross-day combinations
all_intra_J3 = [] #Store all intra J3 data to calculate cutoff for inter-J3
for n_i in range(num_days):
    data_dict[n_i] = dict()
    #Ask for directory of the dataset hdf5 file
    dir_name = easygui.diropenbox(msg = 'Where is the hdf5 file from the ' + str(n_i + 1) + ' day?', title = str(n_i + 1) + ' day of data')
    data_dict[n_i]['dir_name'] = dir_name
    #Find hdf5 in directory
    file_list = os.listdir(dir_name)
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
    data_dict[n_i]['hdf5_name'] = hdf5_name
    #Open hdf5 file
    hf5 = tables.open_file(os.path.join(dir_name,hdf5_name), 'r')
    num_neur = len(hf5.root.unit_descriptor[:])
    all_neur_inds.append(list(np.arange(num_neur)))
    data_dict[n_i]['num_neur'] = num_neur
    #Calculate the Intra-J3 data for the units
    intra_J3 = []
    for unit in range(num_neur):
        # Only go ahead if this is a single unit
        if hf5.root.unit_descriptor[unit]['single_unit'] == 1:
            exec("wf_day1 = hf5.root.sorted_units.unit%03d.waveforms[:]" % (unit))
            if wf_type == 'norm_waveform':
                wf_day1 = wf_day1 / np.std(wf_day1)
            pca = PCA(n_components = 4)
            pca.fit(wf_day1)
            pca_wf_day1 = pca.transform(wf_day1)
            intra_J3.append(calculate_J3(pca_wf_day1[:int(wf_day1.shape[0]*(1.0/3.0)), :], pca_wf_day1[int(wf_day1.shape[0]*(2.0/3.0)):, :]))
    data_dict[n_i]['intra_J3'] = intra_J3
    all_intra_J3.extend(intra_J3)
    #Pull unit info for all units
    all_unit_info = []
    for unit in range(num_neur):
        all_unit_info.append(get_unit_info(hf5.root.unit_descriptor[unit]))
    data_dict[n_i]['all_unit_info'] = all_unit_info
    #Pull unit waveforms for all units
    all_unit_waveforms = []
    for unit in range(num_neur):
        exec("wf = hf5.root.sorted_units.unit%03d.waveforms[:]" % (unit))
        all_unit_waveforms.append(wf)
    data_dict[n_i]['all_unit_waveforms'] = all_unit_waveforms
    #Close hdf5 file
    hf5.close()
    
# Ask the user for the output directory to save the held units and plots in
save_dir = easygui.diropenbox(msg = 'Where do you want to save the held units and plots?', title = 'Output directory')

#Save the data dictionary just in case want in future
np.save(os.path.join(save_dir,'data_dict.npy'),data_dict,allow_pickle=True)

#Calculate the intra-J3 percentile cutoff
all_intra_J3_cutoff = np.percentile(all_intra_J3, percent_criterion)

held_unit_storage = [] #placeholder storage for held units across days

#Calculate all pairwise unit tests
all_neur_combos = list(itertools.product(*all_neur_inds))
all_day_combos = list(itertools.combinations(np.arange(num_days),2))

all_inter_J3 = []
for nc in all_neur_combos:
    unit_info_same = 1
    for dc in all_day_combos:
        if data_dict[dc[0]]['all_unit_info'][nc[dc[0]]] != \
            data_dict[dc[1]]['all_unit_info'][nc[dc[1]]]:
                unit_info_same = 0
    
    if unit_info_same == 1:
        #Collect waveforms to be compared
        waveforms = [] #list of numpy arrays
        all_waveforms = []
        for day in range(len(nc)):
            wf = data_dict[day]['all_unit_waveforms'][nc[day]]
            if wf_type == 'norm_waveform':
                waveforms.append(wf/np.nanstd(wf))
                all_waveforms.extend(wf/np.nanstd(wf))
            else:
                waveforms.append(wf)
                all_waveforms.extend(wf)
                
        #Fit PCA to waveforms
        pca = PCA(n_components = 4)
        pca.fit(np.array(all_waveforms))
        day_pca = []
        for day in range(len(nc)):
            day_pca.append(pca.transform(np.array(waveforms[day])))
                
        #Calculate the inter_J3 across days
        all_days_inter_J3 = []
        for dc in all_day_combos:
            all_days_inter_J3.extend([calculate_J3(day_pca[dc[0]], day_pca[dc[1]])])
        all_inter_J3.append(all_days_inter_J3) 
        
        #Do all inter_J3 match the cutoff?
        if np.sum((np.array(all_intra_J3_cutoff) <= all_intra_J3_cutoff).astype('int')) = num_days:
            

#%% OLD CODE

# Open the hdf5 file
hf51 = tables.open_file(hdf5_name, 'r')

# Now do the same for the second day of data
dir_name = easygui.diropenbox(msg = 'Where is the hdf5 file from the second day?', title = 'Second day of data')
os.chdir(dir_name)
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
hf52 = tables.open_file(hdf5_name, 'r')

# Ask the user for the output directory to save the held units and plots in
dir_name = easygui.diropenbox(msg = 'Where do you want to save the held units and plots?', title = 'Output directory')
os.chdir(dir_name)

# Ask the user for the percentile criterion to use to determine held units
percent_criterion = easygui.multenterbox(msg = 'What percentile of the intra unit J3 distribution do you want to use to pull out held units?', fields = ['Percentile criterion (1-100) - lower is more conservative (e.g., 95)'])
percent_criterion = float(percent_criterion[0])

# Ask the user for the percentile criterion to use to determine held units
wf_type = easygui.multenterbox('Which types of waveforms to be used for held_unit analysis',
                               'Type either "raw_CAR_waveform" or "norm_waveform"',
                               ['waveform type'],
                               ['norm_waveform'])[0]

# Make a file to save the numbers of the units that are deemed to have been held across days
f = open(f'held_units_{wf_type}_{percent_criterion}.txt', 'w')
print('Day1', '\t', 'Day2', file=f)

# Calculate the intra-unit J3 numbers by taking every unit, and calculating the J3 between the first 3rd and last 3rd of its spikes
intra_J3 = []
# Run through the units on day 1
for unit1 in range(len(hf51.root.unit_descriptor[:])):
    # Only go ahead if this is a single unit
    if hf51.root.unit_descriptor[unit1]['single_unit'] == 1:
        exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1))
        if wf_type == 'norm_waveform':
            wf_day1 = wf_day1 / np.std(wf_day1)
        pca = PCA(n_components = 4)
        pca.fit(wf_day1)
        pca_wf_day1 = pca.transform(wf_day1)
        intra_J3.append(calculate_J3(pca_wf_day1[:int(wf_day1.shape[0]*(1.0/3.0)), :], pca_wf_day1[int(wf_day1.shape[0]*(2.0/3.0)):, :]))
# Run through the units on day 2
for unit2 in range(len(hf52.root.unit_descriptor[:])):
    # Only go ahead if this is a single unit
    if hf52.root.unit_descriptor[unit2]['single_unit'] == 1:
        exec("wf_day2 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit2))
        if wf_type == 'norm_waveform':
            wf_day2 = wf_day2 / np.std(wf_day2)
        pca = PCA(n_components = 4)
        pca.fit(wf_day2)
        pca_wf_day2 = pca.transform(wf_day2)
        intra_J3.append(calculate_J3(pca_wf_day2[:int(wf_day2.shape[0]*(1.0/3.0)), :], pca_wf_day2[int(wf_day2.shape[0]*(2.0/3.0)):, :]))

# Now calculate the inter unit J3 numbers for units of the same type on the same electrode - mark them as held if they're less than the 95th percentile of intra_J3
# Run through the units on day 1
# unit features critical for held_unit_analysis
#unit_info_labels = ['electrode_number', 'fast_spiking', 'regular_spiking', 'single_unit']
     
inter_J3 = []
for unit1 in range(len(hf51.root.unit_descriptor[:])):
    # Only go ahead if this is a single unit
    if hf51.root.unit_descriptor[unit1]['single_unit'] == 1:
        # Run through the units on day 2 and check if it was present (same electrode and unit type)
        for unit2 in range(len(hf52.root.unit_descriptor[:])):
            print(unit1, unit2, len(hf51.root.unit_descriptor[:]), len(hf52.root.unit_descriptor[:]))
            if get_unit_info(hf52.root.unit_descriptor[unit2]) == \
               get_unit_info(hf51.root.unit_descriptor[unit1]):
            # if hf52.root.unit_descriptor[unit2] == hf51.root.unit_descriptor[unit1]:
                # Load up the waveforms for unit1 and unit2
                exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1))
                exec("wf_day2 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit2))
                if wf_type == 'norm_waveform':
                    wf_day1 = wf_day1 / np.std(wf_day1)
                    wf_day2 = wf_day2 / np.std(wf_day2)
                #energy1 = np.sqrt(np.sum(wf_day1**2, axis = 1))/wf_day1.shape[1]
                #energy2 = np.sqrt(np.sum(wf_day2**2, axis = 1))/wf_day2.shape[1]

                #pca_wf_day1 = np.divide(wf_day1.T, energy1).T
                #pca_wf_day2 = np.divide(wf_day2.T, energy2).T

                # Run the PCA - pick the first 3 principal components
                pca = PCA(n_components = 4)
                pca.fit(np.concatenate((wf_day1, wf_day2), axis = 0))
                pca_wf_day1 = pca.transform(wf_day1)
                pca_wf_day2 = pca.transform(wf_day2)
                # Get inter-day J3
                inter_J3.append(calculate_J3(pca_wf_day1, pca_wf_day2))

                # Only say that this unit is held if inter_J3 <= 95th percentile of intra_J3
                #print inter_J3, np.percentile(intra_J3, 95.0)
                #wait = raw_input()
                if inter_J3[-1] <= np.percentile(intra_J3, percent_criterion):
                    print(unit1, '\t', unit2, file=f)
                    # Also plot both these units on the same graph
                    exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1))
                    exec("wf_day2 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit2))
                    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
                    t = np.arange(wf_day1.shape[1]/10)
                    mean_wfs = np.mean(wf_day1[:, ::10], axis = 0)
                    max_, min_ = np.max(mean_wfs), np.min(mean_wfs)
                    #plt.plot(t - 15, wf_day1[:, ::10].T, linewidth = 0.01, color = 'red')
                    ax[0].plot(t - 15, np.mean(wf_day1[:, ::10], axis = 0), linewidth = 5.0, color = 'black')
                    ax[0].plot(t - 15, np.mean(wf_day1[:, ::10], axis = 0) - np.std(wf_day1[:, ::10], axis = 0), 
                               linewidth = 2.0, color = 'black', alpha = 0.5)
                    ax[0].plot(t - 15, np.mean(wf_day1[:, ::10], axis = 0) + np.std(wf_day1[:, ::10], axis = 0), 
                               linewidth = 2.0, color = 'black', alpha = 0.5)
                    ax[0].axhline(max_, color='r', ls='--')
                    ax[0].axhline(min_, color='r', ls='--')
                    ax[0].set_xlabel('Time (samples (30 per ms))', fontsize = 12)
                    ax[0].set_ylabel('Voltage (microvolts)', fontsize = 12)
                    ax[0].set_ylim([np.min(np.concatenate((wf_day1, wf_day2), axis = 0)) - 20, np.max(np.concatenate((wf_day1, wf_day2), axis = 0)) + 20])
                    ax[0].set_title('Unit %i, total waveforms = %i' % (unit1, wf_day1.shape[0]) + '\n' + 'Electrode: %i, J3: %f' % (hf51.root.unit_descriptor[unit1]['electrode_number'], inter_J3[-1]) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf51.root.unit_descriptor[unit1]['single_unit'], hf51.root.unit_descriptor[unit1]['regular_spiking'], hf51.root.unit_descriptor[unit1]['fast_spiking']), fontsize = 12)
                    #plt.tick_params(axis='both', which='major', labelsize=32)

                    #plt.subplot(122)
                    #t = np.arange(wf_day2.shape[1]/10)
                    #plt.plot(t - 15, wf_day2[:, ::10].T, linewidth = 0.01, color = 'red')
                    ax[1].plot(t - 15, np.mean(wf_day2[:, ::10], axis = 0), linewidth = 5.0, color = 'black')
                    ax[1].plot(t - 15, np.mean(wf_day2[:, ::10], axis = 0) - np.std(wf_day2[:, ::10], axis = 0), 
                               linewidth = 2.0, color = 'black', alpha = 0.5)
                    ax[1].plot(t - 15, np.mean(wf_day2[:, ::10], axis = 0) + np.std(wf_day2[:, ::10], axis = 0), 
                               linewidth = 2.0, color = 'black', alpha = 0.5)
                    ax[1].axhline(max_, color='r', ls='--')
                    ax[1].axhline(min_, color='r', ls='--')
                    ax[1].set_xlabel('Time (samples (30 per ms))', fontsize = 12)
                    #ax[1].set_ylabel('Voltage (microvolts)', fontsize = 35)
                    ax[1].set_ylim([np.min(np.concatenate((wf_day1, wf_day2), axis = 0)) - 20, np.max(np.concatenate((wf_day1, wf_day2), axis = 0)) + 20])
                    ax[1].set_title('Unit %i, total waveforms = %i' % (unit2, wf_day2.shape[0]) + '\n' + 'Electrode: %i' % (hf52.root.unit_descriptor[unit2]['electrode_number']) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf52.root.unit_descriptor[unit2]['single_unit'], hf52.root.unit_descriptor[unit2]['regular_spiking'], hf52.root.unit_descriptor[unit2]['fast_spiking']), fontsize = 12)
                    #plt.tick_params(axis='both', which='major', labelsize=32)
                    plt.tight_layout()
                    fig.savefig(f'Unit{unit1}_and_Unit{unit2}_{wf_type}.png', bbox_inches = 'tight')
                    plt.close('all')

# Plot the intra and inter J3 in a different file
fig = plt.figure()
plt.hist(inter_J3, bins = 20, alpha = 0.3, label = 'Across-session J3')
plt.hist(intra_J3, bins = 20, alpha = 0.3, label = 'Within-session J3')
# Draw a vertical line at the percentile criterion used to choose held units
plt.axvline(np.percentile(intra_J3, percent_criterion), linewidth = 5.0, color = 'black', linestyle = 'dashed')
plt.xlabel('J3', fontsize = 12)
plt.ylabel('Number of single unit pairs', fontsize = 12)
#plt.tick_params(axis='both', which='major', labelsize=32)
fig.savefig(f'J3_distributions_{wf_type}.png', bbox_inches = 'tight')
plt.close('all')

# Close the hdf5 files and the file with the held units
hf51.close()
hf52.close()
f.close()


'''
intra_J3 = []                    
for unit1 in range(len(hf51.root.unit_descriptor[:])):
    if hf51.root.unit_descriptor[unit1]['single_unit'] == 1:
        exec("wf_day1 = hf51.root.sorted_units.unit%03d.waveforms[:]" % (unit1 + 1))
        
        
        #for run in range(num_random_runs):
        # Do it for day 1
        x = np.arange(wf_day1.shape[0])
        #np.random.shuffle(x)
        #print wf_day1[x[:wf_day1.shape[0]/2], :].shape, len(x)
        intra_J3.append(calculate_J3(wf_day1[x[:int(wf_day1.shape[0]*0.5)], :], wf_day1[x[int(wf_day1.shape[0]*0.5):], :]))

intra_J3 = []                    
for unit1 in range(len(hf52.root.unit_descriptor[:])):
    J3 = []
    if hf52.root.unit_descriptor[unit1]['single_unit'] == 1:
        exec("wf_day1 = hf52.root.sorted_units.unit%03d.waveforms[:]" % (unit1 + 1))
        
        
        for run in range(num_random_runs):
        # Do it for day 1
            x = np.arange(wf_day1.shape[0])
            np.random.shuffle(x)
        #print wf_day1[x[:wf_day1.shape[0]/2], :].shape, len(x)
            J3.append(calculate_J3(wf_day1[x[:int(wf_day1.shape[0]*0.33)], :], wf_day1[x[int(wf_day1.shape[0]*0.67):], :]))    

        x = np.arange(wf_day1.shape[0])
        J3.append(calculate_J3(wf_day1[x[:int(wf_day1.shape[0]*0.33)], :], wf_day1[x[int(wf_day1.shape[0]*0.67):], :]))

        intra_J3.append(J3)            
'''
'''
                # Divide the waveforms for unit1 and unit2 in equal splits - do this randomly num_random_runs times
                intra_J3 = []
                for run in range(num_random_runs):
                    # Do it for day 1
                    x = np.arange(wf_day1.shape[0])
                    np.random.shuffle(x)
                    print wf_day1[x[:wf_day1.shape[0]/2], :].shape
                    intra_J3.append(calculate_J3(pca_wf_day1[x[:wf_day1.shape[0]/2], :], pca_wf_day1[x[wf_day1.shape[0]/2:], :]))
                    # and for day 2
                    x = np.arange(wf_day2.shape[0])
                    np.random.shuffle(x)
                    intra_J3.append(calculate_J3(pca_wf_day2[x[:wf_day2.shape[0]/2], :], pca_wf_day2[x[wf_day2.shape[0]/2:], :]))

In [74]: plt.plot(np.arange(45), np.mean(data, axis = 0), linewidth = 5.0, color = 'black')
Out[74]: [<matplotlib.lines.Line2D at 0x7ff7312f8050>]

In [75]: plt.plot(np.arange(45), np.mean(data, axis = 0) + np.std(data, axis = 0), linewidth = 2.0, color = 'black', alpha = 0.3)
Out[75]: [<matplotlib.lines.Line2D at 0x7ff7312f8b50>]

In [76]: plt.plot(np.arange(45), np.mean(data, axis = 0) - np.std(data, axis = 0), linewidth = 2.0, color = 'black', alpha = 0.3)
Out[76]: [<matplotlib.lines.Line2D at 0x7ff731304190>]

In [77]: plt.show()

'''



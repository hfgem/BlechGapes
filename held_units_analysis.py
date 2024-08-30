#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:32:05 2024

@author: cmazzio + hfgem

This code uses held unit analysis results to automatically import the days to 
be compared and continues to analyses.
"""

import sys
import os

current_path = os.path.realpath(__file__)
os.chdir(('/').join(current_path.split('/')[:-1]))

import tables
import numpy as np
from tkinter.filedialog import askdirectory
from functions.held_units_funcs import *

# Ask the user for the directory where the held units analysis results live
print('Where did you save the held units results?')
held_save_dir = askdirectory()

#Import the held units csv
held_units, num_units, num_days = import_held_units_array(held_save_dir)

#Import the held units original directories / names / animal_names
directories, hdf5_names, animal_name = import_held_units_dict(held_save_dir, num_days)

#Now load spike times
spike_times_dict = import_spike_times_dict(hdf5_names,directories)
        

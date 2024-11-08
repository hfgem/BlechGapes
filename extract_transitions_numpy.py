from pytau.changepoint_io import DatabaseHandler
fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()
# Get fits for a particular experiment
dframe = fit_database.fit_database
from pytau.changepoint_analysis import PklHandler
from tqdm import tqdm
import os
import numpy as np

for i, this_row in tqdm(dframe.iterrows()):
	print(f'Processing : {this_row["data.basename"]}')
	save_path = this_row['exp.save_path']
	tau_np_save_path = save_path + '_scaled_mode_tau'
	spikes_np_save_path = save_path + '_raw_spikes'
	if not os.path.exists(np_save_path + '.npy'):
		this_handler = PklHandler(save_path)
		#Import changepoints for each delivery
		scaled_mode_tau = this_handler.tau.scaled_mode_tau #num trials x num cp
		raw_spikes = this_handler.firing.raw_spikes
		# Output to np
		np.save(tau_np_save_path, scaled_mode_tau)
		np.save(spikes_np_save_path, raw_spikes)
	else:
		print(f'np file already exists')

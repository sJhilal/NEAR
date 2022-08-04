# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:52:43 2022

@author: 39351
"""

from functions.NEAR import NEAR

path_of_folder = 'C:\\Users\\39351\\Desktop\\FBK\\NEAR_sou\\python\\'

# First example from eeglab sample data 
data_path = path_of_folder + 'test_2\\eeglab_data.set'
chanlocation_file = path_of_folder + 'test_2\\eeglab_chan32.locs'

# Second example from https://osf.io/r7s9b/
# Fails to interpolate because of following error (missing digitization info)
# RuntimeError: Cannot fit headshape without digitization , info["dig"] is None
# Also, LOF picks 37 chans as bad because data file does not only contain EEG scalp channels but also others(ear, eog,ankle... etc 73 total)
#data_path = path_of_folder + 'test_2\\sub-01_ses-02_task-SSVEP_eeg.vhdr'

lowpass=40
highpass=0.1


EEG_NEAR = NEAR(data_path,lowpass=40,highpass=0.1)

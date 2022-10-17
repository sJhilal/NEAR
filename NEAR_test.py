# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:52:43 2022

@author: sJhilal
"""

from functions.NEAR import NEAR

path_of_folder = os.path.dirname(os.path.abspath(__file__))

# Data example from eeglab sample data
data_path = os.path.join(path_of_folder, "test_data", "eeglab_data.set")
chanlocation_file = os.path.join(path_of_folder, "test_data", "eeglab_chan32.locs")

lowpass=40
highpass=0.1


EEG_NEAR = NEAR(data_path,lowpass=40,highpass=0.1)

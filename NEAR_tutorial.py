# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:36:19 2022

@author: sJhilal
"""

import mne

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from functions.flatlines import clean_flatlines
from functions.asr import ASR, sliding_window


# Step 0: Dataset Parameters 

# Change accordingly
path_of_folder = 'C:\\Users\\39351\\Desktop\\FBK\\NEAR_sou\\python\\'

# First example from eeglab sample data 
data_path = path_of_folder + 'test_2\\eeglab_data.set'
chanlocation_file = path_of_folder + 'test_2\\eeglab_chan32.locs'

# Second example from https://osf.io/r7s9b/
# Fails to interpolate because of following error (missing digitization info)
# RuntimeError: Cannot fit headshape without digitization , info["dig"] is None
# Also, LOF picks 37 chans as bad because data file does not only contain EEG scalp channels but also others(ear, eog,ankle... etc 73 total)
# data_path = path_of_folder + 'test_2\\sub-01_ses-02_task-SSVEP_eeg.vhdr'

lowpass=40
highpass=0.1

# Step 1: Import data

EEG=mne.io.read_raw(data_path, preload=True)

# Step 2: Import the channel locations


try:
    chanlocation_file
except:
    chanlocation_file = None

if chanlocation_file:
    montage = mne.channels.read_custom_montage(chanlocation_file)
    EEG = EEG.set_montage(montage=montage)

# Step 3: Set montage digitization

try:
    montage
except:
    montage = None

if montage:
    EEG.set_montage(montage=montage)

# Step 4: Filter data

if highpass and lowpass:    
    EEG.filter(highpass, lowpass)
elif highpass:
    EEG.filter(EEG.info['highpass'], lowpass)
elif lowpass:
    EEG.filter(highpass, EEG.info['lowpass'])  

    
# Step 5: Define scalp channels

try:
    scalp_channels
except:
    scalp_channels = EEG.info['ch_names']

scalp_channels_idx = [EEG.info['ch_names'].index(ch_name) for ch_name in scalp_channels]

    
# Step 6: Remove flat channels

EEG_data, new_channels_inds, removed_channels = clean_flatlines(EEG._data[scalp_channels_idx],EEG.info['sfreq'])

# Step 7: Remove bad channels using LOF

clf = LocalOutlierFactor()
LOF = clf.fit_predict(EEG_data)

flat_chans = [scalp_channels[idx] for idx in range(len(removed_channels)) if removed_channels[idx]==True]
LOF_bads = [scalp_channels[new_channels_inds[idx]] for idx in range(len(LOF)) if LOF[idx]==-1]

new_channels_inds_LOF = [new_channels_inds[idx] for idx in range(len(LOF)) if LOF[idx]==1]

all_bads = flat_chans + LOF_bads
all_bads = list(dict.fromkeys(all_bads))

EEG_data = EEG_data[[idx for idx in range(len(LOF)) if LOF[idx]==1]]
    
EEG.info['bads'] = all_bads

# Step 8: Run ASR to correct or remove bad segments

good_scalp_channels = [chan for chan in scalp_channels if chan not in all_bads]

sfreq = EEG.info['sfreq']

# Train on a clean portion of data
asr = ASR(method='euclid')
train_idx = np.arange(0 * sfreq, 30 * sfreq, dtype=int)
_, sample_mask = asr.fit(EEG_data[:, train_idx])

# Apply filter using sliding (non-overlapping) windows
X = sliding_window(EEG_data, window=int(sfreq), step=int(sfreq))
Y = np.zeros_like(X)
for i in range(X.shape[1]):
    Y[:, i, :] = asr.transform(X[:, i, :])

EEG_raw = X.reshape(EEG_data.shape[0], -1)  # reshape to (n_chans, n_times)
EEG_clean = Y.reshape(EEG_data.shape[0], -1)

EEG.crop(tmax=len(EEG_clean[1])/sfreq, include_tmax=False)
EEG._data[new_channels_inds_LOF] = EEG_clean

non_scalp_channels = [chan for chan in EEG.info['ch_names'] if chan not in scalp_channels]
EEG.interpolate_bads(exclude=non_scalp_channels)
    

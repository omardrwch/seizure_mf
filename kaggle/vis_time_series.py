"""
Visualize raw data
"""

import numpy as np
import os

import matplotlib.pyplot as plt
import utils
import config as cfg

import mne

subject = 'Dog_4'
s_freq   = cfg.info[subject]['s_freq'] 
interictal_file_idx = 6
preictal_file_idx   = 6

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------

interictal_data = utils.get_interictal_data(interictal_file_idx, subject)['data']
preictal_data   = utils.get_preictal_data(preictal_file_idx, subject)['data']
n_channels = interictal_data.shape[0]

#-------------------------------------------------------------------------------
# Plot
#-------------------------------------------------------------------------------

# Initialize an info structure
info = mne.create_info(
    ch_names=['iEEG%d'%i for i in range(1, n_channels+1)],
    ch_types=['eeg']*n_channels,
    sfreq=s_freq
)


raw_interictal = mne.io.RawArray(interictal_data, info)
raw_preictal   = mne.io.RawArray(preictal_data,   info)

raw_interictal.plot(scalings='auto', duration = 60, show = False) 
raw_preictal.plot(scalings='auto', duration = 60, show = False)

plt.show()
import numpy as np
import mfanalysis as mf
import os
from scipy.io import loadmat

import matplotlib.pyplot as plt
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))


# subject               = 'Patient_1'
# interictal_files_idx  = np.arange(1, 51)
# preictal_files_idx    = np.arange(1, 19)
# n_channels            = 15
# ALL_FILES = True


subject               = 'Dog_3'
interictal_files_idx  = np.arange(20, 30)
preictal_files_idx    = list(range(1,18))
n_channels            = 15
ALL_FILES = True

s_freq  = 399.609756
fmin    = 0.2
fmax    = 2      
j1,j2   = utils.get_scales(s_freq, fmin, fmax)

# interictal_files_idx  = np.arange(30, 32)
# preictal_files_idx    = np.arange(10, 12)
# n_channels            = 3




# MF analysis object
mfa = mf.MFA()
mfa.wt_name = 'db3'
mfa.p = np.inf
mfa.j1 = j1      # [11, 15] for low freqs,  [5, 9] for [8, 230] Hz,  [6, 9] for [8, 100] Hz
mfa.j2 = j2
mfa.q = [2]#np.arange(-8, 9)
mfa.n_cumul = 3
mfa.gamint = 1
mfa.verbose = 1
mfa.wtype = 0


if not ALL_FILES:
    # Get interictal signal
    interictal_signal = utils.get_interictal_data(18, subject)['data'][0,:]

    mfa.analyze(interictal_signal)

    mfa.plot_cumulants(show = False)
    mfa.plot_structure(show = False)
    mfa.plot_spectrum(show = False)
    plt.show()


    # Get preictal signal
    preictal_signal = utils.get_preictal_data(18, subject)['data'][0,:]

    mfa.analyze(preictal_signal)

    mfa.plot_cumulants(show = False)
    mfa.plot_structure(show = False)
    mfa.plot_spectrum(show = False)
    plt.show()


if ALL_FILES:
    #------------------------------------------------------------------------
    # All channels interictal
    #------------------------------------------------------------------------

    c1_inter = np.zeros((len(interictal_files_idx), n_channels))
    c2_inter = np.zeros((len(interictal_files_idx), n_channels))

    cumulants_interictal = [0 for i in range(n_channels)]

    for ii, file_idx in enumerate(interictal_files_idx):
        interictal_data = utils.get_interictal_data(file_idx, subject)['data']
        for channel in range(n_channels):
            signal = interictal_data[channel, :]
            mfa.analyze(signal)
            cp  = mfa.cumulants.log_cumulants

            c1_inter[ii, channel] = cp[0]
            c2_inter[ii, channel] = cp[1]

            print("----------------- interictal", channel)

            if ii == 0:
                cumulants_interictal[channel] = mfa.cumulants
            else:
                cumulants_interictal[channel].sum(mfa.cumulants)


    #------------------------------------------------------------------------
    # All channels preictal
    #------------------------------------------------------------------------

    c1_pre = np.zeros((len(preictal_files_idx), n_channels))
    c2_pre = np.zeros((len(preictal_files_idx), n_channels))

    cumulants_preictal = [0 for i in range(n_channels)]

    for ii, file_idx in enumerate(preictal_files_idx):
        preictal_data = utils.get_preictal_data(file_idx, subject)['data']

        for channel in range(n_channels):
            signal = preictal_data[channel, :]
            mfa.analyze(signal)
            cp  = mfa.cumulants.log_cumulants

            c1_pre[ii, channel] = cp[0]
            c2_pre[ii, channel] = cp[1]


            print("----------------- preictal", channel)

            if ii == 0:
                cumulants_preictal[channel] = mfa.cumulants
            else:
                cumulants_preictal[channel].sum(mfa.cumulants)


    utils.compare_distributions(c1_inter, c1_pre, n_channels, 'c1 interictal vs. preictal')
    utils.compare_distributions(c2_inter, c2_pre, n_channels, 'c2 interictal vs. preictal')


    utils.plot_cumulant_time_evolution(c1_inter, c2_inter, title = 'interictal')
    utils.plot_cumulant_time_evolution(c1_pre, c2_pre, title = 'preictal')


    for ii in range(n_channels):
        cumulants_interictal[ii].plot('interictal, channel %d'%ii)
        cumulants_preictal[ii].plot('preictal, channel %d'%ii)
        plt.show()

    plt.show()
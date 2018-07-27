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


subject               = 'Patient_1'
interictal_files_idx  = np.arange(15, 30)
preictal_files_idx    = list(range(1,19))
n_channels            = 15  # 15 for people, 16 for dogs
ALL_FILES = True

s_freq  = 5000 #399.609756
fmin    = 0.2
fmax    = 2  
j1,j2   = utils.get_scales(s_freq, fmin, fmax)
# j1 = 7
# j2 = 12

# interictal_files_idx  = np.arange(30, 32)
# preictal_files_idx    = np.arange(10, 12)
# n_channels            = 3


# MF analysis object
mfa = mf.MFA()
mfa.wt_name = 'db4'             # !!!
mfa.p  = 0.25                   # !!!
mfa.j1 = j1     
mfa.j2 = j2
mfa.q  = [2]
mfa.n_cumul = 4
mfa.gamint  = 2.0             # !!!
mfa.verbose = 1
mfa.wtype   = 0


if not ALL_FILES:

    mfa.q = np.arange(-8, 9)

    # Get interictal signal
    interictal_signal = utils.get_interictal_data(2, subject)['data'][11,:]

    mfa.analyze(interictal_signal)

    structure_dwt = mf.StructureFunction(mfa.wavelet_coeffs,
                                         mfa.q,
                                         mfa.j1,
                                         mfa.j2_eff,
                                         mfa.wtype)

    
    mfa.plot_cumulants(show = False)
    mfa.plot_structure(show = False)
    mfa.plot_spectrum (show = False)
    plt.figure(); structure_dwt.plot('A', 'B')
    plt.show()


    # Get preictal signal
    preictal_signal = utils.get_preictal_data(2, subject)['data'][11,:]

    mfa.analyze(preictal_signal)

    # structure_dwt = mf.StructureFunction(mfa.wavelet_coeffs,
    #                                      mfa.q,
    #                                      mfa.j1,
    #                                      mfa.j2_eff,
    #                                      mfa.wtype)

    mfa.plot_cumulants(show = False)
    mfa.plot_structure(show = False)
    mfa.plot_spectrum(show = False)
    # plt.figure(); structure_dwt.plot('C', 'D')
    plt.show()


if ALL_FILES:
    #------------------------------------------------------------------------
    # All channels interictal
    #------------------------------------------------------------------------

    H_inter = np.zeros((len(interictal_files_idx), n_channels))
    c1_inter = np.zeros((len(interictal_files_idx), n_channels))
    c2_inter = np.zeros((len(interictal_files_idx), n_channels))
    c3_inter = np.zeros((len(interictal_files_idx), n_channels))
    c4_inter = np.zeros((len(interictal_files_idx), n_channels))

    cumulants_interictal = [0 for i in range(n_channels)]

    for ii, file_idx in enumerate(interictal_files_idx):
        interictal_data = utils.get_interictal_data(file_idx, subject)['data']
        print("----------------- interictal file", file_idx)
        for channel in range(n_channels):
            signal = interictal_data[channel, :]
            mfa.analyze(signal)

            structure_dwt = mf.StructureFunction(mfa.wavelet_coeffs,
                                                 [2],
                                                 mfa.j1,
                                                 mfa.j2_eff,
                                                 mfa.wtype)

            cp  = mfa.cumulants.log_cumulants

            H_inter[ii, channel]  = structure_dwt.zeta[0]/2

            # if H_inter[ii, channel] < 0:
            #     print(file_idx, channel)

            c1_inter[ii, channel] = cp[0]
            c2_inter[ii, channel] = cp[1]
            c3_inter[ii, channel] = cp[2]
            c4_inter[ii, channel] = cp[3]

            if ii == 0:
                cumulants_interictal[channel] = mfa.cumulants
            else:
                cumulants_interictal[channel].sum(mfa.cumulants)


    #------------------------------------------------------------------------
    # All channels preictal
    #------------------------------------------------------------------------
    H_pre = np.zeros((len(preictal_files_idx), n_channels))
    c1_pre = np.zeros((len(preictal_files_idx), n_channels))
    c2_pre = np.zeros((len(preictal_files_idx), n_channels))
    c3_pre = np.zeros((len(preictal_files_idx), n_channels))
    c4_pre = np.zeros((len(preictal_files_idx), n_channels))


    cumulants_preictal = [0 for i in range(n_channels)]

    for ii, file_idx in enumerate(preictal_files_idx):
        preictal_data = utils.get_preictal_data(file_idx, subject)['data']
        print("----------------- preictal file", file_idx)
        for channel in range(n_channels):
            signal = preictal_data[channel, :]
            mfa.analyze(signal)

            structure_dwt = mf.StructureFunction(mfa.wavelet_coeffs,
                                                 [2],
                                                 mfa.j1,
                                                 mfa.j2_eff,
                                                 mfa.wtype)
 
            cp  = mfa.cumulants.log_cumulants

            H_pre[ii, channel]  = structure_dwt.zeta[0]/2
            c1_pre[ii, channel] = cp[0]
            c2_pre[ii, channel] = cp[1]
            c3_pre[ii, channel] = cp[2]
            c4_pre[ii, channel] = cp[3]


            if ii == 0:
                cumulants_preictal[channel] = mfa.cumulants
            else:
                cumulants_preictal[channel].sum(mfa.cumulants)


    utils.compare_distributions(H_inter,  H_pre, n_channels,  'H interictal vs. preictal')
    utils.compare_distributions(c1_inter, c1_pre, n_channels, 'c1 interictal vs. preictal')
    utils.compare_distributions(c2_inter, c2_pre, n_channels, 'c2 interictal vs. preictal')
    utils.compare_distributions(c3_inter, c3_pre, n_channels, 'c3 interictal vs. preictal')
    utils.compare_distributions(c4_inter, c4_pre, n_channels, 'c4 interictal vs. preictal')

    utils.plot_cumulant_time_evolution(H_inter, c2_inter, title = 'H and c2 interictal')
    utils.plot_cumulant_time_evolution(c1_inter, c2_inter, title = 'interictal')
    utils.plot_cumulant_time_evolution(H_pre, c2_pre, title = 'H and c2 preictal')
    utils.plot_cumulant_time_evolution(c1_pre, c2_pre, title = 'preictal')


    for ii in range(n_channels):
        cumulants_interictal[ii].plot('interictal, channel %d'%ii)
        cumulants_preictal[ii].plot('preictal, channel %d'%ii)
        plt.show()

    plt.show()
"""
For more info about the dataset:
http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3
"""

import numpy as np 
import pandas as pd
from scipy.signal import welch
import os.path as op
import matplotlib.pyplot as plt
from scipy.stats import linregress



# Useful dictionary
set2letter = {'A':'Z', 
              'B':'O',
              'C':'N',
              'D':'F',
              'E':'S'}


# Parameters
s_freq = 173.61 # sample frequency; but spectrum is only meaninful
                # in the range [0.5, 85] Hz

mf_params = {'wt_name': 'db3',
             'j1': 5,
             'j2': 9,
             'q':  np.array([2]),
             'n_cumul': 2,
             'gamint': 1.55,
             'wtype': 0,
             'verbose': 1
             }   # value of p need to be defined


def read_file(set_name, index):
    """
    set_name = A, B, C, D or E
    index = from 1 to 100
    """
    path = op.join('./data', 'set'+set_name)
    try:
        fname = set2letter[set_name] + str(index).zfill(3) + '.txt'
        _data = pd.read_csv(op.join(path, fname), sep='\n', header = None)
    except:
        fname = set2letter[set_name] + str(index).zfill(3) + '.TXT'
        _data = pd.read_csv(op.join(path, fname), sep='\n', header = None)

    return np.array(_data.values, dtype = float).squeeze()


def get_scales(fs, fmin, fmax):
    """
    Compute scales corresponding to the analyzed frequencies
    """
    f0 = (3.0/4.0)*fs
    j1 = int(np.ceil(np.log2(f0/fmax)))
    j2 = int(np.ceil(np.log2(f0/fmin)))
    return j1, j2


def plot_psd(signal, fs, name = '', f1 = None, f2 = None, nperseg = 1024):
    f, px = welch(signal, fs, scaling = 'spectrum', nperseg=nperseg)
    plt.figure()
    plt.loglog(f, px)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Power spectrum - ' + name)
    if (f1 is not None) and (f2 is not None):
        ff = f[ np.logical_and(f>=f1, f<=f2) ].copy()
        PP = px[ np.logical_and(f>=f1, f<=f2) ].copy()
        log_ff = np.log10(ff)
        log_PP = np.log10(PP)
        slope, intercept, r_value, p_value, std_err = linregress(log_ff,log_PP)
        log_PP_fit = slope*log_ff + intercept
        PP_fit    =  10.0**(log_PP_fit)
        plt.loglog(ff, PP_fit, label = 'beta=%f'%(slope))
        plt.legend()
    plt.grid()
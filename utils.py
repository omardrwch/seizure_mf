import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from scipy.signal import welch

def get_scales(fs, min_f, max_f):
    """
    Compute scales corresponding to the analyzed frequencies
    """
    f0 = (3.0/4.0)*fs
    j1 = int(np.ceil(np.log2(f0/max_f)))
    j2 = int(np.ceil(np.log2(f0/min_f)))   
    return j1, j2 
    

def get_interictal_data(file_idx, subject):
    filename = subject + '_interictal_segment_%s.mat'%(str(file_idx).zfill(4))
    filename = os.path.join('prediction_data', subject, filename)
    contents = loadmat(filename)

    output = {}
    output['data']       = contents['interictal_segment_%d'%file_idx][0][0][0]
    output['length_sec'] = contents['interictal_segment_%d'%file_idx][0][0][1][0][0]
    output['s_freq']     = contents['interictal_segment_%d'%file_idx][0][0][2][0][0]
    output['channels']   = contents['interictal_segment_%d'%file_idx][0][0][3].squeeze()
    output['sequence']   = contents['interictal_segment_%d'%file_idx][0][0][4][0][0]
    return output

def get_preictal_data(file_idx, subject):
    filename = subject + '_preictal_segment_%s.mat'%(str(file_idx).zfill(4))
    filename = os.path.join('prediction_data', subject, filename)
    contents = loadmat(filename)

    output = {}
    output['data']       = contents['preictal_segment_%d'%file_idx][0][0][0]
    output['length_sec'] = contents['preictal_segment_%d'%file_idx][0][0][1][0][0]
    output['s_freq']     = contents['preictal_segment_%d'%file_idx][0][0][2][0][0]
    output['channels']   = contents['preictal_segment_%d'%file_idx][0][0][3].squeeze()
    output['sequence']   = contents['preictal_segment_%d'%file_idx][0][0][4][0][0]
    return output



def compare_distributions(data1, data2, n_channels, title = ''):
    """
    For each cortical region, plot mean and std for train and test data.

    data1:  shape (n_samples,  n_features)
    data2:  shape (n_samples,  n_features)
    """
    plt.figure()
    plt.title(title)
    plt.plot(np.arange(n_channels), np.median(data1, axis=0), 'bo-', label = 'interictal (median)')
    plt.plot(np.arange(n_channels), np.median(data2, axis=0), 'ro-', label = 'preictal (median)')
    plt.legend()


    plt.fill_between(np.arange(n_channels),data1.mean(axis=0) - data1.std(axis=0),
                                data1.mean(axis=0) + data1.std(axis=0), alpha=0.25,
                             color="b")
    plt.fill_between(np.arange(n_channels),data2.mean(axis=0) - data2.std(axis=0),
                             data2.mean(axis=0) + data2.std(axis=0), alpha=0.25,
                             color="r")

    plt.grid()



def plot_cumulant_time_evolution(c1, c2, title = ''):
    plt.figure()
    plt.title(title)
    plt.subplot(211)
    plt.plot( np.arange(c1.shape[0]), c1, 'o-' )
    plt.xlabel('sample')
    plt.ylabel('c1')
    plt.subplot(212)
    plt.plot( np.arange(c2.shape[0]), c2, 'o-' )   
    plt.xlabel('sample')
    plt.ylabel('c2')



def plot_spectrum(signal, fs, nperseg = 1024):
    f, Pxx_spec = welch(signal, fs, nperseg=nperseg, scaling='spectrum')
    plt.figure()
    plt.loglog(f, np.sqrt(Pxx_spec))
    plt.xlabel(' (log) frequency [Hz]')
    plt.ylabel(' (log) Power [V RMS]')
    
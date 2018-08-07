import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
import config as cfg
import h5py
from scipy.stats import linregress


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


def get_test_data(file_idx, subject, folder = None):
    filename = subject + '_test_segment_%s.mat'%(str(file_idx).zfill(4))
    filename = os.path.join('prediction_data', subject, filename)
    if folder is not None:
        filename = os.path.join(folder, filename)

    contents = loadmat(filename)

    output = {}
    output['data']       = contents['test_segment_%d'%file_idx][0][0][0]
    output['length_sec'] = contents['test_segment_%d'%file_idx][0][0][1][0][0]
    output['s_freq']     = contents['test_segment_%d'%file_idx][0][0][2][0][0]
    output['channels']   = contents['test_segment_%d'%file_idx][0][0][3].squeeze()
    # output['sequence']   = contents['test_segment_%d'%file_idx][0][0][4][0][0]

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



def load_classif_data(subject, options):

    # Setup
    p_idx = options['p_idx']
    features  = options['features'] # e.g, ['c1', 'c2'], elements should be
                                    # in ['hurst', 'c1', 'c2', 'c3', 'c4']

    interictal_files_idx = cfg.info[subject]['interictal_files_idx']
    preictal_files_idx  = cfg.info[subject]['preictal_files_idx']
    n_interictal        = len(interictal_files_idx)
    n_preictal          = len(preictal_files_idx)
    n_channels          = cfg.info[subject]['n_channels']

    n_features          = n_channels*len(features)


    # Load interictal data
    X_interictal = np.zeros((n_interictal, n_features))
    sequence_interictal = np.zeros(n_interictal)

    for ii, file_idx in enumerate(interictal_files_idx):
        filename = 'cumulants_interictal_%d_p_%d.h5'%(file_idx, p_idx)
        filename = os.path.join('cumulant_features', subject, filename)

        with h5py.File(filename, "r") as file:
            temp = ()
            for feat in features:
                aux = file[feat][:]
                if feat =='c2' and options['clip_c2']:
                    aux = aux.clip(max = 0)

                temp = temp + (aux,)




            X_interictal[ii, :] = np.hstack(temp)
            sequence_interictal[ii] = file['sequence'].value

    y_interictal = np.zeros(n_interictal)

    # Load preictal data
    X_preictal = np.zeros((n_preictal, n_features))
    sequence_preictal = np.zeros(n_preictal)

    for ii, file_idx in enumerate(preictal_files_idx):
        filename = 'cumulants_preictal_%d_p_%d.h5'%(file_idx, p_idx)
        filename = os.path.join('cumulant_features', subject ,filename)

        with h5py.File(filename, "r") as file:
            temp = ()
            for feat in features:
                aux = file[feat][:]

                if feat =='c2' and options['clip_c2']:
                    aux = aux.clip(max = 0)

                temp = temp + (aux,)

            X_preictal[ii, :] = np.hstack(temp)
            sequence_preictal[ii] = file['sequence'].value

    y_preictal = np.ones(n_preictal)


    # Merge
    X = np.vstack((X_interictal, X_preictal))
    y = np.hstack((y_interictal, y_preictal))


    return X, y, sequence_interictal, sequence_preictal

def load_classif_test_data(subject, options, test_folder = None):

    # Setup
    p_idx     = options['p_idx']
    features  = options['features'] # e.g, ['c1', 'c2'], elements should be
                                    # in ['hurst', 'c1', 'c2', 'c3', 'c4']

    test_files_idx  = cfg.info[subject]['test_files_idx']
    n_test          = len(test_files_idx)
    n_channels      = cfg.info[subject]['n_channels']
    n_features      = n_channels*len(features)


    # Load interictal data
    X = np.zeros((n_test, n_features))

    for ii, file_idx in enumerate(test_files_idx):
        filename = 'cumulants_test_%d_p_%d.h5'%(file_idx, p_idx)
        filename = os.path.join('cumulant_features', subject, filename)
        if test_folder is not None:
            filename = os.path.join(test_folder, filename)


        with h5py.File(filename, "r") as file:
            temp = ()
            for feat in features:
                aux = file[feat][:]
                if feat =='c2' and options['clip_c2']:
                    aux = aux.clip(max = 0)

                temp = temp + (aux,)

            X[ii, :] = np.hstack(temp)
    return X



def load_classif_data_2(subject, options, j1, j2):
    """
    Allow us to select scales.
    """

    # Index
    ind_j1 = j1 - 1
    ind_j2 = j2 - 1

    # Setup
    p_idx     = options['p_idx']
    features  = options['features'] # e.g, ['c1', 'c2'], elements should be
                                    # in ['hurst', 'c1', 'c2', 'c3', 'c4']

    interictal_files_idx = cfg.info[subject]['interictal_files_idx']
    preictal_files_idx   = cfg.info[subject]['preictal_files_idx']
    n_interictal         = len(interictal_files_idx)
    n_preictal           = len(preictal_files_idx)
    n_channels           = cfg.info[subject]['n_channels']

    n_features           = n_channels*len(features)

    # Load interictal data
    X_interictal = np.zeros((n_interictal, n_features))
    sequence_interictal = np.zeros(n_interictal)

    for ii, file_idx in enumerate(interictal_files_idx):
        filename = 'cumulants_interictal_%d_p_%d.h5'%(file_idx, p_idx)
        filename = os.path.join('cumulant_features_2', subject, filename)

        with h5py.File(filename, "r") as file:
            temp = ()
            for feat in features:
                if feat == 'hurst':
                    aux = file[feat][:]

                else:
                    x_reg = np.arange(j1, j2+1)
                    aux_j = file[feat+'j'][:]
                    aux_j = aux_j[:, ind_j1:ind_j2+1]
                    n_channels = aux_j.shape[0]
                    aux  = np.zeros(n_channels)
                    for ch in range(n_channels):
                        y_reg = aux_j[ch, :]
                        slope, intercept, _, _, _ = linregress(x_reg,y_reg)
                        aux[ch] = np.log2(np.exp(1))*slope


                # clip c2
                if feat =='c2' and options['clip_c2']:
                    aux = aux.clip(max = 0)

                temp = temp + (aux,)


            X_interictal[ii, :] = np.hstack(temp)
            sequence_interictal[ii] = file['sequence'].value

    y_interictal = np.zeros(n_interictal)

    # Load preictal data
    X_preictal = np.zeros((n_preictal, n_features))
    sequence_preictal = np.zeros(n_preictal)

    for ii, file_idx in enumerate(preictal_files_idx):
        filename = 'cumulants_preictal_%d_p_%d.h5'%(file_idx, p_idx)
        filename = os.path.join('cumulant_features_2', subject ,filename)

        with h5py.File(filename, "r") as file:
            temp = ()
            for feat in features:
                if feat == 'hurst':
                    aux = file[feat][:]

                else:
                    x_reg = np.arange(j1, j2+1)
                    aux_j = file[feat+'j'][:]
                    aux_j = aux_j[:, ind_j1:ind_j2+1]
                    n_channels = aux_j.shape[0]
                    aux  = np.zeros(n_channels)
                    for ch in range(n_channels):
                        y_reg = aux_j[ch, :]
                        slope, intercept, _, _, _ = linregress(x_reg,y_reg)
                        aux[ch] = np.log2(np.exp(1))*slope


                # clip c2
                if feat =='c2' and options['clip_c2']:
                    aux = aux.clip(max = 0)

                temp = temp + (aux,)

            X_preictal[ii, :] = np.hstack(temp)
            sequence_preictal[ii] = file['sequence'].value

    y_preictal = np.ones(n_preictal)


    # Merge
    X = np.vstack((X_interictal, X_preictal))
    y = np.hstack((y_interictal, y_preictal))


    return X, y, sequence_interictal, sequence_preictal



def load_cumulant(subject, condition, cumulant, options, j1, j2):
    """
    cumulant: 'c1', 'c2', 'c3' or 'c4'
    """

    # Index
    ind_j1 = j1 - 1
    ind_j2 = j2 - 1
    x_reg = np.arange(j1, j2+1)

    # Setup
    p_idx     = options['p_idx']

    files_idx   = cfg.info[subject][condition+'_files_idx']
    n_files     = len(files_idx)
    n_channels  = cfg.info[subject]['n_channels']

    # Load data
    cumulants = None
    log_cumulants = None
    # log_cumulants_debug = None

    sequence = np.zeros(n_files)

    for ii, file_idx in enumerate(files_idx):
        filename = 'cumulants_interictal_%d_p_%d.h5'%(file_idx, p_idx)
        filename = os.path.join('cumulant_features_2', subject, filename)

        with h5py.File(filename, "r") as file:
            sequence[ii] = file['sequence'].value

            cumul     = file[cumulant+'j'][:]
            # log_cumul_debug = file[cumulant][:]
            n_channels = cumul.shape[0]
            log_cumul  = np.zeros(n_channels)
            for ch in range(n_channels):
                y_reg = cumul[ch, ind_j1:ind_j2+1]
                slope, intercept, _, _, _ = linregress(x_reg,y_reg)
                log_cumul[ch] = np.log2(np.exp(1))*slope          

        if ii == 0:
            cumulants = np.zeros((n_files, cumul.shape[0], cumul.shape[1])) # (file, sensor, nj)
            log_cumulants = np.zeros((n_files, len(log_cumul))) # (file, sensor)
            # log_cumulants_debug = np.zeros((n_files, len(log_cumul))) # (file, sensor)


        cumulants[ii,:,:] = cumul
        log_cumulants[ii,:] = log_cumul
        # log_cumulants_debug[ii, :] = log_cumul_debug

    return cumulants, log_cumulants, sequence #, log_cumulants_debug

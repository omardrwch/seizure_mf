"""
Visualization of cumulants stored in /cumulant_features_2
"""

import numpy as np
import os

import matplotlib.pyplot as plt
import utils
import config as cfg
from scipy.stats import linregress


#--------------------------------------------------------------------
# Choose subject and options
#--------------------------------------------------------------------
subject = 'Dog_2'
options = {'p_idx': 2}
fmin    = 0.2
fmax    = 2

seq = 1 # sequence

s_freq  = cfg.info[subject]['s_freq']
j1,j2   = utils.get_scales(s_freq, fmin, fmax)

#--------------------------------------------------------------------
# Load data
#--------------------------------------------------------------------
C1j_inter, c1_inter, _ = utils.load_cumulant(subject, 'interictal', 'c1', options, j1, j2)
C2j_inter, c2_inter, seqs_inter = utils.load_cumulant(subject, 'interictal', 'c2', options, j1, j2)

C1j_pre, c1_pre, _ = utils.load_cumulant(subject, 'preictal', 'c1', options, j1, j2)
C2j_pre, c2_pre, seqs_pre = utils.load_cumulant(subject, 'preictal', 'c2', options, j1, j2)

#--------------------------------------------------------------------
# Select sequence
#--------------------------------------------------------------------
# def select_sequence(data, sequence, target_sequence):
#     data = data[sequence == target_sequence, :]
#     return data

# c1_inter = select_sequence(c1_inter, seqs_inter, seq)
# c2_inter = select_sequence(c2_inter, seqs_inter, seq)
# c1_pre   = select_sequence(c1_pre, seqs_pre,   seq)
# c2_pre   = select_sequence(c2_pre, seqs_pre,   seq)

#--------------------------------------------------------------------
# Average over all sensors
#--------------------------------------------------------------------
c1_inter = c1_inter.mean(axis = 1).reshape(-1,1)
c2_inter = c2_inter.mean(axis = 1).reshape(-1,1)
c1_pre = c1_pre.mean(axis = 1).reshape(-1,1)
c2_pre = c2_pre.mean(axis = 1).reshape(-1,1)

print('c1 interictal = %0.5f +- %0.5f'%(c1_inter.mean(axis = 0), c1_inter.std(axis = 0)))
print('c1 preictal = %0.5f +- %0.5f'%(c1_pre.mean(axis = 0), c1_pre.std(axis = 0)))

print('c2 interictal = %0.5f +- %0.5f'%(c2_inter.mean(axis = 0), c2_inter.std(axis = 0)))
print('c2 preictal   = %0.5f +- %0.5f'%(c2_pre.mean(axis = 0), c2_pre.std(axis = 0)))


plt.figure()
plt.plot(c1_inter, c2_inter, 'bo')
plt.plot(c1_pre, c2_pre, 'ro')


#--------------------------------------------------------------------
# Visualize
#--------------------------------------------------------------------

def plot_cumulants(cumulants_list, j1=j1, j2=j2, title = '', labels = None):
    colors = ['b', 'r']
    x_reg  = np.arange(j1, j2+1)
    plt.figure()
    plt.title(title)
    for ii, cumulants in enumerate(cumulants_list):
        x_plot = np.arange(1, len(cumulants)+1)
        y_plot = cumulants
        y_reg  = cumulants[j1-1:j2]
        
        plt.plot(x_plot, y_plot, colors[ii]+'o--', alpha = 0.75)


        # linear regression
        log2_e  = np.log2(np.exp(1))
        slope, intercept, r_value, p_value, std_err = linregress(x_reg,y_reg)
        y1 = slope*j1 + intercept
        y2 = slope*j2 + intercept
        log_cumul = log2_e*slope
        if labels is not None:
            plt.plot( [j1, j2], [y1, y2], colors[ii]+'-', linewidth=2,
                      label = labels[ii]+', slope*log2(e) = %0.3f'%log_cumul)
        else:
            plt.plot( [j1, j2], [y1, y2], colors[ii]+'-', linewidth=2,
                      label = 'slope*log2(e) = %0.3f'%log_cumul)


    plt.xlabel('j')
    plt.ylabel('C(j)')
    plt.legend()
    plt.grid(True)



utils.compare_distributions(c1_inter, c1_pre, c1_inter.shape[1], 'c1 interictal vs. preictal')
utils.compare_distributions(c2_inter, c2_pre, c2_inter.shape[1], 'c2 interictal vs. preictal')

plot_cumulants([C1j_inter.mean(axis=0).mean(axis=0),
                C1j_pre.mean(axis=0).mean(axis=0)],
                title = 'C1',
                labels = ['interictal', 'preictal'])

plot_cumulants([C2j_inter.mean(axis=0).mean(axis=0),
                C2j_pre.mean(axis=0).mean(axis=0)],
                title = 'C2',
                labels = ['interictal', 'preictal'])


plt.show()
"""
Compare MF properties of healthy and ictal signals.
""" 

import numpy as np
import os
import mfanalysis as mf
import matplotlib.pyplot as plt
import utils

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20

N_SPLITS = 3
scoring = 'roc_auc'

#----------------------------------------------------------------
# Select sets and labels
#----------------------------------------------------------------
sets = ['C', 'D', 'E']
labels = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 1}
files_per_set = 100

#----------------------------------------------------------------
# Create MF object for feature extraction
#----------------------------------------------------------------
p = 2  # value for p-leaders
mfa = mf.MFA(**utils.mf_params)
mfa.n_cumul = 2
mfa.p = p


#----------------------------------------------------------------
# Feature extraction
#----------------------------------------------------------------
n_samples = files_per_set*len(sets)
n_features = mfa.n_cumul

all_data = np.zeros((n_samples, 1, 4097))


X = np.zeros((n_samples, n_features))
y = -1*np.ones(n_samples)
count = 0
for set_name in sets:
    for index in range(1, files_per_set+1):
        data = utils.read_file(set_name, index)

        mfa.analyze(data)
        X[count, :] = mfa.cumulants.log_cumulants
        y[count]    = labels[set_name]


        all_data[count, 0, :] = data

        count += 1

c1_healthy = X[y==0, 0]
c2_healthy = X[y==0, 1]

c1_ictal   = X[y==1, 0]
c2_ictal   = X[y==1, 1]


print('c1 healthy: %0.5f +- %0.5f'%(c1_healthy.mean(), c1_healthy.std()))
print('c1 ictal: %0.5f +- %0.5f'%(c1_ictal.mean(), c1_ictal.std()))
print("----")
print('c2 healthy: %0.5f +- %0.5f'%(c2_healthy.mean(), c2_healthy.std()))
print('c2 ictal: %0.5f +- %0.5f'%(c2_ictal.mean(), c2_ictal.std()))


plt.figure()
plt.plot(c1_healthy, c2_healthy, 'bo', label = 'normal')
plt.plot(c1_ictal, c2_ictal, 'ro', label = 'seizure')
plt.xlabel('$c_1$')
plt.ylabel('$c_2$')
plt.legend()
plt.ylim([-0.2, 0.1])
plt.show()
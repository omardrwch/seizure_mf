import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import welch

import matplotlib.pyplot as plt
import utils

subject = 'Dog_1'
s_freq  = 399.609756

# Get interictal signal
interictal_signal = utils.get_interictal_data(1, subject)['data'][0,:]
# Get preictal signal
preictal_signal   = utils.get_preictal_data(1, subject)['data'][0,:]


utils.plot_spectrum(interictal_signal, s_freq, 50000)
utils.plot_spectrum(preictal_signal, s_freq, 50000)

plt.show()
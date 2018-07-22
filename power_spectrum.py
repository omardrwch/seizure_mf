import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import welch

import matplotlib.pyplot as plt
import utils

subject = 'Patient_1'

# Get interictal signal
interictal_signal = utils.get_interictal_data(1, subject)['data'][0,:]
# Get preictal signal
preictal_signal   = utils.get_preictal_data(1, subject)['data'][0,:]


utils.plot_spectrum(interictal_signal, 5000, 20000)
utils.plot_spectrum(preictal_signal, 5000, 20000)

plt.show()
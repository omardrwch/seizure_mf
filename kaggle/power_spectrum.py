import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import welch

import matplotlib.pyplot as plt
import utils

subject = 'Dog_4'
s_freq  = 399.609756


for ii in range(3):
	# Get interictal signal
	interictal_signal = utils.get_interictal_data(8, subject)['data'][ii*2,:]
	# Get preictal signal
	preictal_signal   = utils.get_preictal_data(1, subject)['data'][ii*2,:]

	utils.plot_psd(interictal_signal, s_freq, name = 'interictal', f1 = 0.2, f2 = 2, nperseg = 50000)
	utils.plot_psd(preictal_signal, s_freq, name = 'preictal', f1 = 0.2, f2 = 2, nperseg = 50000)



	plt.show()
import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import welch

import matplotlib.pyplot as plt
import utils

subject = 'Dog_3'
s_freq  = 399.609756


for ii in range(3):
	# Get interictal signal
	interictal_signal = utils.get_interictal_data(5, subject)['data'][ii*2,:]
	# Get preictal signal
	preictal_signal   = utils.get_preictal_data(5, subject)['data'][ii*2,:]


	utils.plot_spectrum(interictal_signal, s_freq, 10000)
	utils.plot_spectrum(preictal_signal, s_freq, 10000)

	plt.show()
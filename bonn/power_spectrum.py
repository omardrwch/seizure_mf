import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import welch

import matplotlib.pyplot as plt
import utils


data = utils.read_file('E', 60)
s_freq = utils.s_freq

# utils.plot_spectrum(data, s_freq, 512, 0.5, 8)
utils.plot_psd(data, s_freq, name = '', f1 = 0.5, f2 = 8, nperseg = 1024*1.5)
plt.show()

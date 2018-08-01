import numpy as np
import os
import mfanalysis as mf
import matplotlib.pyplot as plt
import utils

s_freq = utils.s_freq
fmin = 0.5
fmax = 8.0

j1, j2 = utils.get_scales(s_freq, fmin, fmax)

mfa = mf.MFA()
mfa.p = np.inf
mfa.wtype = 0
mfa.j1 = j1
mfa.j2 = j2
mfa.gamint = 1.5
mfa.q = np.arange(-8,9)
mfa.verbose = 0.0


for ii in range(1, 101):
    data = utils.read_file('E', ii)
    mfa.analyze(data)
    if mfa.hmin < 0:
        print("negative hmin !!!!!!")
    if mfa.eta_p < 0:
        print("negative eta_p !!!!!!")



# mfa.verbose = 2.0

# data = utils.read_file('E', 5)
# mfa.analyze(data)
# if mfa.hmin < 0:
#     print("negative hmin !!!!!!")
# if mfa.eta_p < 0:
#     print("negative eta_p !!!!!!")

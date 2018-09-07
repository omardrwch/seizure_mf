import numpy as np
import os
import mfanalysis as mf
import matplotlib.pyplot as plt
import utils

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20


data_normal = utils.read_file('A', 12) 
# data_normal += data_normal.std()*np.random.normal(loc = 0, scale = 1.0, size=len(data_normal))

data_ictal  = utils.read_file('E', 98) 
# data_ictal += data_ictal.std()*np.random.normal(loc = 0, scale = 1.0, size=len(data_ictal))



plt.figure()
plt.subplot(1,2,1)
plt.plot(data_normal, 'b-')
plt.title('Normal')


plt.subplot(1,2,2)
plt.plot(data_ictal, 'r-')
plt.title('Seizure')



#----------------------------------------------------------------
# Create MF object for feature extraction
#----------------------------------------------------------------
p = 2.0  # value for p-leaders
mfa = mf.MFA(**utils.mf_params)
mfa.n_cumul = 2
mfa.p = p
mfa.q = np.linspace(-8, 8, 50)



mfa.analyze(data_normal)
Dq_normal = mfa.spectrum.Dq 
hq_normal = mfa.spectrum.hq


mfa.analyze(data_ictal)
Dq_ictal = mfa.spectrum.Dq 
hq_ictal = mfa.spectrum.hq


rcParams['figure.figsize'] = 12, 8

plt.figure()
plt.title('Multifractal Spectrum')
plt.plot(hq_normal, Dq_normal, 'b-', label='normal', linewidth = 2)
plt.plot(hq_ictal,  Dq_ictal,  'r-', label='seizure', linewidth = 2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('$\mathcal{D}(h)$')

plt.show()
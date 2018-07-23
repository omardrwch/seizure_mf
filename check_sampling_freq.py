from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import utils

subject = 'Dog_3'
interictal_files_idx  = np.arange(1, 10)
preictal_files_idx    = np.arange(1, 25)



for inter_file_idx in interictal_files_idx:
	data = utils.get_interictal_data(inter_file_idx, subject)
	s_freq = data['s_freq']
	duration = data['length_sec']
	seq    = data['sequence'] 
	print('s_freq = %f, duration = %f, sequence = %f'%(s_freq, duration, seq))

# for pre_file_idx in preictal_files_idx:
# 	data = utils.get_preictal_data(pre_file_idx, subject)
# 	s_freq = data['s_freq']
# 	duration = data['length_sec']
# 	seq    = data['sequence'] 
# 	print('s_freq = %f, duration = %f, sequence = %f'%(s_freq, duration, seq))


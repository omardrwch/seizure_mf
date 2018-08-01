from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import utils


test_data_loc = 'D:\\parietal'


subject = 'Dog_5'
interictal_files_idx  = np.arange(1, 10)
preictal_files_idx    = np.arange(1, 10)
test_files_idx    = np.arange(1, 30)



# for inter_file_idx in interictal_files_idx:
# 	data = utils.get_interictal_data(inter_file_idx, subject)
# 	s_freq = data['s_freq']
# 	duration = data['length_sec']
# 	seq    = data['sequence'] 
# 	print('s_freq = %f, duration = %f, sequence = %f, data shape 1 = %d'%(s_freq, duration, seq, data['data'].shape[0]))


# print("----------------")

# for pre_file_idx in preictal_files_idx:
# 	data = utils.get_preictal_data(pre_file_idx, subject)
# 	s_freq = data['s_freq']
# 	duration = data['length_sec']
# 	seq    = data['sequence'] 
# 	print('s_freq = %f, duration = %f, sequence = %f, data shape 1 = %d'%(s_freq, duration, seq, data['data'].shape[0]))



for test_file_idx in test_files_idx:
	data = utils.get_test_data(test_file_idx, subject, test_data_loc)
	s_freq = data['s_freq']
	duration = data['length_sec']
	# seq    = data['sequence'] 
	print('s_freq = %f, duration = %f, data shape 1 = %d'%(s_freq, duration, data['data'].shape[0]))



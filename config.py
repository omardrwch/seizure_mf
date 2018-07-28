import numpy as np


#--------------------------------------------------------------------
# Information about subjects and files
#--------------------------------------------------------------------

subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']

info = {}

info['Dog_1'] = { 's_freq': 399.609756,
                  'n_channels': 16,
                  'length_sec': 600,
                  'interictal_files_idx': np.arange(1, 481),
                  'preictal_files_idx'  : np.arange(1, 25),
                  'test_files_idx'  : np.arange(1, 503)
                }


info['Dog_2'] = { 's_freq': 399.609756,
                  'n_channels': 16,
                  'length_sec': 600,
                  'interictal_files_idx': np.arange(1, 501),
                  'preictal_files_idx'  : np.arange(1, 43),
                  'test_files_idx'  : np.arange(1, 1001)
                }

info['Dog_3'] = { 's_freq': 399.609756,
                  'n_channels': 16,
                  'length_sec': 600,
                  'interictal_files_idx': np.arange(1, 1441),
                  'preictal_files_idx'  : np.arange(1, 72),
                  'test_files_idx'  : np.arange(1, 908)
                }

info['Dog_4'] = { 's_freq': 399.609756,
                  'n_channels': 16,
                  'length_sec': 600,
                  'interictal_files_idx': np.arange(1, 805),
                  'preictal_files_idx'  : np.arange(1, 98),
                  'test_files_idx'  : np.arange(1, 991)
                }

info['Dog_5'] = { 's_freq': 399.609756,
                  'n_channels': 15,
                  'length_sec': 600,
                  'interictal_files_idx': np.arange(1, 451),
                  'preictal_files_idx'  : np.arange(1, 31),
                  'test_files_idx'  : np.arange(1, 192)
                }

info['Patient_1'] = { 's_freq': 5000.0,
                      'n_channels': 15,
                      'length_sec': 600,
                      'interictal_files_idx': np.arange(1, 51),
                      'preictal_files_idx'  : np.arange(1, 19),
                      'test_files_idx'  : np.arange(1, 196)
                    }

info['Patient_2'] = { 's_freq': 5000.0,
                      'n_channels': 24,
                      'length_sec': 600,
                      'interictal_files_idx': np.arange(1, 43),
                      'preictal_files_idx'  : np.arange(1, 19),
                      'test_files_idx'  : np.arange(1, 151)
                    }

#--------------------------------------------------------------------
# Parameters for MF analysis
#--------------------------------------------------------------------

# p_list = [(0.25,   0), 
#           (1.0,    1), 
#           (2.0,    2), 
#           (4.0,    3), 
#           (np.inf, 4)]  # (value_of_p, index_of_p)
p_list = [(0.25,   0), 
          (2.0,    2), 
          (np.inf, 4)]  # (value_of_p, index_of_p)


mf_params = {}

mf_params['Dog_1']  =  { 'wt_name': 'db4',
                        'j1': 8,
                        'j2': 11,
                        'q' : np.array([2]),
                        'n_cumul': 4,
                        'gamint' : 2,
                        'wtype'  : 0,
                        'verbose': 1
                        }
mf_params['Dog_2']  =  { 'wt_name': 'db4',
                        'j1': 8,
                        'j2': 11,
                        'q' : np.array([2]),
                        'n_cumul': 4,
                        'gamint' : 2,
                        'wtype'  : 0,
                        'verbose': 1
                        }
mf_params['Dog_3']  =  { 'wt_name': 'db4',
                        'j1': 8,
                        'j2': 11,
                        'q' : np.array([2]),
                        'n_cumul': 4,
                        'gamint' : 2,
                        'wtype'  : 0,
                        'verbose': 1
                        }

mf_params['Dog_4']  =  { 'wt_name': 'db4',
                        'j1': 8,
                        'j2': 11,
                        'q' : np.array([2]),
                        'n_cumul': 4,
                        'gamint' : 2,
                        'wtype'  : 0,
                        'verbose': 1
                        }

mf_params['Dog_5']  =  { 'wt_name': 'db4',
                        'j1': 8,
                        'j2': 11,
                        'q' : np.array([2]),
                        'n_cumul': 4,
                        'gamint' : 2,
                        'wtype'  : 0,
                        'verbose': 1
                        }

mf_params['Patient_1']  =  { 'wt_name': 'db4',
                          'j1': 11,
                          'j2': 15,
                          'q' : np.array([2]),
                          'n_cumul': 4,
                          'gamint' : 2,
                          'wtype'  : 0,
                          'verbose': 1
                          }

mf_params['Patient_2']  =  { 'wt_name': 'db4',
                          'j1': 11,
                          'j2': 15,
                          'q' : np.array([2]),
                          'n_cumul': 4,
                          'gamint' : 2,
                          'wtype'  : 0,
                          'verbose': 1
                          }
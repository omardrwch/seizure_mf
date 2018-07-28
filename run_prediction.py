"""
Classify the test set.
"""

import numpy as np
import utils
import utils_classif
import config as cfg
import pandas as pd
import os


test_data_loc = 'D:\\parietal'

#-----------------------------------------------------------
# Classification parameters for each subject
#-----------------------------------------------------------
params = {}


params['Dog_1'] = { 'options':   { 'p_idx':0,
          					       'features': ['c1'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }


params['Dog_2'] = { 'options':   { 'p_idx':0,
          					       'features': ['c1'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }

params['Dog_3'] = { 'options':   { 'p_idx':0,
          					       'features': ['c1'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }


params['Dog_4'] = { 'options':   { 'p_idx':0,
          					       'features': ['c1'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }


params['Dog_5'] = { 'options':   { 'p_idx':4,
          					       'features': ['c1'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }

params['Patient_1'] = { 'options':   { 'p_idx':0,
          					       'features': ['c1'],
                                   'clip_c2': False},
                        'classifier_name': 'random_forest'
                      }


params['Patient_2'] = { 'options':   { 'p_idx':0,
          					       'features': ['c1'],
                                   'clip_c2': False},
                        'classifier_name': 'random_forest'
                      }




# params['Dog_1'] = { 'options':   { 'p_idx':0,
#           					       'features': ['c1', 'c2'],
#                                    'clip_c2': False},
#                     'classifier_name': 'random_forest'
#                   }


# params['Dog_2'] = { 'options':   { 'p_idx':0,
#           					       'features': ['c1', 'c2'],
#                                    'clip_c2': False},
#                     'classifier_name': 'random_forest'
#                   }

# params['Dog_3'] = { 'options':   { 'p_idx':0,
#           					       'features': ['c1', 'c2'],
#                                    'clip_c2': False},
#                     'classifier_name': 'random_forest'
#                   }


# params['Dog_4'] = { 'options':   { 'p_idx':0,
#           					       'features': ['hurst', 'c2'],
#                                    'clip_c2': False},
#                     'classifier_name': 'random_forest'
#                   }


# params['Dog_5'] = { 'options':   { 'p_idx':4,
#           					       'features': ['c1', 'c2', 'c3'],
#                                    'clip_c2': False},
#                     'classifier_name': 'random_forest'
#                   }

# params['Patient_1'] = { 'options':   { 'p_idx':0,
#           					       'features': ['c1', 'c2'],
#                                    'clip_c2': False},
#                         'classifier_name': 'random_forest'
#                       }


# params['Patient_2'] = { 'options':   { 'p_idx':0,
#           					       'features': ['c1', 'c2'],
#                                    'clip_c2': False},
#                         'classifier_name': 'random_forest'
#                       }


#-----------------------------------------------------------
# Run
#-----------------------------------------------------------

data_dict = {'clip': [], 'preictal': []}
df = pd.DataFrame.from_dict(data_dict)

for subject in cfg.subjects:
	print(subject, params[subject])

	clf, fit_params = utils_classif.get_classifier(params[subject]['classifier_name'], 
							   None, None)

	X_train, y_train, _, _ = utils.load_classif_data(subject, params[subject]['options'])

	X_test  = utils.load_classif_test_data(subject, params[subject]['options'], test_data_loc)


	# train
	clf.fit(X_train, y_train, **fit_params)

	# predict
	y_scores = clf.predict_proba(X_test)[:, 1]


	# Save
	clips = []
	for test_file_idx in range( X_test.shape[0] ):
		clip = '%s_test_segment_%s.mat'%(subject, str(test_file_idx+1).zfill(4))
		clips.append(clip)

	data_dict = {'clip': clips, 'preictal': y_scores}

	df = pd.concat( [df, pd.DataFrame.from_dict(data_dict)], ignore_index  = True)

	outfilename = os.path.join('submissions', 'two.csv')

	df.to_csv(outfilename, index=False)
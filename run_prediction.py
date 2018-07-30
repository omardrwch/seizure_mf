"""
Classify the test set.
"""

import numpy as np
import utils
import utils_classif
import config as cfg
import pandas as pd
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score


random_state = 123

test_data_loc = 'D:\\parietal'

RUN_ON_TEST = False  # If true, generate file for submission
                     # If false, run on all training data
                    
#-----------------------------------------------------------
# Classification parameters for each subject
#-----------------------------------------------------------
params = {}



params['Dog_1'] = { 'options':   { 'p_idx':0,
                                     'features': ['c1', 'c2'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }


params['Dog_2'] = { 'options':   { 'p_idx':0,
                                     'features': ['c1', 'c2'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }

params['Dog_3'] = { 'options':   { 'p_idx':0,
                                     'features': ['c1', 'c2'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }


params['Dog_4'] = { 'options':   { 'p_idx':0,
                                     'features': ['hurst', 'c2'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }


params['Dog_5'] = { 'options':   { 'p_idx':4,
                                     'features': ['c1', 'c2', 'c3'],
                                   'clip_c2': False},
                    'classifier_name': 'random_forest'
                  }

params['Patient_1'] = { 'options':   { 'p_idx':0,
                                     'features': ['c1', 'c2'],
                                   'clip_c2': False},
                        'classifier_name': 'random_forest'
                      }


params['Patient_2'] = { 'options':   { 'p_idx':0,
                                     'features': ['c1', 'c2'],
                                   'clip_c2': False},
                        'classifier_name': 'random_forest'
                      }


#-----------------------------------------------------------
# Run on test
#-----------------------------------------------------------

if RUN_ON_TEST:
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


#-----------------------------------------------------------
# Run on train
#-----------------------------------------------------------

else:
    N_SPLITS = 5

    cv  = StratifiedShuffleSplit(n_splits     = N_SPLITS, 
                                 test_size    = 0.3, 
                                 random_state = random_state )


    # Create CV iterators
    generators = {}
    for subject in cfg.subjects:
        X, y, _, _ = utils.load_classif_data(subject, params[subject]['options'])
        gen = cv.split(X, y)
        generators[subject] = gen


    # Run 
    aucs     = []
    global_auc   = []

    for split in range(N_SPLITS):
        auc_list = []
        y_test_all   = []
        y_scores_all = []

        print("------------- split ", split)
        for subject in cfg.subjects:
            print(subject)
            X, y, _, _ = utils.load_classif_data(subject, params[subject]['options'])

            train_index, test_index = next(generators[subject])


            X_train = X[train_index, :]
            X_test  = X[test_index,  :]

            y_train = y[train_index]
            y_test  = y[test_index]

            clf, fit_params = utils_classif.get_classifier(params[subject]['classifier_name'], 
                                               None, None)


            # train
            clf.fit(X_train, y_train, **fit_params)

            # predict
            y_scores = clf.predict_proba(X_test)[:, 1]


            y_test_all += list(y_test)
            y_scores_all += list(y_scores)
            auc_list.append(roc_auc_score(y_test, y_scores))

        aucs.append(auc_list)
        global_auc.append(roc_auc_score(y_test_all, y_scores_all))


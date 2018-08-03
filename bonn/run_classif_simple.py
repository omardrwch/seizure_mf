import numpy as np
import os
import mfanalysis as mf
import matplotlib.pyplot as plt
import utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from mne_features.feature_extraction import FeatureExtractor

N_SPLITS = 3
scoring = 'roc_auc'

#----------------------------------------------------------------
# Select sets and labels
#----------------------------------------------------------------
sets = ['A', 'B', 'C', 'D', 'E']
labels = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 1}
files_per_set = 100

#----------------------------------------------------------------
# Create MF object for feature extraction
#----------------------------------------------------------------
p = 2.0  # value for p-leaders
mfa = mf.MFA(**utils.mf_params)
mfa.n_cumul = 1
mfa.p = p

#----------------------------------------------------------------
# Feature extraction
#----------------------------------------------------------------
n_samples = files_per_set*len(sets)
n_features = mfa.n_cumul

all_data = np.zeros((n_samples, 1, 4097))


X = np.zeros((n_samples, n_features))
y = -1*np.ones(n_samples)
count = 0
for set_name in sets:
    for index in range(1, files_per_set+1):
        data = utils.read_file(set_name, index)

        mfa.analyze(data)
        X[count, :] = mfa.cumulants.log_cumulants
        y[count]    = labels[set_name]


        all_data[count, 0, :] = data

        count += 1


#----------------------------------------------------------------
# Run classification
#----------------------------------------------------------------

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42)

scores = cross_val_score(clf, X, y, cv=skf, scoring = scoring)
print('Cross-validation ' + scoring + ' score = %1.3f (+/- %1.5f)' % (np.mean(scores), np.std(scores)))


# See feature importances
clf.fit(X,y)
print("Feature importances = ", clf.feature_importances_)

#----------------------------------------------------------------
# Run classification baseline
#----------------------------------------------------------------
selected_funcs = ['line_length', 'kurtosis', 'ptp_amp', 'skewness']

pipe = Pipeline([('fe', FeatureExtractor(sfreq=utils.s_freq,
                                         selected_funcs=selected_funcs)),
                 ('clf', RandomForestClassifier(n_estimators=100,
                                                max_depth=4, random_state=42))])

scores = cross_val_score(pipe, all_data, y, cv=skf, scoring = scoring)
print('[baseline] Cross-validation ' + scoring + ' score = %1.3f (+/- %1.5f)' % (np.mean(scores), np.std(scores)))

#----------------------------------------------------------------
# Run classification combined
#----------------------------------------------------------------
fe = FeatureExtractor(sfreq=utils.s_freq, selected_funcs=selected_funcs)
X_baseline = fe.fit_transform(all_data)


all_X = np.hstack((X, X_baseline))

scores = cross_val_score(clf, all_X, y, cv=skf, scoring = scoring)
print('[combined] Cross-validation ' + scoring + ' score = %1.3f (+/- %1.5f)' % (np.mean(scores), np.std(scores)))

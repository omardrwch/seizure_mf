"""
Run classification+cross-validation on training data.
"""

import numpy as np
import utils
import utils_classif
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from scipy import interp


random_state = 123
n_jobs       = 10

#-----------------------------------------------------------
# Load features
#-----------------------------------------------------------
subject = 'Patient_1'
options = {'p_idx':0,
           'features': ['c1', 'c2'],
           'clip_c2': False}
if 'Dog' in subject:
    fs = 399.609756
else:
    fs  = 5000.0


freq1 = 0.2
freq2 = 2

j1, j2 = utils.get_scales(fs, freq1, freq2)

X, y, sequence_interictal, sequence_preictal = utils.load_classif_data_2(subject, options, j1, j2)
# X, y, sequence_interictal, sequence_preictal = utils.load_classif_data(subject, options)



# X = X[:, [3,6,14,15]]


# groups = np.arange(X.shape[0])
groups = np.hstack((sequence_interictal, 100 + sequence_preictal))


#-----------------------------------------------------------
# Classification parameters
#-----------------------------------------------------------
classifier_name = 'select_lda_kbest_lda'
train_sizes = [0.2, 0.3, 0.5, 0.7]
scoring     = ['roc_auc']
n_splits    = 20

#-----------------------------------------------------------
# Run
#-----------------------------------------------------------
train_sizes_abs, train_scores, test_scores, clf, fit_params, outer_cv = \
    utils_classif.my_learning_curve(classifier_name,
                                    X, y, groups, train_sizes, scoring,
                                    n_splits, random_state, n_jobs, return_clf_cv = True)


utils_classif.plot_learning_curve(train_sizes_abs, train_scores, test_scores,
                                  title='', ylim = [0.4, 1.0],
                                  fignum = None)
plt.draw()




# Feature importances
w, positive = utils_classif.get_feature_importances(clf, fit_params, X, y)

plt.figure()
plt.title('weights (abs val)')
plt.plot(w, 'o')
plt.draw()

print("!!!!!!! 'best' sensor = ", np.abs(w).argmax())


#-----------------------------------------------------------
# Debug
#-----------------------------------------------------------

auc_list = []

tprs = []
mean_fpr = np.linspace(0, 1, 100)


for train_index, test_index in outer_cv.split(X, y):
    X_train = X[train_index, :]
    X_test  = X[test_index,  :]

    y_train = y[train_index]
    y_test  = y[test_index]

    clf.fit(X_train, y_train, **fit_params)

    try:
        y_pred = clf.predict_proba(X_test)[:, 1]
    except:
        y_pred = clf.predict(X_test)


    auc_list.append(roc_auc_score(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0


plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(auc_list)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()



#-----------------------------------------------------------
# Visualize train/test
#-----------------------------------------------------------

def compare_distributions(data_train, data_test, title = '', fignum = None, colors = ['b', 'r'], labels = ['train', 'test']):
    """
    For each cortical region, plot mean and std for train and test data.

    data_train: shape (n_train, n_features)
    data_test:  shape (_test, n_features)
    """
    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)


    plt.title(title)
    plt.plot(np.arange(data_train.shape[1]), data_train.mean(axis=0), colors[0] + 'o-', label = labels[0])
    plt.plot(np.arange(data_train.shape[1]), data_test.mean(axis=0),  colors[1] + 'o-', label = labels[1])
    plt.xlabel('sensor')
    plt.ylabel('mean feature')
    plt.legend()


    plt.fill_between(np.arange(data_train.shape[1]),data_train.mean(axis=0) - data_train.std(axis=0),
                                data_train.mean(axis=0) + data_train.std(axis=0), alpha=0.25,
                             color=colors[0])
    plt.fill_between(np.arange(data_train.shape[1]),data_test.mean(axis=0) - data_test.std(axis=0),
                             data_test.mean(axis=0) + data_test.std(axis=0), alpha=0.25,
                             color=colors[1])


outer_cv.n_splits = 3

for train_index, test_index in outer_cv.split(X, y, groups = groups):
    train_0 = train_index[y[train_index]==0]
    train_1 = train_index[y[train_index]==1]

    test_0 = test_index[y[test_index]==0]
    test_1 = test_index[y[test_index]==1]


    X_train_0 = X[train_0, :]
    X_train_1 = X[train_1, :]

    X_test_0 = X[test_0, :]
    X_test_1 = X[test_1, :]

    compare_distributions(X_train_0, X_test_0, title = 'X interictal')
    compare_distributions(X_train_1, X_test_1, title = 'X preictal')


    compare_distributions(np.vstack((X_train_0,X_test_0)), np.vstack((X_train_1,X_test_1)) ,
                         title = 'both', fignum = 100,
                         labels = ['interictal', 'preictal'])

    plt.show()

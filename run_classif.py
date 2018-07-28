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
n_jobs       = 1

#-----------------------------------------------------------
# Load features
#-----------------------------------------------------------
subject = 'Patient_2'
options = {'p_idx':0,
           'features': ['c1', 'c2'],
           'clip_c2': False}

X, y, sequence_interictal, sequence_preictal = utils.load_classif_data(subject, options)

groups = np.arange(X.shape[0])

#-----------------------------------------------------------
# Classification parameters
#-----------------------------------------------------------
classifier_name = 'random_forest'
train_sizes = [0.3, 0.5, 0.7, 0.9]
scoring     = ['roc_auc']
n_splits    = 20


#-----------------------------------------------------------
# Run
#-----------------------------------------------------------
train_sizes_abs, train_scores, test_scores = \
    utils_classif.my_learning_curve(classifier_name, 
                                    X, y, groups, train_sizes, scoring, 
                                    n_splits, random_state, n_jobs)


utils_classif.plot_learning_curve(train_sizes_abs, train_scores, test_scores, 
                                  title='', ylim = [0.4, 1.0], 
                                  fignum = None)
plt.draw()



# cv  = StratifiedShuffleSplit(n_splits     = 20, 
#                              test_size    = 0.8, 
#                                    random_state = random_state )
# clf, fit_params = utils_classif.get_classifier(classifier_name, cv, groups)
# w, positive = utils_classif.get_feature_importances(clf, fit_params, X, y)



#-----------------------------------------------------------
# Debug
#-----------------------------------------------------------
np.random.seed(456)
from sklearn.svm import SVC

cv  = StratifiedShuffleSplit(n_splits     = 30, 
                             test_size    = 0.3, 
                             random_state = random_state )



auc_list = []

tprs = []
mean_fpr = np.linspace(0, 1, 100)


for train_index, test_index in cv.split(X, y):
    X_train = X[train_index, :]
    X_test  = X[test_index,  :]

    y_train = y[train_index]
    y_test  = y[test_index]

    # print(y_train)

    # clf = SVC(kernel='linear', C = 0.05, tol = 1e-8)
    clf = RandomForestClassifier(n_estimators=150)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]

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



# #-----------------------------------------------------------
# # Visualize train/test
# #-----------------------------------------------------------


# cv  = StratifiedShuffleSplit(n_splits     = 5, 
#                                    test_size    = 0.2, 
#                                    random_state = random_state )


# def compare_distributions(data_train, data_test, title = '', fignum = None, colors = ['b', 'r']):
#     """
#     For each cortical region, plot mean and std for train and test data.

#     data_train: shape (n_train, n_features)
#     data_test:  shape (_test, n_features)
#     """
#     if fignum is None:
#         plt.figure()
#     else:
#         plt.figure(fignum)


#     plt.title(title)
#     plt.plot(np.arange(data_train.shape[1]), data_train.mean(axis=0), colors[0] + 'o-', label = 'train')
#     plt.plot(np.arange(data_train.shape[1]), data_test.mean(axis=0),  colors[1] + 'o-', label = 'test')
#     plt.xlabel('sensor')
#     plt.ylabel('mean feature')
#     plt.legend()


#     plt.fill_between(np.arange(data_train.shape[1]),data_train.mean(axis=0) - data_train.std(axis=0),
#                                 data_train.mean(axis=0) + data_train.std(axis=0), alpha=0.25,
#                              color=colors[0])
#     plt.fill_between(np.arange(data_train.shape[1]),data_test.mean(axis=0) - data_test.std(axis=0),
#                              data_test.mean(axis=0) + data_test.std(axis=0), alpha=0.25,
#                              color=colors[1])


# for train_index, test_index in cv.split(X, y, groups = groups):
#     train_0 = train_index[y[train_index]==0]
#     train_1 = train_index[y[train_index]==1]

#     test_0 = test_index[y[test_index]==0]
#     test_1 = test_index[y[test_index]==1]


#     X_train_0 = X[train_0, :]
#     X_train_1 = X[train_1, :]

#     X_test_0 = X[test_0, :]
#     X_test_1 = X[test_1, :]

#     compare_distributions(X_train_0, X_test_0, title = 'X interictal')
#     compare_distributions(X_train_1, X_test_1, title = 'X preictal')


#     compare_distributions(X_train_0, X_test_0, title = 'both', fignum = 100)
#     compare_distributions(X_train_1, X_test_1, title = 'both',  fignum = 100, colors = ['m', 'y'])

#     plt.show()
    # X_train_0 = [train_index, :]


    # X_test = [train_index, :]

    # print(y[train_index].sum())
    # print(y[test_index].sum())

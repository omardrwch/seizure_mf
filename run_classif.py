import numpy as np
import utils
import utils_classif
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

random_state = 123
n_jobs       = 3

#-----------------------------------------------------------
# Load features
#-----------------------------------------------------------
subject = 'Dog_3'
options = {'p_idx':0,
           'features': ['hurst', 'c2', 'c3', 'c4']}

X, y, sequence_interictal, sequence_preictal = utils.load_classif_data(subject, options)

groups = np.arange(X.shape[0])

#-----------------------------------------------------------
# Classification parameters
#-----------------------------------------------------------
classifier_name = 'linear_svm_scaled'
train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
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
plt.show()



cv  = StratifiedShuffleSplit(n_splits     = 20, 
                             test_size    = 0.8, 
                                   random_state = random_state )
clf, fit_params = utils_classif.get_classifier(classifier_name, cv, groups)
w, positive = utils_classif.get_feature_importances(clf, fit_params, X, y)



# #-----------------------------------------------------------
# # Debug
# #-----------------------------------------------------------
# np.random.seed(456)
# from sklearn.svm import SVC

# cv  = StratifiedShuffleSplit(n_splits     = 20, 
#                              test_size    = 0.9, 
#                                    random_state = random_state )


# err_list = []

# for train_index, test_index in cv.split(X, y):
#     X_train = X[train_index, :]
#     X_test  = X[test_index,  :]

#     y_train = y[train_index]
#     y_test  = y[test_index]

#     print(y_train)

#     clf = SVC(kernel='linear', C = 0.05, tol = 1e-8)
#     # clf = RandomForestClassifier(n_estimators = 50)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)

#     err_abs = (y_pred != y_test).sum()

#     err = err_abs/len(y_pred)

#     err_list.append(err)
#     # print("err = ", err)

# err_list = np.array(err_list)
# acc_test = 1 - err_list

# print("-----------------")
# print("acc = %f +- %f"%(acc_test.mean(), acc_test.std()))




# #-----------------------------------------------------------
# # Visualize train/test
# #-----------------------------------------------------------
# 

# cv  = StratifiedShuffleSplit(n_splits     = 5, 
#                                    test_size    = 0.8, 
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
#     plt.plot(np.arange(16), data_train.mean(axis=0), colors[0] + 'o-', label = 'train')
#     plt.plot(np.arange(16), data_test.mean(axis=0),  colors[1] + 'o-', label = 'test')
#     plt.xlabel('sensor')
#     plt.ylabel('mean feature')
#     plt.legend()


#     plt.fill_between(np.arange(16),data_train.mean(axis=0) - data_train.std(axis=0),
#                                 data_train.mean(axis=0) + data_train.std(axis=0), alpha=0.25,
#                              color=colors[0])
#     plt.fill_between(np.arange(16),data_test.mean(axis=0) - data_test.std(axis=0),
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
#     # X_train_0 = [train_index, :]


#     # X_test = [train_index, :]

#     # print(y[train_index].sum())
#     # print(y[test_index].sum())

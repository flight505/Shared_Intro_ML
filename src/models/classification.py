import operator
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import torch
from loader import reader
#from train_neural_net import Net
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import tree


def KNN(neighbors, X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf = clf.fit(X_train, y_train)
    return clf, clf.score(X_test, y_test)

def LogReg(lamb, X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver = 'lbfgs', C=1/lamb)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, metrics.accuracy_score(y_test, y_pred)

def DecisionTrees(X_train, y_train, X_test, y_test):
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, metrics.accuracy_score(y_test, y_pred)

def baseline(y_train, y_test):
    most_common = Counter(y_train).most_common(1)[0][0]
    preds = [most_common]*len(y_train)
    return accuracy_score(y_train, preds)

task = 'classification'
x_train, y_train, x_unseen, y_unseen = reader().get_all(task)


knn_candidate_parameter = [1, 3, 5, 7, 9, 11]
lr_candidate_parameter = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
dt_candidate_parameter = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] # fictional

outer_scores_knn, outer_scores_lr, outer_scores_dt = [], [], []
outer_fold_hp_knn, outer_fold_hp_lr, outer_fold_hp_dt = [], [], []
outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for outer_fold, (train_index_outer, test_index_outer) in enumerate(outer.split(x_train, y_train)):
    x_train_outer, x_test_outer = x_train[train_index_outer], x_train[test_index_outer]
    y_train_outer, y_test_outer = y_train[train_index_outer], y_train[test_index_outer]

    param_scores_knn = {param: [] for param in knn_candidate_parameter}
    param_scores_lr = {param: [] for param in lr_candidate_parameter}
    param_scores_dt = {param: [] for param in dt_candidate_parameter}

    inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for idx in range(len((knn_candidate_parameter))):
        for inner_fold, (train_index_inner, test_index_inner) in enumerate(inner.split(x_train_outer, y_train_outer)):
            x_train_inner, x_test_inner = x_train_outer[train_index_inner], x_train_outer[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

            knn_model, E_val = KNN(knn_candidate_parameter[idx], x_train_inner, y_train_inner, x_test_inner, y_test_inner)
            param_scores_knn[knn_candidate_parameter[idx]].append(E_val)

            logreg_model, E_val = LogReg(lr_candidate_parameter[idx], x_train_inner, y_train_inner, x_test_inner, y_test_inner)
            param_scores_lr[lr_candidate_parameter[idx]].append(E_val)

            dt_model, E_val = DecisionTrees(x_train_inner, y_train_inner, x_test_inner, y_test_inner)
            param_scores_dt[dt_candidate_parameter[idx]].append(E_val)

            bas_score = baseline(y_train_inner, y_test_inner)


    for k in param_scores_knn.keys():
        param_scores_knn[k] = np.mean(param_scores_knn[k])

    for k in param_scores_lr.keys():
        param_scores_lr[k] = np.mean(param_scores_lr[k])

    for k in param_scores_dt.keys():
        param_scores_dt[k] = np.mean(param_scores_dt[k])
    
    opt_knn = max(param_scores_knn.items(), key=operator.itemgetter(1))[0]
    opt_lr = max(param_scores_lr.items(), key=operator.itemgetter(1))[0]
    opt_dt = max(param_scores_dt.items(), key=operator.itemgetter(1))[0]
    outer_fold_hp_knn.append(opt_knn)
    outer_fold_hp_lr.append(opt_lr)
    outer_fold_hp_dt.append(opt_dt)
            
    clf, E_test = KNN(opt_knn, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
    outer_scores_knn.append(round(E_test,3))

    clf, E_test = LogReg(opt_lr, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
    outer_scores_lr.append(round(E_test,3))

    clf, E_test = DecisionTrees(x_train_outer, y_train_outer, x_test_outer, y_test_outer)
    outer_scores_dt.append(round(E_test,3))

# print outer_scores_* and outer_fold_hp_* for table in project description

print(outer_scores_knn)
print(outer_fold_hp_knn)
print(max(outer_scores_knn), outer_fold_hp_knn[np.argmax(outer_scores_knn)])
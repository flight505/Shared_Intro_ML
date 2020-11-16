import operator
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import torch
from loader import reader
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


task = 'classification'
x_train, y_train, x_unseen, y_unseen = reader().get_all(task)

columns = x_train.columns

X = x_train.values
y = y_train.values
x_unseen = x_unseen.values


def KNN(neighbors, X_train, y_train, X_test, y_test):
    clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=neighbors), random_state=7)
    clf.fit(X_train, y_train)
    return clf, clf.score(X_test, y_test)

def removal(X, x_unseen):
    candidates = []
    for col_idx in range(X.shape[1]):
        X_selection_train = np.delete(X, col_idx, axis=1)
        X_selection_unseen = np.delete(x_unseen, col_idx, axis=1)
        candidates.append(KNN(1, X_selection_train, y, X_selection_unseen, y_unseen)[1])
    return candidates

# KNN
initial_score = KNN(1, X, y, x_unseen, y_unseen)[1]
X_temp = X
x_unseen_temp = x_unseen

removed_columns = []
iter = X.shape[1]
scores = {}
scores[iter] = initial_score
for i in range(iter-1):
    removed_features = iter-i-1
    candidates = removal(X_temp, x_unseen_temp)
    best_score = max(candidates)
    scores[removed_features] = best_score
    X_temp = np.delete(X_temp, np.argmax(candidates), axis=1)
    x_unseen_temp = np.delete(x_unseen_temp, np.argmax(candidates), axis=1)
    removed_columns.append(columns[np.argmax(candidates)])
    columns = np.delete(columns, np.argmax(candidates))

best_score_overall = max(scores.items(), key=operator.itemgetter(1))[0]
optimal_columns = removed_columns[-best_score_overall+1:]
optimal_columns.append(columns[0])
print(optimal_columns)

idx = sorted([x_train.columns.tolist().index(opt_col) for opt_col in optimal_columns])
optimal_X_train = X[:, idx]
optimal_X_unseen = x_unseen[:, idx]

print(KNN(1, optimal_X_train, y_train, optimal_X_unseen, y_unseen)[1])
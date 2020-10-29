import operator
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from loader import reader

def KNN(neighbors, X_train, y_train, X_test, y_test):
    clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=neighbors))
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

task = 'classification'
x_train, y_train, x_test, y_test = reader().get_all(task)

X = x_train.values
y = y_train.values

#X = np.random.rand(10, 3)
#y = np.array([1,3,4,1,2,3,3,4,2,1])

outer_scores = []
outer = KFold(n_splits=10, shuffle=True, random_state=42)
for outer_fold, (train_index_outer, test_index_outer) in enumerate(outer.split(X)):
    print("Outer fold: "+str(outer_fold))
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

    inner_mean_scores = []

    candidate_parameter = [1, 2, 3, 4, 5]
    for param in candidate_parameter:

        inner_scores = []

        inner = KFold(n_splits=10, shuffle=True, random_state=42)
        for inner_fold, (train_index_inner, test_index_inner) in enumerate(inner.split(X_train_outer)):
            print("Inner fold: "+str(inner_fold))
            X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

            inner_scores.append(KNN(param, X_train_inner, y_train_inner, X_test_inner, y_test_inner))

        inner_mean_scores.append(np.mean(inner_scores))

    index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))

    print('Best parameter of %i fold: %i' % (outer_fold + 1, candidate_parameter[index]))

    # look from here downwards - might be wrong
    outer_scores.append(KNN(candidate_parameter[index], X_train_outer, y_train_outer, X_test_outer, y_test_outer))

print('Unbiased prediction error: %.3f' % (np.mean(outer_scores)))

clf3 = BaggingClassifier(KNeighborsClassifier(n_neighbors=candidate_parameter[index]))
clf3.fit(X, y)

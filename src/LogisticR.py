
'''
### Logistic Regression

Classification: In this part of the report you are to solve a relevant
classification problem for your data and statistically evaluate your
result. The tasks will closely mirror what you just did in the last
section. The three methods we will compare is a baseline, logistic
regression, and one of the other four methods from below (referred to
as method 2).

Logistic regression for classification. Once more, we can use a
regularization parameter λ ≥ 0 to control complexity

'''
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    auc,
    roc_curve,
)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.loader import reader

"""
using Marcos loader function to get the split data
if using Marcos remove train_test_split & StandardScaler - already done
"""

X_train, y_train, X_test, y_test = reader().get_all('classification')


def LogReg(lamb, X_train, y_train, X_test, y_test):
    """
    Super basic Logistic Regression on the data
    """

    model = LogisticRegression(solver='liblinear', C=1/lamb)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')


LogReg(X_train, y_train, X_test, y_test)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

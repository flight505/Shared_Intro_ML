import operator
import numpy as np
import random
from collections import Counter
import pandas as pd
from imblearn.over_sampling import SMOTENC
import scipy.stats as st
import scipy.stats

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

from statsmodels.stats.contingency_tables import mcnemar

from loader import reader


# creating class for feed forward neural network
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()  
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(hidden_dim, input_dim)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(hidden_dim), 0))
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(output_dim, hidden_dim)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(output_dim), 0))
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2, self.b_2)
        return x

# KNN algorithm, where the number of neighbors is provided as argument together with the data to train and test on
# returns predictions and error
def KNN(neighbors, X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, 1-accuracy_score(y_test, y_pred)

# Logistic Regression algorithm, where lambda is provided as argument together with the data to train and test on
# returns predictions and error
def LogReg(lamb, X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver = 'lbfgs', C=1/lamb, max_iter=400)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, 1-accuracy_score(y_test, y_pred)

# same as above but returns also the coefficient for each feature 
def LogReg_coef(lamb, X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver = 'lbfgs', C=1/lamb, max_iter=400)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, 1-accuracy_score(y_test, y_pred), model.coef_, model.intercept_

# Baseline prediction, takes targets of train and target of test set. Finds most common class in train
# and creates a prediction array of same length of test set.
# returns predictions and error
def baseline(y_train, y_test):
    most_common = Counter(y_train).most_common(1)[0][0]
    preds = [most_common]*len(y_test)
    return preds, 1-accuracy_score(y_test, preds)

# feed forward neural network algorithm.
# subtracts 1 from targets because pytorch wants target to range from 0 to number of classes,
# while our dataset starts from 1. Then, we convert np array to pytorch tensor
# and store dimensionalities in variables, finally initialize an instance of the Net class defined above.
# we also defined loss function for classification and loss function optimizer.
# we then train for 500 iterations with batches of 32 elements sampled from train set.
# one the weights have been optimized we predict on the test set and evaluate performance
# returns predictions and error
def ANN(h, X_train, y_train, X_test, y_test):
    y_train = y_train-1
    y_test = y_test-1
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    input_dim = X_train.shape[1]
    output_dim = np.unique(y_train).shape[0]
    ann_model = Net(input_dim, h, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ann_model.parameters(), lr=0.01)
    X_train = X_train.view(-1, input_dim).requires_grad_()
    batch_size = 32
    for epoch in range(500):
        optimizer.zero_grad()
        batch_idx = random.sample(range(0, X_train.shape[0]), batch_size) # Clear gradients w.r.t. parameters
        outputs = ann_model(X_train[batch_idx].float()) # Forward pass to get output/logits
        loss = criterion(outputs, y_train[batch_idx])
        loss.backward()
        optimizer.step()
    X_test = X_test.view(-1, input_dim).requires_grad_()
    outputs = ann_model(X_test.float())
    _, y_pred = torch.max(outputs.data, 1)
    return y_pred.detach().numpy()+1, 1-accuracy_score(y_test, y_pred)

# performs mcnemar test. takes predictions of two models as well as the ground truth vector
# follows algorithm explained in the book to create contingency table
# then performs test using statsmodel library
# returns pvalue and "response"
def perform_stat_test(y_true, model1, model2, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = model1 - y_true == 0
    c2 = model2 - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p
        


if __name__ == "__main__":
    
    test_run = False # set it true to run the digits classification task, false to run our HCV staging classification task
                    # done for testing purposes
    if test_run:
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        n_samples = X_digits.shape[0]
        x_train = X_digits[:int(.7 * n_samples)]
        y_train = y_digits[:int(.7 * n_samples)]+1
        x_unseen = X_digits[int(.7 * n_samples):]
        y_unseen = y_digits[int(.7 * n_samples):]+1
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_unseen = scaler.transform(x_unseen)
    else:
        task = 'classification'
        x_train, y_train, x_unseen, y_unseen = reader().get_all(task)

    x_train = np.append(x_train, x_unseen, 0)
    y_train = np.append(y_train, y_unseen, 0)
    """
    # performing data balancing using SMOTE
    aug_x, aug_y = [],[]

    oversample = SMOTENC(sampling_strategy='all', categorical_features=[16,17,18,19,20,21,22,23])
    outer = KFold(n_splits=10, shuffle=True, random_state=42)
    for outer_fold, (train_index_outer, test_index_outer) in enumerate(outer.split(x_train, y_train)):
        print(f"Performing {outer_fold+1}/10 SMOTE...")
        x_train_outer, x_test_outer = x_train[train_index_outer], x_train[test_index_outer]
        y_train_outer, y_test_outer = y_train[train_index_outer], y_train[test_index_outer]

        x_train_outer, y_train_outer = oversample.fit_resample(x_train_outer, y_train_outer)
        x_comb = np.append(x_train_outer,x_test_outer, 0)
        y_comb = np.append(y_train_outer,y_test_outer, 0)
        aug_x.extend(x_comb)
        aug_y.extend(y_comb)

    x_train = np.array(aug_x)
    y_train = np.array(aug_y)
    """

    # we defined candidate hyperparameters for each model - they need to have same dimensionality
    knn_candidate_parameter = [1, 3, 5, 7, 9, 11, 13, 15]
    lr_candidate_parameter = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 5.0, 25.0]
    ann_candidate_parameter = [1, 8, 16, 32, 64, 128, 256, 512]
    #ann_candidate_parameter = [512]*8

    # data structure initialization
    outer_scores_knn, outer_scores_lr, outer_scores_ann, outer_scores_bas = [], [], [], []
    outer_fold_hp_knn, outer_fold_hp_lr, outer_fold_hp_ann = [], [], []
    knn_preds_all, logreg_preds_all, ann_preds_all, bas_preds_all, y_true_all = [],[],[],[],[]

    # divide in K1=10 folds that will be used for estimating generalization error
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for outer_fold, (train_index_outer, test_index_outer) in enumerate(outer.split(x_train, y_train)):
        print(f"Running {outer_fold+1}/10 fold...")
        x_train_outer, x_test_outer = x_train[train_index_outer], x_train[test_index_outer]
        y_train_outer, y_test_outer = y_train[train_index_outer], y_train[test_index_outer]

        # initialize data structures to store performances for each candidate hyper parameters in inner loop
        param_scores_knn = {param: [] for param in knn_candidate_parameter}
        param_scores_lr = {param: [] for param in lr_candidate_parameter}
        param_scores_ann = {param: [] for param in ann_candidate_parameter}
        
        #divide in K2=10 folds used for finding optimal hyperparameters for each outer fold
        inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for idx in range(len((knn_candidate_parameter))):
            for inner_fold, (train_index_inner, test_index_inner) in enumerate(inner.split(x_train_outer, y_train_outer)):
                x_train_inner, x_test_inner = x_train_outer[train_index_inner], x_train_outer[test_index_inner]
                y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

                # for each inner fold and each candidate hyperparameter we run the models via the function defined above
                # we also store the error in the structures defined for each outer loop
                _, E_val = KNN(knn_candidate_parameter[idx], x_train_inner, y_train_inner, x_test_inner, y_test_inner)
                param_scores_knn[knn_candidate_parameter[idx]].append(E_val)

                _, E_val = LogReg(lr_candidate_parameter[idx], x_train_inner, y_train_inner, x_test_inner, y_test_inner)
                param_scores_lr[lr_candidate_parameter[idx]].append(E_val)

                _, E_val = ANN(ann_candidate_parameter[idx], x_train_inner, y_train_inner, x_test_inner, y_test_inner)
                param_scores_ann[ann_candidate_parameter[idx]].append(E_val)


        # we average the error obtained in the inner loop in order to have one measure for each outer fold
        for k in param_scores_knn.keys():
            param_scores_knn[k] = np.mean(param_scores_knn[k])

        for k in param_scores_lr.keys():
            param_scores_lr[k] = np.mean(param_scores_lr[k])

        for k in param_scores_ann.keys():
            param_scores_ann[k] = np.mean(param_scores_ann[k])
        
        # we then find for what hyperparameter we obtained the lower error on average
        # and we store each outer fold optimum hyperparamers in a list
        opt_knn = min(param_scores_knn.items(), key=operator.itemgetter(1))[0]
        opt_lr = min(param_scores_lr.items(), key=operator.itemgetter(1))[0]
        opt_ann = min(param_scores_ann.items(), key=operator.itemgetter(1))[0]
        outer_fold_hp_knn.append(opt_knn)
        outer_fold_hp_lr.append(opt_lr)
        outer_fold_hp_ann.append(opt_ann)
                
        # we run once more the model on the whole outer train and test set with optimum hyperparameter
        # we store errors and predictions - the latter will be useful for mcnamer test
        knn_preds, E_test = KNN(opt_knn, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_knn.append(round(E_test,3))
        knn_preds_all.extend(knn_preds)

        logreg_preds, E_test = LogReg(opt_lr, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_lr.append(round(E_test,3))
        logreg_preds_all.extend(logreg_preds)

        ann_preds, E_test = ANN(opt_ann, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_ann.append(round(E_test,3))
        ann_preds_all.extend(ann_preds)

        base_preds, E_test = baseline(y_train_outer, y_test_outer)
        outer_scores_bas.append(round(E_test,3))
        bas_preds_all.extend(base_preds)

        y_true_all.extend(y_test_outer)

    # generating table as required in project description
    table = pd.DataFrame()
    table['outer_fold'] = np.array(range(1,11))
    table['LogReg hp'] = outer_fold_hp_lr
    table['LogReg Error'] = outer_scores_lr
    table['KNN hp'] = outer_fold_hp_knn
    table['KNN Error'] = outer_scores_knn
    table['ANN hp'] = outer_fold_hp_ann
    table['ANN Error'] = outer_scores_ann
    table['Baseline Error'] = outer_scores_bas
    
    print(table)

    # we perform the mcnamer test to see if one model is statistically better than another
    knn_bas_test = perform_stat_test(np.array(y_true_all), np.array(knn_preds_all), np.array(bas_preds_all))
    knn_lr_test = perform_stat_test(np.array(y_true_all), np.array(knn_preds_all), np.array(logreg_preds_all))
    lr_bas_test = perform_stat_test(np.array(y_true_all), np.array(logreg_preds_all), np.array(bas_preds_all))
    ann_bas_test = perform_stat_test(np.array(y_true_all), np.array(ann_preds_all), np.array(bas_preds_all))

    print(knn_bas_test)
    print(knn_lr_test)
    print(lr_bas_test)
    print(ann_bas_test)

    
    preds, error, coefs, intercepts = LogReg_coef(25, x_train, y_train, x_unseen, y_unseen)
    print(error, coefs, intercepts)
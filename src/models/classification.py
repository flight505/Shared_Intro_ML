import operator
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import torch
from loader import reader
#from train_neural_net import Net
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, preprocessing, tree
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F


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


def KNN(neighbors, X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf = clf.fit(X_train, y_train)
    return clf, 1-clf.score(X_test, y_test)

def LogReg(lamb, X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver = 'lbfgs', C=1/lamb, max_iter=400)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, 1-metrics.accuracy_score(y_test, y_pred)

def DecisionTrees(X_train, y_train, X_test, y_test):
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, 1-metrics.accuracy_score(y_test, y_pred)

def baseline(y_train, y_test):
    most_common = Counter(y_train).most_common(1)[0][0]
    preds = [most_common]*len(y_train)
    return 1-accuracy_score(y_train, preds)

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
    _, predicted = torch.max(outputs.data, 1)
    return ann_model, 1-accuracy_score(y_test, predicted)


if __name__ == "__main__":
    
    test_run = True # set it true to run the digits classification task, false to run our HCV staging classification task
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


    knn_candidate_parameter = [1, 3, 5, 7, 9, 11]
    lr_candidate_parameter = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    dt_candidate_parameter = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] # fictional - going to delete DT anyways
    ann_candidate_parameter = [1, 8, 16, 32, 64, 128]

    outer_scores_knn, outer_scores_lr, outer_scores_dt, outer_scores_ann = [], [], [], []
    outer_fold_hp_knn, outer_fold_hp_lr, outer_fold_hp_dt, outer_fold_hp_ann = [], [], [], []
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for outer_fold, (train_index_outer, test_index_outer) in enumerate(outer.split(x_train, y_train)):
        print(f"Running {outer_fold+1}/10 fold...")
        x_train_outer, x_test_outer = x_train[train_index_outer], x_train[test_index_outer]
        y_train_outer, y_test_outer = y_train[train_index_outer], y_train[test_index_outer]

        param_scores_knn = {param: [] for param in knn_candidate_parameter}
        param_scores_lr = {param: [] for param in lr_candidate_parameter}
        param_scores_dt = {param: [] for param in dt_candidate_parameter}
        param_scores_ann = {param: [] for param in ann_candidate_parameter}

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

                ann_model, E_val = ANN(ann_candidate_parameter[idx], x_train_inner, y_train_inner, x_test_inner, y_test_inner)
                param_scores_ann[ann_candidate_parameter[idx]].append(E_val)

                bas_score = baseline(y_train_inner, y_test_inner)


        for k in param_scores_knn.keys():
            param_scores_knn[k] = np.mean(param_scores_knn[k])

        for k in param_scores_lr.keys():
            param_scores_lr[k] = np.mean(param_scores_lr[k])

        for k in param_scores_dt.keys():
            param_scores_dt[k] = np.mean(param_scores_dt[k])

        for k in param_scores_ann.keys():
            param_scores_ann[k] = np.mean(param_scores_ann[k])
        
        opt_knn = min(param_scores_knn.items(), key=operator.itemgetter(1))[0]
        opt_lr = min(param_scores_lr.items(), key=operator.itemgetter(1))[0]
        opt_dt = min(param_scores_dt.items(), key=operator.itemgetter(1))[0]
        opt_ann = min(param_scores_ann.items(), key=operator.itemgetter(1))[0]
        outer_fold_hp_knn.append(opt_knn)
        outer_fold_hp_lr.append(opt_lr)
        outer_fold_hp_dt.append(opt_dt)
        outer_fold_hp_ann.append(opt_ann)
                
        clf, E_test = KNN(opt_knn, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_knn.append(round(E_test,3))

        clf, E_test = LogReg(opt_lr, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_lr.append(round(E_test,3))

        clf, E_test = DecisionTrees(x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_dt.append(round(E_test,3))

        clf, E_test = ANN(opt_ann, x_train_outer, y_train_outer, x_test_outer, y_test_outer)
        outer_scores_ann.append(round(E_test,3))

    # print also outer_fold_hp_* for table in project description
    print(outer_scores_knn)
    print(outer_scores_lr)
    print(outer_scores_dt)
    print(outer_scores_ann)
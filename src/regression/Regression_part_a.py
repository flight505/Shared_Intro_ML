#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:33:59 2020

@author: olebatting
"""
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
import torch
from scipy import stats

#load data
df = pd.read_csv('HCV-Egy-Data.csv')
# df =pd.read_csv('avocado.csv')

#Split into X and y
df.pop('Baselinehistological staging')
y = df.pop('Baseline histological Grading')
y = y.to_numpy()

# df.pop('Date')
# df.pop('type')
# df.pop('region')
# y=df.pop('AveragePrice')
# y=y.to_numpy()
# print(len(y))

#set mean=0 and standard deviation=1 ------------
for i in df:
    
    df[i] = df[i].astype(float)
    mu = np.mean(df[i])
    std = np.std(df[i])
    df[i] -= mu
    df[i] /= std


X = df.to_numpy()
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
M = M+1

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda and hidden units
lambdas = np.power(10.,np.arange(-2,10,0.5))
# lambdas = np.power(10.,np.arange(-20,2,1))


# Initialize variables
Error_test_rlr = np.zeros((K,len(lambdas)))


for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    
    #regularized linear regression
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    
    for l,la in enumerate(lambdas):
        
        lambdaI = la * np.eye(M)
        lambdaI[0,0] = 0
        w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        Error_test_rlr[k,l] = np.square(y_test-X_test @ w).sum(axis=0)/y_test.shape[0]
            
        
    
    plt.semilogx(lambdas,Error_test_rlr[k,:], label='k = %d' % k)
    
    

    #baseline
    Error_test_baseline = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    print(Error_test_baseline,min(Error_test_rlr[k,:]))

plt.legend()    
plt.ylabel('Test Error')
plt.xlabel('Lambda')
plt.title('Regularized linear regression: K-fold')
plt.show()    



Error_test_rlr_T = Error_test_rlr.T

Error_mu_rlr = np.mean(Error_test_rlr_T, axis=1)
print(min(Error_mu_rlr))

plt.semilogx(lambdas,Error_mu_rlr, label='mean')
plt.ylabel('Test Error')
plt.xlabel('Lambda')
plt.title('Regularized linear regression: Mean')
plt.show() 
    
    

    
    
    
    

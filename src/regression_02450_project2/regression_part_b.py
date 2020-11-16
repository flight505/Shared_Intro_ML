# from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           # title, subplot, show, grid)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import *
import torch
# from scipy import stats
import datetime

now = datetime.datetime.now()
print(now)

dist = True

#load data
df = pd.read_csv('HCV-Egy-Data.csv')
# df = pd.read_csv('avocado.csv')

#Split into X and y
df.pop('Baselinehistological staging')
y = df.pop('Baseline histological Grading')
y = y.to_numpy()

# df.pop('Date')
# df.pop('type')
# df.pop('region')
# y=df.pop('AveragePrice')
# y=y.to_numpy()


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
K = 5
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda and hidden units
lambdas = np.power(10.,np.arange(-2,10,0.5))
nhus = [1,2,3,4,5]
# lambdas = np.power(10.,np.arange(-20,2,1))
# nhus = [1,30,60,90,120]

# Parameters for neural network classifier
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# Initialize variables
#Error_train_baseline = np.empty((K,1))
Error_test_baseline = np.zeros((K,1))
#Error_train_rlr_outer = np.empty((K,1))
Error_test_rlr_outer = np.zeros((K,1))
#Error_train_ann_outer = np.empty((K,1))
Error_test_ann_outer = np.zeros((K,1))
#w_rlr = np.empty((M,K))
Opt_lambdas = np.zeros((K,1))
Opt_nhus = np.zeros((K,1))

y_true = []

yhat0A = []
r0A = []

yhat0B = []
r0B = []

yhatAB = []
rAB = []


for (k, (train_index_outer, test_index_outer)) in enumerate(CV.split(X,y)):
    print('outer fold '+str(k))
    
    # extract training and test set for current CV fold
    X_train_outer = X[train_index_outer]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = y[test_index_outer]
    
    
    Error_test_rlr = np.zeros(len(lambdas))
    Error_test_ann = np.zeros(len(nhus))
    
    
    #inner 
    for (k_inner, (train_index_inner, test_index_inner)) in enumerate(CV.split(X_train_outer,y_train_outer)):
        print('inner fold '+str(k_inner))
        
        
        # extract training and test set for current CV fold
        X_train_inner = X_train_outer[train_index_inner]
        y_train_inner = y_train_outer[train_index_inner]
        X_test_inner = X_train_outer[test_index_inner]
        y_test_inner = y_train_outer[test_index_inner]
        
        
        #regularized linear regression
        Xty = X_train_inner.T @ y_train_inner
        XtX = X_train_inner.T @ X_train_inner
        
        
        for l,la in enumerate(lambdas):
            lambdaI = la * np.eye(M)
            lambdaI[0,0] = 0
            
            w = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            Error_test_rlr[l] += np.square(y_test_inner-X_test_inner @ w).sum(axis=0)/y_test_inner.shape[0]
            if l==11 and k_inner==K-1 and not dist:
                print(w)
                plt.hist(y_test_outer,bins=16)
                plt.title('y_test distribution')
                plt.show()
                plt.hist(X_test_outer @ w,bins=16)
                plt.title('prediction value distribution')
                plt.show()
                
        
        for h,nhu in enumerate(nhus):
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, nhu), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(nhu, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            X_train = torch.Tensor(X_train_inner)
            y_train = torch.Tensor([[y] for y in y_train_inner])
            X_test = torch.Tensor(X_test_inner)
            y_test = torch.Tensor([[y] for y in y_test_inner])
            
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter,)
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            y_test_est = y_test_est.detach().numpy().squeeze()
            
            # Determine errors and errors
            Error_test_ann[h] += np.square(y_test_inner-y_test_est).sum(axis=0)/y_test_inner.shape[0]
            print(nhu,Error_test_ann[h])
            
            
        if k_inner==K-1:
            plt.semilogx(lambdas,Error_test_rlr/K)
            plt.ylabel('Test Error')
            plt.xlabel('Lambda')
            plt.title('Regularized linear regression')
            plt.show()

            plt.plot(nhus,Error_test_ann/K)
            plt.ylabel('Test Error')
            plt.xlabel('Hidden units')
            plt.title('Artificial neural network')
            plt.show()
            
    
    #baseline
    Error_test_baseline[k] = np.square(y_test_outer-y_test_outer.mean()).sum(axis=0)/y_test_outer.shape[0]
    
    
    #regularized linear regression
    opt_lambda = 0
    min_error_rlr = 0
    for l,la in enumerate(lambdas):
        if l==0 or Error_test_rlr[l]<min_error_rlr:
            opt_lambda = la
            min_error_rlr = Error_test_rlr[l]
    
    Opt_lambdas[k] = opt_lambda
    
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0
    
    Xty = X_train_inner.T @ y_train_inner
    XtX = X_train_inner.T @ X_train_inner
    
    w_opt = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    Error_test_rlr_outer[k] = np.square(y_test_outer-X_test_outer @ w_opt).sum(axis=0)/y_test_outer.shape[0]
    
    
    #artificial neural network
    opt_nhu = 0
    min_error_ann = 0
    for h,nhu in enumerate(nhus):
        if  h==0 or Error_test_ann[h]<min_error_ann:
            opt_nhu = nhu
            min_error_ann = Error_test_ann[h]
    
    Opt_nhus[k] = opt_nhu
    
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_nhu), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_nhu, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    X_train = torch.Tensor(X_train_outer)
    y_train = torch.Tensor([[y] for y in y_train_outer])
    X_test = torch.Tensor(X_test_outer)
    y_test = torch.Tensor([[y] for y in y_test_outer])
    
    net, final_loss, learning_curve = train_neural_net(model,
                                               loss_fn,
                                               X=X_train,
                                               y=y_train,
                                               n_replicates=n_replicates,
                                               max_iter=max_iter,)
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    y_test_est = net(X_test)
    y_test_est = y_test_est.detach().numpy().squeeze()
    print(datetime.datetime.now()-now)
    
    
    # Determine errors and errors
    y_test = y_test.numpy()
    Error_test_ann_outer[k] = np.square(y_test_outer-y_test_est).sum(axis=0)/y_test_outer.shape[0]
    
    yhat0 = [[y_test_outer.mean()]]*y_test_outer.shape[0]
    yhatA = [[yi] for yi in (X_test_outer @ w_opt)]
    yhatB = [[yi] for yi in y_test_est]
    y_true.append(y_test)
    
    yhat0A.append( np.concatenate([yhat0, yhatA], axis=1) )
    yhat0B.append( np.concatenate([yhat0, yhatB], axis=1) )
    yhatAB.append( np.concatenate([yhatA, yhatB], axis=1) )
    
    r0A.append( np.mean( np.abs( yhat0-y_test ) ** 2 - np.abs( yhatA-y_test) ** 2 ) )
    r0B.append( np.mean( np.abs( yhat0-y_test ) ** 2 - np.abs( yhatB-y_test) ** 2 ) )
    rAB.append( np.mean( np.abs( yhatA-y_test ) ** 2 - np.abs( yhatB-y_test) ** 2 ) )


    
         
print('baseline')
print(Error_test_baseline)
print('rlr')
print(Error_test_rlr_outer)
print(Opt_lambdas)
print('ann')
print(Error_test_ann_outer)
print(Opt_nhus)

print(datetime.datetime.now()-now)

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII0A, CI_setupII0A = correlated_ttest(r0A, rho, alpha=alpha)
p_setupII0B, CI_setupII0B = correlated_ttest(r0B, rho, alpha=alpha)
p_setupIIAB, CI_setupIIAB = correlated_ttest(rAB, rho, alpha=alpha)

print('baseline vs rlr')
print( p_setupII0A )
print( CI_setupII0A )

print('baseline vs ann')
print( p_setupII0B )
print( CI_setupII0B )

print('rlr vs ann')
print( p_setupIIAB )
print( CI_setupIIAB )

print(datetime.datetime.now()-now)
            
    
    
    

    
    
    
    

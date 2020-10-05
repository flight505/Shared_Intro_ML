import pandas as pd
from sklearn.preprocessing._data import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from numpy.linalg.linalg import svd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import streamlit as st

dataset = pd.read_csv("src/data/HCV-Egy-Data.csv", delimiter=',')

def PCA_run(dataset):
    features_col = list(dataset.columns)[:-2]
    targets_col = list(dataset.columns)[-1]

    x = dataset.loc[:, features_col].values

    new_x = []
    cols = [1,3,4,5,6,7,8,9]
    for column in range(x.shape[1]):
        col_vector = x[:,column]
        if column not in cols:
            new_x.append(col_vector)
        else:
            integer_encoded = LabelEncoder().fit_transform(col_vector)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)
            for one_hot_enc_col in range(onehot_encoded.shape[1]):
                new_x.append(onehot_encoded[:,one_hot_enc_col])

    new_x = np.array(new_x).T

    y = dataset.loc[:,[targets_col]].values
    new_x = StandardScaler().fit_transform(new_x)


    pca = PCA(n_components=new_x.shape[1])
    principalComponents = pca.fit_transform(new_x)

    U,S,Vh = svd(new_x, full_matrices=True)

    rho = (S*S) / (S*S).sum() 

    threshold = st.slider('threshold', min_value=0.1, max_value=4.0, value=0.9, step=0.2)

    # Plot variance explained
    fig = figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()

import pandas as pd
from sklearn.preprocessing._data import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from numpy.linalg.linalg import svd
from matplotlib import pyplot as plt

dataset = pd.read_csv("src/data/HCV-Egy-Data.csv", delimiter=',')

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
        
x = np.array(new_x).T

y = dataset.loc[:,[targets_col]].values
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=x.shape[1])
principalComponents = pca.fit_transform(x)
print(principalComponents)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

U,S,Vh = svd(x, full_matrices=True)

rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
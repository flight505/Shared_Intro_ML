import pandas as pd
from sklearn.preprocessing._data import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from numpy.linalg.linalg import svd
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
from matplotlib.pyplot import figure

dataset = pd.read_csv("src/data/HCV-Egy-Data.csv", delimiter=',')

features_col = list(dataset.columns)[:-2]
new_features_col = ['Age', 'Male','Female','BMI',
                    'No Fever','Fever', 'No Nausea/Vomiting','Nausea/Vomiting',
                    'No Headache','Headache', 'No Diarrhea','Diarrhea',
                    'No Fatigue+Boneache','Fatigue+Boneache',
                    'No Jaundice','Jaundice', 'No Epigastric pain', 'Epigastric pain',
                    'WBC','RBC','HGB','Plat','AST 1','ALT 1','ALT4','ALT 12','ALT 24','ALT 36','ALT 48',
                    'ALT after 24 w','RNA Base', 'RNA 4','RNA 12','RNA EOT','RNA EF']
targets_col_reg = list(dataset.columns)[-2]
targets_col_clas = list(dataset.columns)[-1]


x = dataset.loc[:, features_col].values

new_x = []
cols = [1,3,4,5,6,7,8,9] # binary columns
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
x_df = pd.DataFrame(new_x)
x_df.columns = new_features_col


y_clas = dataset.loc[:,[targets_col_clas]].values
y_reg = dataset.loc[:,[targets_col_reg]].values

y_reg = y_reg/np.max(y_reg)
y_reg = np.add(y_reg, y_clas)
y_reg = y_reg-np.min(y_reg)
y_reg = y_reg/np.max(y_reg)



f = plt.figure(figsize=(15, 15))
plt.matshow(x_df.corr(), fignum=f.number)
plt.xticks(range(x_df.shape[1]), x_df.columns, fontsize=10, rotation=90)
plt.yticks(range(x_df.shape[1]), x_df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=16, y=1.22)
plt.savefig('src/viz/plots/cross_correlation_matrix.png')



cols_to_drop = [1,3,4,5,6,7,8,9]
plt.clf()
plt.cla()
plt.style.use('ggplot')
fig = figure()
fig, ax = plt.subplots(5, 6)
fig.suptitle('Features Distribution')
for col in range(x.shape[1]):
    if col not in cols_to_drop:
        n, bins, _ = plt.hist(x[:,col], bins=20, rwidth=0.50)
        data = []
        for index in range(n.shape[0]):
            data += [bins[index]]*int(n[index])
        density = gaussian_kde(data)
        xs = np.linspace(min(bins), max(bins), 50)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        matrix_row, matrix_col = int(col/6), int(col%6)
        ax[matrix_row, matrix_col].plot(xs, density(xs))
    else:
        matrix_row, matrix_col = int(col/6), int(col%6)
        ax[matrix_row, matrix_col].hist(x[:,col])
        
for i, ax in enumerate(ax.flat):
    if i == 29:
        ax.cla()
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
    


plt.savefig('src/viz/plots/features_dist.png')


plt.clf()
plt.hist(y_clas)
plt.title("Target Distribution for Classification")
plt.savefig('src/viz/plots/target_classes.png')

plt.clf()
plt.hist(y_reg)
plt.title("Target Distribution for Regression")
plt.savefig('src/viz/plots/target_regr.png')



"""
plt.clf()
plt.boxplot(data_to_plot)
plt.title("Box-and-Whisker for continuous variables")
plt.show()
"""
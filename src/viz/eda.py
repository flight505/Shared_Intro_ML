import pandas as pd
from sklearn.preprocessing._data import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from numpy.linalg.linalg import svd
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing

dataset = pd.read_csv("src/data/HCV-Egy-Data.csv", delimiter=',')

features_col = list(dataset.columns)[:-2]
new_features_col = ['Age', 'Male','Female','BMI',
                    'No Fever','Fever', 'No Nausea/Vomiting','Nausea/Vomiting',
                    'No Headache','Headache', 'No Diarrhea','Diarrhea',
                    'No Fatigue','Fatigue',
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

f = plt.figure(figsize=(15, 15))
plt.matshow(x_df.corr(), fignum=f.number)
plt.xticks(range(x_df.shape[1]), x_df.columns, fontsize=10, rotation=90)
plt.yticks(range(x_df.shape[1]), x_df.columns, fontsize=10)
cb = plt.colorbar(cmap='plasma')
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=16, y=1.22)
plt.savefig('src/viz/plots/cross_correlation_matrix.png')



cat_cols = ['Gender','BMI','Fever','Nausea/Vomting','Headache ',
                'Diarrhea ','Fatigue ','Jaundice ','Epigastric_pain ']
cont_cols = [x for x in features_col if x not in cat_cols]



plt.clf()
plt.cla()
plt.style.use('ggplot')
fig = figure()
fig, ax = plt.subplots(3, 6)
fig.suptitle('Age & Medical Measurements Distribution')
for i, ax in zip(range(18), ax.flat):
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
    sns.distplot(dataset[cont_cols[i]], bins=20, label=cont_cols[i], rug=True, ax=ax)
    ax.set_xlabel(cont_cols[i], fontsize=10)

plt.savefig('src/viz/plots/med_measurements_dist.png')


plt.clf()
plt.cla()
plt.style.use('ggplot')
fig = figure()
fig, ax = plt.subplots(3, 3)
fig.suptitle('Gender, BMI & Sympthoms Distribution')
for i, ax in zip(range(9), ax.flat):
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
    sns.countplot(dataset[cat_cols[i]], label=cat_cols[i], ax=ax, palette="rocket")
    ax.set_xlabel(cat_cols[i], fontsize=10)

plt.savefig('src/viz/plots/sympthoms_measurements_dist.png')





y_clas = dataset.loc[:,[targets_col_clas]].values
class_names = ['Portal Fibrosis', 'Few Septa', 'Many Septa', 'Cirrhosis']
t_names = []
for i in range(len(y_clas)):
    t_names.append(class_names[y_clas[i][0]-1])
t_names = np.array(t_names)

plt.clf()
sns.countplot(t_names, palette="rocket", order=['Portal Fibrosis','Few Septa','Many Septa','Cirrhosis'])
plt.title("Target Distribution for Classification")
plt.savefig('src/viz/plots/target_classes.png')


plt.clf()
sns.countplot(dataset['Baseline_histological_Grading'], palette="rocket")
plt.title("Target Distribution for Regression")
plt.xticks([])
plt.savefig('src/viz/plots/target_regr.png')


h_order=[1,2, 3, 4]
plt.clf()
plt.cla()
plt.style.use('ggplot')
fig = figure()
fig, ax = plt.subplots(3, 6)
fig.suptitle('Age & Medical Measurements vs Staging')
for i, ax in zip(range(18), ax.flat):
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
    sns.scatterplot(dataset[cont_cols[i]], dataset['Baselinehistological_staging'], hue_order=h_order,
            hue=dataset['Baselinehistological_staging'], legend=None, label=cont_cols[i], palette="vlag", ax=ax,size=1)
    ax.set_xlabel(cont_cols[i], fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:0], labels[:0])

plt.savefig('src/viz/plots/meas_vs_staging.png')

plt.clf()
plt.cla()
plt.style.use('ggplot')
fig = figure()
fig, ax = plt.subplots(3, 3)
fig.suptitle('Gender, BMI & Sympthoms vs Staging')
for i, ax in zip(range(9), ax.flat):
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
    sns.countplot(x=dataset[cat_cols[i]], hue=dataset["Baselinehistological_staging"],
                data=dataset, ax=ax,
                hue_order=h_order,palette='vlag')
    ax.set_xlabel(cat_cols[i], fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:0], labels[:0])

plt.savefig('src/viz/plots/sympt_vs_staging.png')

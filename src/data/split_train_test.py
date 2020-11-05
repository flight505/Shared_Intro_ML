import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("src/data/HCV-Egy-Data.csv", delimiter=',')
dataset.drop(labels=['RNA_12','RNA_EOT','RNA_EF'], axis=1, inplace=True)

features_col = list(dataset.columns)[:-2]
new_features_col = ['Age', 'Gender','BMI','Fever','Nausea/Vomting',
                   'Headache ','Diarrhea ','Fatigue ','Jaundice ','Epigastric_pain ',
                    'WBC','RBC','HGB','Plat','AST 1','ALT 1','ALT4','ALT 12','ALT 24','ALT 36','ALT 48',
                    'ALT after 24 w','RNA Base', 'RNA 4']

targets_col_reg = list(dataset.columns)[-2]
targets_col_clas = list(dataset.columns)[-1]

alt_36_mean = int(np.mean(dataset[dataset['ALT_36'] != 5]['ALT_36']))
alt_48_mean = int(np.mean(dataset[dataset['ALT_48'] != 5]['ALT_48']))
alt_after_24_w_mean = int(np.mean(dataset[dataset['ALT_after_24_w'] != 5]['ALT_after_24_w']))

dataset['ALT_36_temp'] = np.where(dataset['ALT_36']==5, alt_36_mean, dataset['ALT_36'])
dataset['ALT_48_temp'] = np.where(dataset['ALT_48']==5, alt_48_mean, dataset['ALT_48'])
dataset['ALT_after_24_w_temp'] = np.where(dataset['ALT_after_24_w']==5, alt_after_24_w_mean, dataset['ALT_after_24_w'])

dataset['ALT_36'] = dataset['ALT_36_temp']
dataset['ALT_48'] = dataset['ALT_48_temp']
dataset['ALT_after_24_w'] = dataset['ALT_after_24_w_temp']

dataset.drop(labels=['ALT_36_temp','ALT_48_temp','ALT_after_24_w_temp'], axis=1, inplace=True)

y_class = dataset['Baselinehistological_staging']
dataset.drop(labels=['Baselinehistological_staging', 'Baseline_histological_Grading'], axis=1, inplace=True)


x_train, x_test, y_train, y_test = train_test_split(dataset, y_class,
                                                    stratify=y_class, 
                                                    test_size=0.2)


x_train = x_train.loc[:, features_col]
x_test = x_test.loc[:, features_col]
x_train_to_std, x_train_bin = pd.DataFrame(), pd.DataFrame()
x_test_to_std, x_test_bin = pd.DataFrame(), pd.DataFrame()
cols = ['Gender','Fever','Nausea/Vomting','Headache ','Diarrhea ','Fatigue ','Jaundice ','Epigastric_pain ']
for column in x_train:
    if column not in cols:
        x_train_to_std[column] = x_train[column]
        x_test_to_std[column] = x_test[column]
    else:
        x_train_bin[column] = x_train[column]-1
        x_test_bin[column] = x_test[column]-1
        
scaler = StandardScaler()
scaler.fit(x_train_to_std)
x_train_std = scaler.transform(x_train_to_std)
x_test_std = scaler.transform(x_test_to_std)

x_train_bin = x_train_bin.values
x_test_bin = x_test_bin.values

x_train = np.empty((x_train_std.shape[0], x_train_std.shape[1]+x_train_bin.shape[1]))
x_test = np.empty((x_test_std.shape[0], x_test_std.shape[1]+x_test_bin.shape[1]))

for i in range(x_train_std.shape[0]):
    x_train[i] = np.concatenate((x_train_std[i], x_train_bin[i]), axis=None)

for i in range(x_test_std.shape[0]):
    x_test[i] = np.concatenate((x_test_std[i], x_test_bin[i]), axis=None)

np.savetxt("src/data/splits/classification/x_train.csv", x_train, delimiter=",")
np.savetxt("src/data/splits/classification/x_test.csv", x_test, delimiter=",")
np.savetxt("src/data/splits/classification/y_train.csv", y_train, delimiter=",")
np.savetxt("src/data/splits/classification/y_test.csv", y_test, delimiter=",")


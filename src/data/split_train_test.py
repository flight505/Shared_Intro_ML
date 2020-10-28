import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

standardize = True

dataset = pd.read_csv("src/data/HCV-Egy-Data.csv", delimiter=',')
dataset.drop(labels=['RNA_12','RNA_EOT','RNA_EF'], axis=1, inplace=True)

features_col = list(dataset.columns)[:-2]
new_features_col = ['Age', 'Male','Female','BMI',
                    'No Fever','Fever', 'No Nausea/Vomiting','Nausea/Vomiting',
                    'No Headache','Headache', 'No Diarrhea','Diarrhea',
                    'No Fatigue','Fatigue',
                    'No Jaundice','Jaundice', 'No Epigastric pain', 'Epigastric pain',
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

x = dataset.loc[:, features_col].values
new_x = []
cols = [1,3,4,5,6,7,8,9] # binary columns
for column in range(x.shape[1]):
    col_vector = x[:,column]
    if column not in cols:
        if standardize:
            col_mean = np.mean(col_vector)
            col_std = np.std(col_vector)
            new_x.append((col_vector-col_mean)/col_std)
        else:
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
x_df['Baselinehistological_staging'] = dataset['Baselinehistological_staging']
x_df['Baseline_histological_Grading'] = dataset['Baseline_histological_Grading']

y_class = x_df['Baselinehistological_staging']
y_regr = x_df['Baseline_histological_Grading']

x_df.drop(labels=['Baselinehistological_staging', 'Baseline_histological_Grading'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(x_df, y_class,
                                                    stratify=y_class, 
                                                    test_size=0.2)

X_train.to_csv("src/data/splits/classification/x_train.csv", index=False)
y_train.to_csv("src/data/splits/classification/y_train.csv", index=False)
X_test.to_csv("src/data/splits/classification/x_test.csv", index=False)
y_test.to_csv("src/data/splits/classification/y_test.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(x_df, y_regr,
                                                    stratify=y_regr, 
                                                    test_size=0.2)

X_train.to_csv("src/data/splits/regression/x_train.csv", index=False)
y_train.to_csv("src/data/splits/regression/y_train.csv", index=False)
X_test.to_csv("src/data/splits/regression/x_test.csv", index=False)
y_test.to_csv("src/data/splits/regression/y_test.csv", index=False)


#!/usr/bin/env python3
 
##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 2 Oct 2020                                 #
# Version:	0.0.1                                        #
##########################################################

import os
os.system('cls||clear') # this line clears the screen 'cls' = windows 'clear' = unix

#%%
import pandas as pd
import plotly.express as px
import altair as alt

dataset = pd.read_csv('/Users/jvang/Documents/Projects/DTU_intro_ML/src/Shared_Intro_ML/src/data/HCV-Egy-Data.csv')

#Rename Columns
dataset.columns=['Age ', 'Gender', 'BMI', 'Fever', 'Nausea/Vomting','Headache',
       'Diarrhea','Fatigue','Jaundice',
       'Epigastric_pain', 'WBC','RBC','HGB', 'Plat','AST_1','ALT_1',
       'ALT_4', 'ALT_12','ALT_24','ALT_36','ALT 48','ALT_after_24w',
       'RNA_Base','RNA 4','RNA_12', 'RNA_EOT','RNA_EF',
       'Baseline_histological_Grading','Baselinehistological_staging']

dataset.head()
data_cat = dataset.astype('category')
#Change the classes columns to categorical for better visualization 
#categorize columns: Gender,Fever,Nausea/Vomting,Headache,Diarrhea,Fatigue,Jaundice
data_cat=dataset[['Gender','Fever','Nausea/Vomting','Headache','Fatigue','Jaundice','Diarrhea','Epigastric_pain',"Baselinehistological_staging",'Baseline_histological_Grading']]
#
##Replacing values (1-2) to (Absent,Present) in Symptoms Features:
#
data_cat['Fever'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Nausea/Vomting'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Headache'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Fatigue'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Jaundice'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Diarrhea'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Epigastric_pain'].replace([1,2],['Absent','Present'],inplace=True)
#
#Replacing the values to names e.g. 1:Male,2:Female 1:Absent,2:present 
data_cat['Gender'].replace([1,2],['Male','Female'],inplace=True)
data_cat['Gender']

#Doing the same for Histological Staging
data_cat['Baselinehistological_staging'].unique()
data_cat['Baselinehistological_staging'].replace([1,2,3,4],['Portal Fibrosis','Few Septa','Many Septa','Cirrhosis'],inplace=True)
data_cat.head()

#Gain insights of the categorical data:


fig = px.histogram(data_cat, x="Baselinehistological_staging", color="Gender", marginal="rug", # can be `box`, `violin`
                         hover_data=data_cat.columns)
#fig.show()



print(data_cat['Baselinehistological_staging'])


alt.Chart(data_cat).mark_bar().encode(
    alt.X('Baselinehistological_staging:N'),
    alt.Y('Gender:Q', axis=alt.Axis(tickMinStep=1))
)


# %%

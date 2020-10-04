#!/usr/bin/env python3
 
##########################################################
# Copyright (c) Jesper Vang <jesper_vang@me.com>         #
# Created on 1 Oct 2020                                 #
# Version:	0.0.1                                        #
##########################################################
import streamlit as st
import pandas as pd

def display_reports():
    report1_check = st.checkbox("📃 Display Report 1 ")
    if report1_check:
        st.markdown(f"""
        # 02450 Introduction to Machine Learning and Data Mining
---
# Report Project 1



### 1. Dataset Description
 
• Explain what your data is about. I.e. what is the overall problem of
interest?

Based on various medical information we're interested in predicting staging of liver fibrosis in patients. More specific we're interested in classifying whether a patient with hepatitis C presents with no fibrosis, portal fibrosis, few septa, many septa or cirrhosis.

• Provide a reference to where you obtained the data.

A dataset based on Egyptian patients who underwent treatment dosages for HCV for about 18 months was found [here](https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+%28HCV%29+for+Egyptian+patients).

• Summarize previous analysis of the data. (i.e. go through one or two of the original source papers and read what they did to the data and summarize their results).

[Nasr et al.] achieved 99.48% average accuracy on 5-fold cross validation from a predition model based on the data. Their goal was to present a noninvasive alternative to serial liver biopsies in assessing fibrosis. They used a Subsumtion Rule Based Classifier to evaluate rules from a discretization set, based on expert recommendations, to produce a decision tree.

• You will be asked to apply (1) classification and (2) regression on your
data in the next report. For now, we want you to consider how this should
be done. Therefore:
Explain, in the context of your problem of interest, what you hope to
accomplish/learn from the data using these techniques?.
Explain which attribute you wish to predict in the regression based on
which other attributes? ~~Which class label will you predict based on which
other attributes in the classification task?
If you need to transform the data in order to carry out these tasks, explain roughly how you plan to do this.~~

We hope to produce a model with hopefully a reduced feature dimensionality (A PCA will be done to the full feature set) to predict staging. It is thus a classification we're aiming for. This classification will be denoted the main machine learning aim in the following.

One of these tasks (1)–(5) is likely more relevant than the rest and will be denoted the main machine learning aim in the following. The purpose of the following questions, which asks you to describe/visualize the data, is to allow you to reflect on the feasibility of this task.

## 2. Detailed description of data and attributes

• Describe if the attributes are discrete/continuous, 

""", unsafe_allow_html=True)
        Describe_data = pd.read_csv('src/data/Describe_data.csv')
        st.write(Describe_data, width=600, height=900)

        st.markdown(
        f"""
        • Give an account of whether there are data issues (i.e. missing values or corrupted data) and describe them if so.



| Data Set Characteristics: | Attribute Characteristics: | Missing Values: |
| ------------------------- | -------------------------- | --------------- |
| Multivariate              | Integer, Real              | Non             |


""", unsafe_allow_html=True)

        st.subheader("Raw Dataset")
        hep_data = pd.read_csv('src/data/HCV-Egy-Data.csv')
        st.write(hep_data, width=600, height=900)
        st.subheader("Basic summary statistics of the attributes")
        st.write(hep_data.describe(include='all').T, width=600, height=900)

        st.markdown(f"""
    ## 3. Data visualization and PCA
Touch upon the following subjects, use visualizations when it appears sensible. Keep in mind the ACCENT principles and Tufte’s guidelines when you visu- alize the data.
• Are there issues with outliers in the data,
• do the attributes appear to be normal distributed,
• are variables correlated,
• does the primary machine learning modeling aim appear to be feasible based on your visualizations.
There are three aspects that needs to be described when you carry out the PCA analysis for the report:
• The amount of variation explained as a function of the number of PCA components included,
• the principal directions of the considered PCA components (either find a way to plot them or interpret them in terms of the features),
• the data projected onto the considered principal components.
If your attributes have different scales you should include the step where the
data is standardizes by the standard deviation prior to the PCA analysis.

<img src="https://i.imgur.com/xR8ZXaM.png" width="600" height="500">

<img src="https://i.imgur.com/9eNaAPW.png" width="600" height="500">

<img src="https://i.imgur.com/rPmhtnT.png" width="600" height="500">
""", unsafe_allow_html=True)


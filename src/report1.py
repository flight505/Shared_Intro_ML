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

In order to get an initial understanding of the data, we plotted the density distributions for those variables that were medical measurements, such has number of red blood cells, RNA and so on. In this category, for simplicity of representation, we also included the age.

<img src="https://i.imgur.com/kwniGlN.png" width="600" height="500">

Overall, we can appreciate an evenly and uniformely distributed set of features. However, few peculiarity and potential issue came up. In fact, we noticed that the feature *ALT after 24 weeks* had some outliers on the left most part of the distribution. Being very few data points, the issue is easily addressable by replacing those values with the mean of the distribution.
A bigger problem was instead identified in the last three measurements of the last row (reading from left to right), where 27\% (385 out of 1385) of the records have shown to have not realistic values. Due to a lack of documentation by the data curators we could not figure out why that was. The aforementioned techniques of replacing the outliers with the mean of the distribution does not seem like a suitable approach, due to the fact that the outliers represent a significant part of the data. Therefore, we might decide to discard such features all together.
Generally, we saw that all the represented by numerical values. However, the measurement relative to the Hemoglobine - which assumes only six different values throughout the dataset - might as well be represented as a categorical variable when designing the model.
<br>

For completeness, we also plot a count of the categorical features. In these we show the distribution of genders, BMIs and sympthoms. Here, we also see a nice and uniform distribution and no class imbalances.


<img src="https://i.imgur.com/Knic9We.png" width="600" height="500">

For the classification, task our dataset does not show any sign of class imbalance. All four possible classes are overall uniformely distributed.


<img src="https://i.imgur.com/fqdw28G.png" width="600" height="500">

Although we have decided to use our dataset for a classification task, it could be easily turned into a regression task. In fact, together with the *Histological staging* we are provided with the *Histological grading* as well.
To be more specific, each of the four stages can be further classified to a grade scale that ranges from 1 to 16. Luckily, there is no missing data in this variable as well. With simple data pre-processing we have been able to shape the dataset in such a way to have a simil-continuous target to apply regression to. Specifically, we have normalized the grading, i.e. we divided each value by 16 (the maximum possible value), and then we added the resulting value to the stage. With this approach we were able to create a target variable ranging from 1.0625 (staging 1 & grading 1) to 5.0 (staging 4 & grading 16). We might further scale them down in the range [0, 1]. This will effectively allow us to threat the problem as a regression task as well.

<img src="https://i.imgur.com/cQKQJqp.png" width="600" height="500">



<img src="https://i.imgur.com/GQI620S.png" width="600" height="500">





Due to the nature of the dataset, with some attributes being continuous ratios and some other being discrete nominal and ordinal, some data manipulation was required before applying PCA. Eight of the twenty-seven attributes have been one-hot encoded, bringing the final dimensionality to thirty-five attributes. After doing so, the data has been standardized - so each column has been transformed to have zero mean and standard deviation equal to one.
The result of the PCA application can be seen in [Figure 1]()

<img src="https://i.imgur.com/f57YRRp.png" width="600" height="500">


The result of the component analysis is that twenty-three component needs to be used to retain at least 90\% of the variance in the data. Reducing the number of components - and hence dimensionality of the data - will clearly cause an almost constant increase of information loss.
Dimensionality reduction does not seem to suit very well the chosen data. The cross-correlation among features does indeed confirm this belief.

<img src="https://i.imgur.com/MGGcqi4.png" width="600" height="500">

In [Figure 2]() it is possible to see that all feature columns are uncorrelated to each other. The only exception being the binary one-hot encoded features (pair-wise negative correlations). The only slight correlation (40\%) is seen in the last three features. However, that is because of the outliers. Applying PCA to only does column should not be that beneficial, since it would only reduce the dimensionality of two units.




## 4. Discussion of what we have learnt
Summarize here the most important things you have learned about the data and give also your thoughts on whether your primary machine learning aim appears to be feasible based on your visualization.

Even columns that one would believe to be possibly correlated, have no correlation at all. (RNA measurement and similar)

""", unsafe_allow_html=True)


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

**Group members**:
- Jesper
- Ole Batting
- Marco Placenti - s202798

**Contribution table**:
| Name	| Section |
|-	|-	|
| Jesper |  2 & 3	|
| Ole Batting | 1 & 4 |
| Marco Placenti | 3 & 4 |

## Links
- [GitHub Code Repository](https://github.com/flight505/Shared_Intro_ML)
- [Web Application](https://hepatitis-disease-prediction.herokuapp.com)

---
### 1. Dataset Description
Based on various medical information we're interested in predicting staging of liver fibrosis in patients. More specific we're interested in classifying whether a patient with hepatitis C presents with portal fibrosis, few septa, many septa or cirrhosis.


A dataset based on Egyptian patients who underwent treatment dosages for HCV for about 18 months was found [here](https://archive.ics.uci.edu/ml/datasets/Hepatitis+C+Virus+%28HCV%29+for+Egyptian+patients).


[[Nasr et al. 2017]](https://www.researchgate.net/publication/323130913_A_novel_model_based_on_non_invasive_methods_for_prediction_of_liver_fibrosis) achieved 99.48% average accuracy on 5-fold cross validation from a predition model based on the data. Their goal was to present a noninvasive alternative to serial liver biopsies in assessing fibrosis. They used a Subsumtion Rule Based Classifier to evaluate rules from a discretization set, based on expert recommendations, to produce a decision tree.


We hope to produce a model with hopefully a reduced feature dimensionality (A PCA will be done to the full feature set) to predict staging. It is thus a classification we're aiming for. This classification is the main machine learning aim.
Besides this a regression will be done for a simil-continuous target comprised of the histological grading. The prediction would then be a continuous value which, even though the target is discrete, would still make sense/be useful. 

With both the classification and the regression we hope to predict the level of liver fibrosis.

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
• Give an account of whether there are data issues (i.e. missing values or corrupted data) and describe them if so.



| Data Set Characteristics:   | Attribute Characteristics: | Missing Values: |
|-----------------------------|----------------------------|-----------------|
| Multivariate                | Integer, Real              | Non             |
• Include basic summary statistics of the attributes.
If your data set contains many similar attributes, you may restrict yourself to
describing a few representative features (apply common sense).

## 3. Data visualization and PCA

In order to get an initial understanding of the data, we plotted the density distributions for those variables that were medical measurements, such as number of red blood cells, RNA and so on. In this category, for simplicity of representation, we also included the age.


<img src="https://i.imgur.com/kwniGlN.png" width="600" height="500">


Overall, we can appreciate an evenly and uniformely distributed set of features. However, few peculiarities and potential issues came up. In fact, we noticed that the feature *ALT after 24 weeks* had some outliers on the left most part of the distribution. Being very few data points, the issue is easily addressable by replacing those values with the mean of the distribution.
A bigger problem was instead identified in the last three measurements of the last row (reading from left to right), where 27\% (385 out of 1385) of the records have shown to have not realistic values. Due to a lack of documentation by the data curators we could not figure out why that was. The aforementioned techniques of replacing the outliers with the mean of the distribution does not seem like a suitable approach, due to the fact that the outliers represent a significant part of the data. Therefore, we might decide to discard such features all together.
Additionally, the measurement relative to the Hemoglobine - which assumes only six different values throughout the dataset - might as well be represented as a categorical variable when designing the model.

Furthermore, we checked if any of these features could have helped us discriminating between classes. In the y-axis we placed the *Histological staging* while in the x-axis the same features we used in the previous plot. We clearly see that there are no clear cluster and the feature values are well distributed among the classes.


<img src="https://i.imgur.com/sM4lJsx.png" width="600" height="500">


<br>

For completeness, we also plot a count of the categorical features. In these we show the distribution of genders, BMIs and sympthoms. Here, we also see a nice and uniform distribution and no class imbalances.


<img src="https://i.imgur.com/dOTbZb8.png" width="600" height="500">

Just as we did before, we check if any of the sympthoms, BMI or the gender can be used to discriminate between classes.


<img src="https://i.imgur.com/gtQb74I.png" width="600" height="500">


We see that it is not the case, also for these features.

<br>

For the classification, task our dataset does not show any sign of class imbalance. All four possible classes are overall uniformely distributed.



<img src="https://i.imgur.com/CVqSYoG.png" width="600" height="500">


Although we have decided to use our dataset for a classification task, it could be easily turned into a regression task. In fact, together with the *Histological staging* we are provided with the *Histological grading* as well.
This measure indicates how quickly the disease is spreading - and for hepathisis is expressed through a value ranging from 1 to 16. Being the range four times larger than the staging, it is better suited for a regression task.


<img src="https://i.imgur.com/kmITjvJ.png" width="600" height="500">


Due to the nature of the dataset, with some attributes having very large ranges and some other being categorizable, some data manipulation was required before applying PCA. Eight of the twenty-seven attributes have been one-hot encoded, bringing the final dimensionality to thirty-five attributes. After doing so, the data has been standardized - so each column has been transformed to have zero mean and standard deviation equal to one.
The result of the PCA application can be seen in [Figure 1]()

![Figure 1](https://i.imgur.com/f57YRRp.png)
<img src="https://i.imgur.com/f57YRRp.png" width="600" height="500">


The result of the component analysis is that twenty-three component needs to be used to retain at least 90\% of the variance in the data. Reducing the number of components - and hence dimensionality of the data - will clearly cause an almost constant increase of information loss.
Dimensionality reduction does not seem to suit very well the chosen data. The cross-correlation among features does indeed confirm this belief.


<img src="https://i.imgur.com/KhQK2wj.png" width="600" height="500">

In the above cross correlation matrix, we see obviously a perfect correlation in the main diagonal and a perfect negative correlation between the binary variables that have been 1-hot encoded. Also we see a weird light correlation between the last three columns, namely for the reasons we have explained before, i.e. high frequency of outliers and corrupted data.

In Figure 2 it is possible to see that all feature columns are uncorrelated to each other. The only exception being the binary one-hot encoded features (pair-wise negative correlations). The only slight correlation (40\%) is seen in the last three features. However, that is because of the outliers. Applying PCA to only does column should not be that beneficial, since it would only reduce the dimensionality of two units.




## 4. Discussion of what we have learnt
We found that [[Nasr et al. 2017]](https://www.researchgate.net/publication/323130913_A_novel_model_based_on_non_invasive_methods_for_prediction_of_liver_fibrosis) achieved 99.48% average accuracy on 5-fold cross validation from a Subsumtion Rule Based Classifier with staging as target.

Overall, the dataset is very well curated despite some small inconsitencies in some of the measurements.
It is still very difficult to say whether our machine learning aim will be able to deliver good performance since we have not seen anything that suggests an obvious discrimination criterion. In fact, all features seem to be uniformely distributed - even among the four stages we set as a target.

Summarize here the most important things you have learned about the data and give also your thoughts on whether your primary machine learning aim appears to be feasible based on your visualization.


## References
[Nasr et al. 2017] M. Nasr, M. Hamdy, S. M. Kamal and K. Elbahnasy: "A novel model based on non invasive methods for prediction of liver fibrosis", Researchgate, December 2017.



<img src="https://i.imgur.com/MGGcqi4.png" width="600" height="500">


""", unsafe_allow_html=True)


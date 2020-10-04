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

| Attribute                       | Lable                                 | Example | Discrete/Continuous | Nominal/Ordinal/Interval/Ratio |
| ------------------------------- | ------------------------------------- | ------- | ------------------- | ------------------------------ |
| Age                             | Age                                   | 56      | Discrete            | Ratio                          |
| Gender                          | Gender                                | 1       | Discrete            | Nominal                        |
| BMI                             | Body mass Index                       | 35      | Discrete            | Ratio                          |
| Fever                           | Fever                                 | 2       | Discrete            | Nominal                        |
| Nausea/Vomting                  | Nausea/Vomting                        | 1       | Discrete            | Nominal                        |
| Headache                        | Headache                              | 1       | Discrete            | Nominal                        |
| Diarrhea                        | Diarrhea                              | 1       | Discrete            | Nominal                        |
| Fatigue & generalized bone ache | Fatigue & generalized bone ache       | 2       | Discrete            | Nominal                        |
| Jaundice                        | Jaundice                              | 2       | Discrete            | Nominal                        |
| Epigastric pain                 | Epigastric pain                       | 2       | Discrete            | Nominal                        |
| WBC                             | White blood cell                      | 7425    | Discrete            | Ratio                          |
| RBC                             | red blood cells                       | 4248807 | Discrete            | Ratio                          |
| HGB                             | Hemoglobin                            | 14      | Discrete            | Ratio                          |
| Plat                            | Platelets                             | 112132  | Discrete            | Ratio                          |
| AST 1                           | aspartate transaminase ratio          | 99      | Discrete            | Ratio                          |
| ALT 1                           | alanine transaminase ratio 1 week     | 84      | Discrete            | Ratio                          |
| ALT 4                           | alanine transaminase ratio 12 weeks   | 52      | Discrete            | Ratio                          |
| ALT 12                          | alanine transaminase ratio 4 weeks    | 109     | Discrete            | Ratio                          |
| ALT 24                          | alanine transaminase ratio 24 weeks   | 81      | Discrete            | Ratio                          |
| ALT 36                          | alanine transaminase ratio 36 weeks   | 5       | Discrete            | Ratio                          |
| ALT 48                          | alanine transaminase ratio 48 weeks   | 5       | Discrete            | Ratio                          |
| ALT after 24                    | w alanine transaminase ratio 24 weeks | 5       | Discrete            | Ratio                          |
| RNA Base                        | RNA Base                              | 655330  | Discrete            | Ratio                          |
| RNA 4                           | RNA 4                                 | 634536  | Discrete            | Ratio                          |
| RNA 12                          | RNA 12                                | 288194  | Discrete            | Ratio                          |
| RNA EOT                         | RNA end-of-treatment                  | 5       | Discrete            | Ratio                          |
| RNA EF                          | RNA Elongation Factor                 | 5       | Discrete            | Ratio                          |
| Baseline histological Grading   | Baseline histological Grading         | 13      | Discrete            | Interval                       |
| Baselinehistological staging    | Baselinehistological staging          | 2       | Discrete            | Interval                       |


• Give an account of whether there are data issues (i.e. missing values or corrupted data) and describe them if so.



| Data Set Characteristics: | Attribute Characteristics: | Missing Values: |
| ------------------------- | -------------------------- | --------------- |
| Multivariate              | Integer, Real              | Non             |
• Include basic summary statistics of the attributes.
If your data set contains many similar attributes, you may restrict yourself to
describing a few representative features (apply common sense).

|                               | count | mean     | std        | min      | 25%      | 50%     | 75%     | max     |
| ----------------------------- | ----- | -------- | ---------- | -------- | -------- | ------- | ------- | ------- |
| Age                           | 1385  | 4.63E+01 | 8.781506   | 32       | 39       | 46      | 54      | 61      |
| Gender                        | 1385  | 1.49E+00 | 0.500071   | 1        | 1        | 1       | 2       | 2       |
| BMI                           | 1385  | 2.86E+01 | 4.076215   | 22       | 25       | 29      | 32      | 35      |
| Fever                         | 1385  | 1.52E+00 | 0.499939   | 1        | 1        | 2       | 2       | 2       |
| Nausea/Vomting                | 1385  | 1.50E+00 | 0.500174   | 1        | 1        | 2       | 2       | 2       |
| Headache                      | 1385  | 1.50E+00 | 0.500165   | 1        | 1        | 1       | 2       | 2       |
| Diarrhea                      | 1385  | 1.50E+00 | 0.500174   | 1        | 1        | 2       | 2       | 2       |
| Fatigue&generalized_bone      | 1385  | 1.50E+00 | 0.500179   | 1        | 1.00E+00 | 1       | 2       | 2       |
| Jaundice                      | 1385  | 1.50E+00 | 0.500179   | 1        | 1        | 2       | 2       | 2       |
| Epigastric_pain               | 1385  | 1.50E+00 | 5.00E-01   | 1        | 1        | 2       | 2       | 2       |
| WBC                           | 1385  | 7.53E+03 | 2668.22033 | 2991     | 5219     | 7498    | 9902    | 12101   |
| RBC                           | 1385  | 4.42E+06 | 346357.712 | 3816422  | 4121374  | 4438465 | 4721279 | 5018451 |
| HGB                           | 1385  | 1.26E+01 | 1.713511   | 10       | 11       | 13      | 14      | 15      |
| Plat                          | 1385  | 1.58E+05 | 38794.7856 | 93013    | 124479   | 157916  | 190314  | 226464  |
| AST_1                         | 1385  | 8.28E+01 | 2.60E+01   | 39       | 60       | 83      | 105     | 128     |
| ALT_1                         | 1385  | 8.39E+01 | 2.59E+01   | 39       | 62       | 83      | 106     | 128     |
| ALT4                          | 1385  | 8.34E+01 | 26.52973   | 39       | 61       | 82      | 107     | 128     |
| ALT_12                        | 1385  | 8.35E+01 | 2.61E+01   | 39       | 60       | 84      | 106     | 128     |
| ALT_24                        | 1385  | 8.37E+01 | 2.62E+01   | 39       | 61       | 83      | 107     | 128     |
| ALT_36                        | 1385  | 8.31E+01 | 2.64E+01   | 5        | 61       | 84      | 106     | 128     |
| ALT_48                        | 1385  | 8.36E+01 | 2.62E+01   | 5        | 61       | 83      | 106     | 128     |
| ALT_after_24_w                | 1385  | 3.34E+01 | 7.073569   | 5        | 2.80E+01 | 34      | 40      | 45      |
| RNA_Base                      | 1385  | 5.91E+05 | 3.54E+05   | 11       | 269253   | 593103  | 886791  | 1201086 |
| RNA_4                         | 1385  | 6.01E+05 | 3.62E+05   | 5        | 270893   | 597869  | 909093  | 1201715 |
| RNA_12                        | 1385  | 2.89E+05 | 2.85E+05   | 5        | 5        | 234359  | 524819  | 3731527 |
| RNA_EOT                       | 1385  | 2.88E+05 | 2.65E+05   | 5        | 5        | 251376  | 517806  | 808450  |
| RNA_EF                        | 1385  | 2.91E+05 | 2.68E+05   | 5        | 5        | 244049  | 527864  | 810333  |
| Baseline_histological_Grading | 1385  | 9.76E+00 | 4.023896   | 3.00E+00 | 6        | 10      | 13      | 16      |
| Baselinehistological_staging  | 1385  | 2.54E+00 | 1.12E+00   | 1        | 2        | 3       | 4       | 4       |

## 3. Data visualization and PCA
Touch upon the following subjects, use visualizations when it appears sensible. Keep in mind the ACCENT principles and Tufte’s guidelines when you visualize the data.
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

<img src="https://i.imgur.com/xR8ZXaM.png" width="600" height="400">

<img src="https://i.imgur.com/9eNaAPW.png" width="600" height="400">

<img src="https://i.imgur.com/rPmhtnT.png" width="600" height="400">





Due to the nature of the dataset, with some attributes being continuous ratios and some other being discrete nominal and ordinal, some data manipulation was required before applying PCA. Eight of the twenty-seven attributes have been one-hot encoded, bringing the final dimensionality to thirty-five attributes. After doing so, the data has been standardized - so each column has been transformed to have zero mean and standard deviation equal to one.
The result of the PCA application can be seen in [Figure 1]()

![Figure 1](https://i.imgur.com/f57YRRp.png =20x20)


The result of the component analysis is that twenty-three component needs to be used to retain at least 90\% of the variance in the data. Reducing the number of components - and hence dimensionality of the data - will clearly cause an almost constant increase of information loss.
Dimensionality reduction does not seem to suit very well the chosen data. The cross-correlation among features does indeed confirm this belief.

![Figure 2](https://i.imgur.com/MGGcqi4.png =20x20)

In [Figure 2]() it is possible to see that all feature columns are uncorrelated to each other. The only exception being the binary one-hot encoded features (pair-wise negative correlations). The only slight correlation (40\%) is seen in the last three features. Applying PCA to only does column should not be that beneficial, since it would only reduce the dimensionality of two units.




## 4. Discussion of what we have learnt
Summarize here the most important things you have learned about the data and give also your thoughts on whether your primary machine learning aim appears to be feasible based on your visualization.

Even columns that one would believe to be possibly correlated, have no correlation at all. (RNA measurement and similar)

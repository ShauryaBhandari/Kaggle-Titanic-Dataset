# Kaggle-Titanic-Dataset
Solution to Kaggle's Titanic Dataset using various ML algorithms 
The goal is to predict the survival or the death of a given passenger based on 12 feature such as sex, age, etc.  


### Problem Statement:
This is a binary classification to detect the survival or death of a passenger onboard the Titanic. 
The model predicts predicts the death or survival of a new passenger. 

### Introduction: 
RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early hours of 15 April 1912, after colliding with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard, and more than 1,500 died, making it one of the deadliest commercial peacetime maritime disasters in modern history. RMS Titanic was the largest ship afloat at the time it entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. 
It was built by the Harland and Wolff shipyard in Belfast. 
Thomas Andrews, her architect, died in the disaster.

### Dataset:
The Titanic dataset can be downloaded from the Kaggle website which provides separate train and test data. 
The train data consists of 891 entries and the test data 418 entries. It has a total of 12 features. 

### Exploratory data analysis:

As in different data projects, we'll first start diving into the data and build up our first intuitions.
In this section, we'll be doing four things.

Data extraction: We'll load the dataset and have a first look at it.

Cleaning: We'll fill in missing values.

Plotting: We'll create some interesting charts that'll (hopefully) spot correlations and hidden insights out of the data.

Assumptions: We'll formulate hypotheses from the charts.

### Modelling:

In this part, we use our knowledge of the passengers based on the features we created and then build a statistical model. 
You can think of this model as a box that crunches the information of any new passenger and decides whether or not he survives.
A variety of ML algorithms were used and models created for SVM, KNN, Logistic Regression etc.
Steps:
1) Break the combined dataset in train set and test set.

2) Use the train set to build a predictive model.

3) Evaluate the model using the train set.

4) Test the model using the test set and generate and output file and import it to another csv file (attached above).

5) Compare the performance of various models and choose the best fit. In this case, Decision Tree model performed the best followed by KNN and SVM.

Thank you for visiting! 


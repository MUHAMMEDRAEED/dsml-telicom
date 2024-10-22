# Title: Churn Prediction using Machine Learning

## Author: MUHAMMED RAEED MK

### Date : 01/10/2024

### Introduction

This project aims to develop a machine learning model to predict customer churn for a telecommunications company. The model uses a combination of exploratory data analysis, feature engineering, and hyperparameter tuning to achieve high accuracy in predicting churn.

### Code Structure

The code is organized into the following sections:

   - Step 1: Load and Explore Data
   - Step 2: EDA (Exploratory Data Analysis)
   - Step 3: Data Preprocessing
   - Step 4: Feature Engineering (One-Hot Encoding for categorical variables)
   - Step 5: Split the Data
   - Step 6: Hyperparameter Tuning with Random Forest
   - Step 7: Model Evaluation
   - Step 8: Save the Best Model
   
### Features

   - Data Loading and Exploration: The model loads a CSV file containing customer data and performs exploratory data analysis to understand the distribution of churn and correlations between features.
   - Data Preprocessing: The model preprocesses the data by handling missing values, converting categorical variables, and scaling numerical features.
   - Feature Engineering: The model applies one-hot encoding to categorical variables and scales numerical features using a column transformer.
   - Hyperparameter Tuning: The model uses GridSearchCV to tune hyperparameters for a random forest classifier and selects the best parameters for optimal performance.
   - Model Evaluation: The model evaluates its performance using classification report and accuracy score on the test set.
   - Model Saving: The model saves the best-performing model to a pickle file for future use.
  
### Overview

This project focuses on predicting customer churn for a telecom company using machine learning techniques. Customer churn, the event of customers discontinuing a service, is a significant business challenge. Accurately predicting churn enables businesses to implement proactive strategies to retain at-risk customers and enhance profitability.

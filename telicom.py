#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Load and Explore Data
file_path = 'telico.csv'  
data = pd.read_csv(file_path)

# View the first few rows
print(data.head())

# Check for missing values and data types
print(data.info())
print(data.isnull().sum())

# Step 2: EDA (Exploratory Data Analysis)
# Visualize Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Visualize correlation heatmap (for numerical columns only)
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Check for unique values in categorical columns
print(data.nunique())

# Step 3: Data Preprocessing
# Remove unnecessary columns (e.g., customer ID)
data.drop(['customerID'], axis=1, inplace=True)

# Handle missing data in 'TotalCharges'
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Convert 'Churn' to binary 0 or 1
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split columns into numerical and categorical
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']

# Step 4: Feature Engineering (One-Hot Encoding for categorical variables)
# Apply OneHotEncoder to categorical columns and scale numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Step 5: Split the Data
X = data.drop('Churn', axis=1)  # Features
y = data['Churn']  # Target

# Split into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Hyperparameter Tuning with Random Forest
# Build a pipeline that includes preprocessing and the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Define the parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__bootstrap': [True, False]
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Step 7: Model Evaluation
# Make predictions on the test set
y_pred = grid_search.predict(X_test)

# Show classification results
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 8: Save the Best Model
joblib.dump(grid_search.best_estimator_, 'best_churn_prediction_model.pkl')


# In[ ]:





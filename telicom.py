#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Load and Explore Data
file_path = 'telico.csv'
data = pd.read_csv(file_path)

print(data.head())
print(data.info())
print(data.isnull().sum())

# Step 2: EDA (Exploratory Data Analysis)
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print(data.nunique())

# Step 3: Data Preprocessing
data.drop(['customerID'], axis=1, inplace=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']

# Step 4: Feature Engineering
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Step 5: Split the Data
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Hyperparameter Tuning
classifiers_and_params = {
    'Logistic Regression': (LogisticRegression(max_iter=500), {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2']
    }),
    'Support Vector Classifier': (SVC(), {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10]
    }),
    'MLP Classifier': (MLPClassifier(max_iter=500), {
        'classifier__hidden_layer_sizes': [(50, 50), (100,)],
        'classifier__activation': ['tanh', 'relu']
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, 30]
    })
}

# Dictionary to store the best estimators
best_estimators = {}

for name, (classifier, param_grid) in classifiers_and_params.items():
    print(f"\nTuning {name}...")
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('classifier', classifier)])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    best_estimators[name] = grid_search.best_estimator_

# Step 7: Model Evaluation
for name, model in best_estimators.items():
    print(f"\nEvaluating {name}...")
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{name} Accuracy Score: {accuracy_score(y_test, y_pred)}")

# Step 8: Save the Best Model (Random Forest in this example)
joblib.dump(best_estimators['Random Forest'], 'best_churn_prediction_model.pkl')

# In[ ]:





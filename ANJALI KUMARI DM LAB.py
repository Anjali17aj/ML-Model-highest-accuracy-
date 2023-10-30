#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv(r"C:\Users\Language-Lab\Desktop\datasets\data.csv")

print(df)




# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

# Load the data from Excel
data = pd.read_excel("data.xlsx")

# Handle missing data
# Impute missing values with "mean" for numeric columns and "most_frequent" for non-numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns
non_numeric_cols = data.select_dtypes(exclude=['number']).columns

imputer_numeric = SimpleImputer(strategy="mean")
imputer_non_numeric = SimpleImputer(strategy="most_frequent")

data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])
data[non_numeric_cols] = imputer_non_numeric.fit_transform(data[non_numeric_cols])

# Encode categorical variables
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['is_smoking'] = label_encoder.fit_transform(data['is_smoking'])

# Split data into features and target variable
X = data.drop(columns=['id', 'TenYearCHD'])
y = data['TenYearCHD']

# Oversample the minority class
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train the model (Random Forest Classifier)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate accuracy and recall
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print or use accuracy and recall as needed
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')


# In[ ]:





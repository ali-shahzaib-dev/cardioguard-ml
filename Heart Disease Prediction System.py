#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'heart.csv')
heart_data = pd.read_csv(csv_path)

print("--- First 5 Rows ---")
print(heart_data.head())

print("\n--- Last 5 Rows ---")
print(heart_data.tail())

print("\n--- Target Value Counts ---")
print(heart_data.target.value_counts())

# Target distribution plot
sns.countplot(x='target', hue='target', data=heart_data, palette='bwr', legend=False)
plt.title('Heart Disease Target Distribution (0 = No Disease, 1 = Disease)')
plt.savefig(os.path.join(BASE_DIR, 'targetDistribution.png'))
plt.close()

# Heart Disease Frequency for Ages plot
pd.crosstab(heart_data.age, heart_data.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(os.path.join(BASE_DIR, 'heartDiseaseAndAges.png'))
plt.close()

countNoDisease = len(heart_data[heart_data.target == 0])
countHaveDisease = len(heart_data[heart_data.target == 1])
print("\nPercentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(heart_data.target)) * 100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(heart_data.target)) * 100)))

print("\n--- Dataset Info ---")
heart_data.info()

print("\n--- Checking for Missing Values ---")
print(heart_data.isnull().sum())

print("\n--- Statistical Summary ---")
print(heart_data.describe())

# Applying Feature Engineering
X = heart_data.drop(columns='target')
Y = heart_data['target']

print("\n--- Features (X) ---")
print(X)

print("\n--- Target (Y) ---")
print(Y)

# Splitting the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# test_size = 0.2 means 20% of data will be used for testing and 80% for training

print("\n--- Data Shapes ---")
print("X shapes (Total, Train, Test):", X.shape, X_train.shape, X_test.shape)
print("Y shapes (Total, Train, Test):", Y.shape, Y_train.shape, Y_test.shape)

# Model Building (Logistic Regression)
accuracies = {}
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# Checking accuracy on Training and Test Data
train_acc = model.score(X_train, Y_train) * 100
acc = model.score(X_test, Y_test) * 100
accuracies['Logistic Regression'] = acc
print("\nTraining Accuracy: {:.2f}%".format(train_acc))
print("Test Accuracy: {:.2f}%".format(acc))

# Building a Predictive System (Sample Patient Test)
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_df = pd.DataFrame([input_data], columns=X.columns)
prediction = model.predict(input_df)

print("\n--- Sample Patient Prediction ---")
print("Input Data:", input_data)
if prediction[0] == 0:
    print("Result: The Person does NOT have a Heart Disease.")
else:
    print("Result: The Person HAS Heart Disease.")



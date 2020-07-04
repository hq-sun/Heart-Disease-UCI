#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:19:16 2020

@author: Heqing Sun
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Get current working directory
os.getcwd()

# Read cleaned csv file from Step 1 - data cleaning
df = pd.read_csv("./data/clean/data_clean.csv")
# df_backup = df.copy()
## 303 obs, 20 vars

# Split training, test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = 0.2, random_state = 123)

# =============================================================================
# Random Forest
# =============================================================================
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
model_rf = RandomForestClassifier(max_depth=5)
model_rf.fit(X_train, y_train)

# For test data
y_predict = model_rf.predict(X_test)
y_pred_proba = model_rf.predict_proba(X_test)[:, 1]

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_predict)
confusion_matrix

# Different metrics
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: %f' % accuracy)
## Accuracy: 0.786885

# precision tp / (tp + fp)
precision = precision_score(y_test, y_predict)
print('Precision: %f' % precision)
## Precision: 0.750000

# recall: tp / (tp + fn)
recall = recall_score(y_test, y_predict)
print('Recall: %f' % recall)
## Recall: 0.870968

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_predict)
print('F1 score: %f' % f1)
## F1 score: 0.805970

# Plot ROC Curve (Receiver Operator Curve)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC Curve for Random Forest Model')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

auc(fpr, tpr)
## AUC: 0.8838709677419355

# Feature Importance - SHAP
# SHAP (SHapley Additive exPlanations) by Lundberg and Lee (2016)41 is a method to explain individual predictions. SHAP is based on the game theoretically optimal Shapley Values.
import shap
explainer = shap.TreeExplainer(model_rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")
## The number of major vessels is the most important one based on shap

shap.summary_plot(shap_values[1], X_test)

# =============================================================================
# Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# For test data
y_predict = model_rf.predict(X_test)
y_pred_proba = model_rf.predict_proba(X_test)[:, 1]

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_predict)
confusion_matrix

# Different metrics
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy: %f' % accuracy)
## Accuracy: 0.803279

# precision tp / (tp + fp)
precision = precision_score(y_test, y_predict)
print('Precision: %f' % precision)
## Precision: 0.787879

# recall: tp / (tp + fn)
recall = recall_score(y_test, y_predict)
print('Recall: %f' % recall)
## Recall: 0.838710

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_predict)
print('F1 score: %f' % f1)
## F1 score: 0.812500

# Plot ROC Curve (Receiver Operator Curve)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC Curve for Logistic Regression Model')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

auc(fpr, tpr)
## AUC: 0.8903225806451612

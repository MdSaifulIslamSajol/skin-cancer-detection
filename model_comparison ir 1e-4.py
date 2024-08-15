#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 05:32:33 2024

@author: saiful
"""

import matplotlib.pyplot as plt
import pandas as pd

# Updated data for each model
models = [
    "ViTForImageClassification2", "ConvNextV2ForImageClassification", "Swinv2ForImageClassification",
    "CvtForImageClassification", "EfficientFormerForImageClassification", "PvtV2ForImageClassification",
    "MobileViTV2ForImageClassification", "resnet", "vgg", "densenet", "googlenet", "efficientnet", "mobilenet"
]

test_accuracy = [
    89.8459, 92.6564, 91.2965, 91.7498, 91.2965, 90.8432, 91.8404,
    91.5684, 91.4778, 91.5684, 91.7498, 92.4751, 91.2058
]

precision = [
    0.7576, 0.8704, 0.7935, 0.7327, 0.7371, 0.8098, 0.7994,
    0.8014, 0.7820, 0.7924, 0.7718, 0.7906, 0.7509
]

recall = [
    0.7046, 0.7604, 0.7066, 0.7442, 0.7139, 0.7006, 0.7403,
    0.7546, 0.6511, 0.6840, 0.7051, 0.7711, 0.7064
]

f1_score = [
    0.7222, 0.8015, 0.7453, 0.7239, 0.7198, 0.7365, 0.7620,
    0.7708, 0.6992, 0.7207, 0.7331, 0.7744, 0.7149
]

# Define standard names for each model
standard_names = [
    "ViT", "ConvNextV2", "Swinv2", "Cvt", "EfficientFormer", "PvtV2",
    "MobileViTV2", "ResNet", "VGG", "DenseNet", "GoogleNet", "EfficientNet", "MobileNet"
]

# Creating a DataFrame with standard names
df_standard = pd.DataFrame({
    'Model': standard_names,
    'Test Accuracy': test_accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score
})

# Sorting DataFrame by each metric
df_accuracy_sorted_standard = df_standard.sort_values(by='Test Accuracy', ascending=False)
df_precision_sorted_standard = df_standard.sort_values(by='Precision', ascending=False)
df_recall_sorted_standard = df_standard.sort_values(by='Recall', ascending=False)
df_f1_sorted_standard = df_standard.sort_values(by='F1 Score', ascending=False)

# Define colors for each model
model_colors = {model: color for model, color in zip(standard_names, plt.cm.get_cmap('tab20', len(models)).colors)}

# Plotting with subplots 2x2 and adjusted size, vertical bars
fig, axs = plt.subplots(2, 2, figsize=(16, 9))

# Test Accuracy
axs[0, 0].bar(df_accuracy_sorted_standard['Model'], df_accuracy_sorted_standard['Test Accuracy'], color=[model_colors[model] for model in df_accuracy_sorted_standard['Model']])
axs[0, 0].set_title('Test Accuracy Comparison')
axs[0, 0].set_xlabel('Models')
axs[0, 0].set_ylabel('Test Accuracy')
axs[0, 0].set_ylim(50, 100)  # Setting y-axis limits
axs[0, 0].tick_params(axis='x', rotation=45)

# Precision
axs[0, 1].bar(df_precision_sorted_standard['Model'], df_precision_sorted_standard['Precision'], color=[model_colors[model] for model in df_precision_sorted_standard['Model']])
axs[0, 1].set_title('Precision Comparison')
axs[0, 1].set_xlabel('Models')
axs[0, 1].set_ylabel('Precision')
axs[0, 1].set_ylim(0.50, 1)  # Setting y-axis limits
axs[0, 1].tick_params(axis='x', rotation=45)

# Recall
axs[1, 0].bar(df_recall_sorted_standard['Model'], df_recall_sorted_standard['Recall'], color=[model_colors[model] for model in df_recall_sorted_standard['Model']])
axs[1, 0].set_title('Recall Comparison')
axs[1, 0].set_xlabel('Models')
axs[1, 0].set_ylabel('Recall')
axs[1, 0].set_ylim(0.50, 1)  # Setting y-axis limits
axs[1, 0].tick_params(axis='x', rotation=45)

# F1 Score
axs[1, 1].bar(df_f1_sorted_standard['Model'], df_f1_sorted_standard['F1 Score'], color=[model_colors[model] for model in df_f1_sorted_standard['Model']])
axs[1, 1].set_title('F1 Score Comparison')
axs[1, 1].set_xlabel('Models')
axs[1, 1].set_ylabel('F1 Score')
axs[1, 1].set_ylim(0.50, 1)  # Setting y-axis limits
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison_lr1e-4.png', dpi=400)  # Saving the figure with 400 DPI
plt.show()

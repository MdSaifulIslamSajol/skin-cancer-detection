import matplotlib.pyplot as plt
import pandas as pd

# Data for each model
models = [
    "ViTForImageClassification", "ConvNextV2ForImageClassification", "Swinv2ForImageClassification",
    "CvtForImageClassification", "EfficientFormerForImageClassification", "PvtV2ForImageClassification",
    "MobileViTV2ForImageClassification", "resnet", "vgg", "densenet", "googlenet", "efficientnet", "mobilenet"
]

test_accuracy = [
    92.6564, 93.2004, 92.7471, 92.2031, 91.5684, 91.7498, 90.7525,
    91.6591, 91.5684, 91.8404, 90.1179, 91.6591, 91.1151
]

precision = [
    0.8135, 0.8569, 0.8493, 0.7485, 0.7653, 0.7714, 0.7734,
    0.8005, 0.7654, 0.7767, 0.7686, 0.7986, 0.7952
]

recall = [
    0.7452, 0.7420, 0.7524, 0.7565, 0.7379, 0.7426, 0.7037,
    0.7006, 0.7492, 0.7406, 0.6901, 0.7774, 0.7428
]

f1_score = [
    0.7727, 0.7884, 0.7921, 0.7493, 0.7453, 0.7515, 0.7279,
    0.7396, 0.7504, 0.7540, 0.7202, 0.7835, 0.7623
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
plt.savefig('model_comparison_lr1e-5.png', dpi=400)  # Saving the figure with 400 DPI
plt.show()

#%%

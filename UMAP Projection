#UMAP PROJECTION

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Remove outliers using IQR (Interquartile Range)
def remove_outliers_iqr(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Remove outliers using Z-score
def remove_outliers_zscore(df, columns, threshold=3):
    z_scores = np.abs(zscore(df[columns], nan_policy='omit'))
    return df[(z_scores < threshold).all(axis=1)]


# Remove rows with null values
features_cleaned = features.dropna()

# 1. Remove outliers using IQR
features_no_outliers_iqr = remove_outliers_iqr(features_cleaned, features_cleaned.columns)

# 2. Remove outliers using Z-score
features_no_outliers_zscore = remove_outliers_zscore(features_cleaned, features_cleaned.columns)

# 3. No outlier removal (just cleaned from NaNs)
features_no_outlier_removal = features_cleaned

# Apply UMAP to each version of the data
umap_model_iqr = umap.UMAP(n_components=2, random_state=42)
umap_results_iqr = umap_model_iqr.fit_transform(features_no_outliers_iqr)

umap_model_zscore = umap.UMAP(n_components=2, random_state=42)
umap_results_zscore = umap_model_zscore.fit_transform(features_no_outliers_zscore)

umap_model_cleaned = umap.UMAP(n_components=2, random_state=42)
umap_results_cleaned = umap_model_cleaned.fit_transform(features_no_outlier_removal)

# Plot the UMAP results
plt.figure(figsize=(18, 5))

# IQR
plt.subplot(1, 3, 1)
plt.scatter(umap_results_iqr[:, 0], umap_results_iqr[:, 1],
            c=df['label'][features_no_outliers_iqr.index], cmap='coolwarm', s=10)
plt.colorbar(label='Label')
plt.title('UMAP (IQR-based Outlier Removal)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Z-score
plt.subplot(1, 3, 2)
plt.scatter(umap_results_zscore[:, 0], umap_results_zscore[:, 1],
            c=df['label'][features_no_outliers_zscore.index], cmap='coolwarm', s=10)
plt.colorbar(label='Label')
plt.title('UMAP (Z-score-based Outlier Removal)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# No outlier removal
plt.subplot(1, 3, 3)
plt.scatter(umap_results_cleaned[:, 0], umap_results_cleaned[:, 1],
            c=df['label'][features_no_outlier_removal.index], cmap='coolwarm', s=10)
plt.colorbar(label='Label')
plt.title('UMAP (No Outlier Removal)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

plt.tight_layout()
plt.show()

# PHASE 1: Dataset Preprocessing

# --- Module 1: Dataset loading ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("/content/muscle_atrophy_synthetic_data.csv")

# Display basic info
print("Dataset Info:\n")
df.info()
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# --- Module 2: Data Visualization ---
sns.set_style("whitegrid")

features = ["muscle_area", "muscle_density", "muscle_texture",
            "fiber_integrity", "area_density_interaction", "age_squared",
            "density_over_age", "muscle_area_6mo_ago", "muscle_area_change","label"]

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(features):
    row, col = divmod(i, 5)
    sns.histplot(df[feature], bins=50, kde=True, ax=axes[row, col])
    axes[row, col].set_title(f"Distribution of {feature}")
    axes[row, col].set_xlabel("Value")
    axes[row, col].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Module 3: Outlier Removal ---
def remove_outliers_iqr(data, columns):
    mask = pd.Series(True, index=data.index)
    outlier_counts = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        col_mask = (data[col] >= lower) & (data[col] <= upper)
        outlier_counts[col] = (~col_mask).sum()
        mask &= col_mask
    cleaned_data = data[mask].copy()
    return cleaned_data, outlier_counts

outlier_columns = features + ["age"]
df_cleaned, outlier_counts = remove_outliers_iqr(df, outlier_columns)

# Retain the label column after cleaning
df_cleaned["label"] = df.loc[df_cleaned.index, "label"]
df_cleaned.to_csv("muscle_atrophy_synthetic_data_cleaned.csv", index=False)

print(f"Original dataset size: {df.shape}")
print(f"Cleaned dataset size: {df_cleaned.shape}")
print("\nOutlier counts per feature:")
for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers")

# --- Module 4: Feature Scaling ---
scaler = StandardScaler()
df_cleaned[features] = scaler.fit_transform(df_cleaned[features])

# --- Module 5: Violin Plots ---
columns = ['age'] + features
num_columns = len(columns)
num_rows = (num_columns + 2) // 3

plt.figure(figsize=(15, 5 * num_rows))
for i, col in enumerate(columns, 1):
    plt.subplot(num_rows, 3, i)
    sns.violinplot(y=df_cleaned[col], color="skyblue")
    plt.title(f"Violin Plot: {col}")
plt.tight_layout()
plt.show()


# --- Module 6: Correlation Matrix ---
plt.figure(figsize=(12, 10))
sns.heatmap(df_cleaned[features + ["age"]].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# --- Module 7: Kolmogorov-Smirnov Test (Single Feature Demo) ---
X_train, X_test, y_train, y_test = train_test_split(
    df_cleaned.drop('label', axis=1), df_cleaned['label'], test_size=0.2, random_state=42
)

feature = "muscle_texture"
data_train = X_train[feature]
data_test = X_test[feature]

ks_statistic, p_value = ks_2samp(data_train, data_test)
print(f"KS Statistic for '{feature}': {ks_statistic:.4f}, P-value: {p_value:.4f}")

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

x_train, y_train_ecdf = ecdf(data_train)
x_test, y_test_ecdf = ecdf(data_test)
diff = np.abs(y_train_ecdf - np.interp(x_train, x_test, y_test_ecdf))
idx = np.argmax(diff)
ks_x = x_train[idx]
ks_y_train = y_train_ecdf[idx]
ks_y_test = np.interp(ks_x, x_test, y_test_ecdf)

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train_ecdf, label="Train ECDF", color="blue")
plt.plot(x_test, y_test_ecdf, label="Test ECDF", color="orange")
plt.vlines(ks_x, ks_y_train, ks_y_test, color="red", linestyle="--", label=f"KS = {ks_statistic:.3f}")
plt.title(f"KS Test ECDF: {feature}")
plt.xlabel(feature)
plt.ylabel("ECDF")
plt.legend()
plt.grid(True)
plt.show()

# --- Module 9: KS Test on All Features ---
ks_results = {}
for feature in X_train.columns:
    stat, p_val = ks_2samp(X_train[feature], X_test[feature])
    ks_results[feature] = {'KS Statistic': stat, 'P-value': p_val}

print("\nKolmogorov-Smirnov Test Results (Train vs Test):\n")
for feat, result in ks_results.items():
    significance = "Significant" if result["P-value"] < 0.05 else "Not Significant"
    print(f"{feat:<25} | KS: {result['KS Statistic']:.4f} | P: {result['P-value']:.4f} | {significance}")

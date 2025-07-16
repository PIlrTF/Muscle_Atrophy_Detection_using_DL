#PHASE 2: Model Selection for Classification
!pip install keras-tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

# Define features and target variable
X = df_cleaned.drop(columns=["label"])  # Drop target variable 'label'
y = df_cleaned["label"]  # Target variable for classification

# Convert continuous target into binary labels (classification)
y = (y > 0).astype(int)  # Threshold: values above 0 are class 1, else class 0

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Address class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define model building function for classification
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(hp.Int('units_1', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(layers.Dense(hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0001, 0.00001])),
        loss='binary_crossentropy',  # Using binary crossentropy for binary classification
        metrics=['accuracy']  # Accuracy as a metric
    )
    return model

# Hyperparameter tuner using Bayesian Optimization
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',  # Use validation accuracy as objective for classification
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='muscle_atrophy_bayes'
)

# Run hyperparameter tuning
tuner.search(X_train_res, y_train_res, epochs=20, validation_split=0.2, verbose=1)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters: {best_hps.values}")

# Train model with best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the model on the correct training set
history = best_model.fit(X_train_res, y_train_res, validation_split=0.2, epochs=100, batch_size=32)

# Evaluate model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot accuracy and loss (for classification)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Confusion Matrix, Classification Report, and ROC Curve
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np

# Predict probabilities and convert to class labels
y_pred_prob = best_model.predict(X_test).ravel()
y_pred_class = (y_pred_prob > 0.5).astype("int32")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

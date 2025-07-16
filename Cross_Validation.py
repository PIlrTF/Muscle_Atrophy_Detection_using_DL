#CROSS VALIDATION

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Convert to NumPy
X_np = X.to_numpy()
y_np = y.to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

for train_index, test_index in skf.split(X_np, y_np):
    X_train_fold, X_test_fold = X_np[train_index], X_np[test_index]
    y_train_fold, y_test_fold = y_np[train_index], y_np[test_index]

    model = keras.Sequential([
        layers.Dense(best_hps.get('units_1'), activation='relu'),
        layers.Dense(best_hps.get('units_2'), activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0)
    y_pred_fold = (model.predict(X_test_fold) > 0.5).astype("int32")
    acc = accuracy_score(y_test_fold, y_pred_fold)
    cv_accuracies.append(acc)

print(f"Cross-validated accuracies: {cv_accuracies}")
print(f"Mean CV Accuracy: {np.mean(cv_accuracies):.4f}")

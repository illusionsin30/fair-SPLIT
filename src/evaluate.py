# ============================================================
# Evaluation utilities for CART models
# ============================================================
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def evaluate_classification(y_true, y_pred, y_train):
    """Print classification metrics: accuracy and majority-class baseline.

    Args:
        y_true: Ground-truth labels for the test set.
        y_pred: Predicted labels from the model.
        y_train: Training labels (used for majority-class baseline).
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Test accuracy:           {acc:.4f}")

    majority = np.argmax(np.bincount(y_train))
    baseline = accuracy_score(y_true, np.full_like(y_true, majority))
    print(f"Majority-class baseline: {baseline:.4f}")


def evaluate_regression(y_true, y_pred, y_train):
    """Print regression metrics: RMSE, R^2, and mean-prediction baseline.

    Args:
        y_true: Ground-truth target values for the test set.
        y_pred: Predicted values from the model.
        y_train: Training target values (used for mean-prediction baseline).
    """
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"Test RMSE: {rmse:.2f}  R^2: {r2:.4f}")

    baseline_rmse = mean_squared_error(
        y_true, np.full_like(y_true, y_train.mean(), dtype=float)
    ) ** 0.5
    print(f"Mean-prediction baseline RMSE: {baseline_rmse:.2f}")

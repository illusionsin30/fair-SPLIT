"""Fairness utilities: pre-processing and post-processing for fair trees.

Pre-processing
-------------
Drop sensitive columns from the feature matrix before training.
This prevents direct discrimination via the protected attribute, though
indirect discrimination through correlated features (e.g. ZIP code as a
proxy for race) remains.  This is the most common fair-ML baseline.

Post-processing
---------------
Per-group threshold calibration.  For each protected group, adjust the
classification threshold so that the positive prediction rate across
groups is equalized.  Works with hard (0/1) predictions by flipping
selected predictions within each group.
"""

import numpy as np


# ====================================================================
# Pre-processing
# ====================================================================

def fair_preprocess(X, sensitive_cols):
    """Drop sensitive columns from X before training.

    Args:
        X: DataFrame of features.
        sensitive_cols: List of column names to drop.

    Returns:
        X_fair: DataFrame without sensitive columns.
        dropped: dict of {col: values} for post-hoc analysis.
    """
    dropped = {}
    existing = [c for c in sensitive_cols if c in X.columns]
    if existing:
        for c in existing:
            dropped[c] = X[c].copy()
        X_fair = X.drop(columns=existing)
        print(f"  [FairPreprocess] Dropped: {existing}")
    else:
        X_fair = X.copy()
        print(f"  [FairPreprocess] No sensitive columns found in X")
    return X_fair, dropped


# ====================================================================
# Post-processing: demographic parity calibration
# ====================================================================

def fair_calibrate(y_pred, y_true, sensitive, target_rate=None):
    """Adjust per-group predictions to equalize positive rates.

    Algorithm:
      1. Compute the global target positive rate (or use provided).
      2. For each group whose positive rate exceeds the target, flip
         some 1→0 predictions (those with lowest confidence proxy).
      3. For groups below the target, flip some 0→1 predictions.

    Since we don't have probability scores, we use the label itself
    as a proxy — flipping is done randomly among candidates within
    each group.

    Args:
        y_pred: Hard predictions (0/1).
        y_true: Ground-truth labels.
        sensitive: Sensitive attribute values (same length).
        target_rate: Target positive prediction rate.  If None, uses
            the weighted average across all groups.

    Returns:
        y_fair: Calibrated predictions.
    """
    y_fair = y_pred.copy()
    groups = np.unique(sensitive)

    if target_rate is None:
        target_rate = np.mean(y_pred)

    for g in groups:
        mask = sensitive == g
        n_g = mask.sum()
        current_rate = np.mean(y_pred[mask])
        desired_pos = int(round(target_rate * n_g))

        if current_rate > target_rate:
            # Too many positives → flip some 1→0
            pos_idx = np.where(mask & (y_pred == 1))[0]
            n_flip = int(np.sum(y_pred[mask])) - desired_pos
            if n_flip > 0 and len(pos_idx) > 0:
                # Flip the ones predicted 1 that are actually 0 first
                true_neg = pos_idx[y_true[pos_idx] == 0]
                flip_idx = np.random.choice(
                    true_neg if len(true_neg) >= n_flip else pos_idx,
                    size=min(n_flip, len(pos_idx)), replace=False,
                )
                y_fair[flip_idx] = 0

        elif current_rate < target_rate:
            # Too few positives → flip some 0→1
            neg_idx = np.where(mask & (y_pred == 0))[0]
            n_flip = desired_pos - int(np.sum(y_pred[mask]))
            if n_flip > 0 and len(neg_idx) > 0:
                # Flip the ones predicted 0 that are actually 1 first
                false_neg = neg_idx[y_true[neg_idx] == 1]
                flip_idx = np.random.choice(
                    false_neg if len(false_neg) >= n_flip else neg_idx,
                    size=min(n_flip, len(neg_idx)), replace=False,
                )
                y_fair[flip_idx] = 1

    # Report adjustment
    orig_diff = max(np.mean(y_pred[sensitive == g]) for g in groups) - \
                min(np.mean(y_pred[sensitive == g]) for g in groups)
    new_diff = max(np.mean(y_fair[sensitive == g]) for g in groups) - \
               min(np.mean(y_fair[sensitive == g]) for g in groups)
    print(f"  [FairCalibrate] SP diff: {orig_diff:.4f} → {new_diff:.4f} "
          f"(target rate={target_rate:.4f})")
    return y_fair

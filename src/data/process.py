"""Preprocessing utilities to make any dataset SPLIT-compatible.

SPLIT now natively supports multiclass classification (K >= 2 classes).
The only conversion needed is for regression targets → binary.

Target conversion rules
----------------------
* Classification (any number of discrete classes) → pass through unchanged.
* Regression / continuous target → binarize at median.
"""

import numpy as np


def prepare_target(y):
    """Ensure *y* is a classification target (discrete labels).

    Returns (y_out, mask) where:
      y_out : (possibly converted) integer labels
      mask  : boolean array of rows to keep (None = keep all)

    Conversion rules
    ----------------
    * Classification (integer labels, any K >= 2) → pass through.
      Non-{0,1} labels are remapped to contiguous 0..K-1.
    * Regression (float dtype, or integer with > 10 unique values) →
      binarize at the median (>= median → 1, < median → 0).
    """
    unique = np.unique(y)
    n_unique = len(unique)

    # --- classification (binary or multiclass) ---
    is_classification = (
        np.issubdtype(y.dtype, np.integer) and n_unique <= 10
    )
    if is_classification:
        if n_unique >= 2:
            # Remap labels to contiguous 0..K-1 if needed
            mapping = {old: new for new, old in enumerate(sorted(unique))}
            if mapping == {k: k for k in mapping}:
                return y.astype(int), None
            print(f"[process] Remapping class labels: {mapping}")
            return np.array([mapping[v] for v in y], dtype=int), None
        # Single class → can't classify
        raise ValueError(
            f"Target has only 1 unique value ({unique[0]}). "
            "Cannot train a classifier."
        )

    # --- regression → median split ---
    threshold = float(np.median(y))
    print(f"[process] Regression target → binary at median = {threshold:.4f}")
    y_bin = (y >= threshold).astype(int)
    return y_bin, None


def prepare_for_split(X, y):
    """Convert (X, y) to a form suitable for SPLIT training.

    Parameters
    ----------
    X : pandas DataFrame  (SPLIT's binarizer handles continuous features
                           internally via the binarize=True flag).
    y : array-like         (may be multiclass or continuous).

    Returns
    -------
    X : DataFrame  (same rows).
    y : ndarray    (integer class labels, 0..K-1).
    """
    y_out, mask = prepare_target(y)
    if mask is not None:
        X = X.loc[mask].reset_index(drop=True)
    return X, y_out

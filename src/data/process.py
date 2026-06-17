"""Preprocessing utilities to make any dataset SPLIT-compatible.

SPLIT natively supports multiclass classification (K >= 2 classes).
Only regression targets need conversion → binary via median split.

Target conversion rules
----------------------
* Classification (string labels or small-n integer labels) → remap to 0..K-1.
* Regression (float dtype and many unique values) → binarize at median.
"""

import numpy as np


def _is_classification(y, n_unique):
    """Return True if y looks like a classification target.

    String/object dtypes are always classification.
    Integer dtypes with <= 10 unique values are classification.
    Float dtypes are never classification (regression).
    """
    if y.dtype.kind in ("U", "S", "O"):
        return True
    if y.dtype.kind == "i" and n_unique <= 10:
        return True
    return False


def prepare_target(y):
    """Ensure *y* is a classification target (discrete integer labels 0..K-1).

    Returns (y_out, mask) where:
      y_out : integer labels
      mask  : boolean array of rows to keep (None = keep all)

    Conversion rules
    ----------------
    * String/object labels (e.g. "yes"/"no", "<=50K"/">50K") →
      mapped to contiguous 0..K-1 within the dataload first, then
      this function just validates and normalises.
    * Integer labels with 2-10 unique values → pass through, remap to 0..K-1.
    * Float dtype or >10 unique ints → regression, binarize at median.
    """
    unique = np.unique(y)
    n_unique = len(unique)

    if n_unique < 2:
        raise ValueError(
            f"Target has only 1 unique value ({unique[0]}). "
            "Cannot train a classifier."
        )

    # --- classification (strings or small-n ints) ---
    if _is_classification(y, n_unique):
        # Convert to int if string
        y_int = y
        if y.dtype.kind in ("U", "S", "O"):
            # string targets – remap sorted-unique → 0..K-1
            mapping = {lab: i for i, lab in enumerate(sorted(unique))}
            print(f"[process] String target remapping: {mapping}")
            y_int = np.array([mapping[v] for v in y], dtype=int)
        elif y.dtype.kind == "i":
            y_int = y.astype(int)
        else:
            y_int = y.astype(int)

        # Remap to contiguous 0..K-1 if not already
        unique_int = np.unique(y_int)
        if set(unique_int) != set(range(len(unique_int))):
            mapping = {old: new for new, old in enumerate(sorted(unique_int))}
            print(f"[process] Remapping class labels: {mapping}")
            y_int = np.array([mapping[v] for v in y_int], dtype=int)

        return y_int, None

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

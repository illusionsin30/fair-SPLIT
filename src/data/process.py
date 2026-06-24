"""Target preprocessing utilities."""

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
    """Ensure *y* is a discrete classification target."""
    unique = np.unique(y)
    n_unique = len(unique)

    if n_unique < 2:
        raise ValueError(
            f"Target has only 1 unique value ({unique[0]}). "
            "Cannot train a classifier."
        )

    if _is_classification(y, n_unique):
        y_int = y
        if y.dtype.kind in ("U", "S", "O"):
            mapping = {lab: i for i, lab in enumerate(sorted(unique))}
            print(f"[process] String target remapping: {mapping}")
            y_int = np.array([mapping[v] for v in y], dtype=int)
        elif y.dtype.kind == "i":
            y_int = y.astype(int)
        else:
            y_int = y.astype(int)

        unique_int = np.unique(y_int)
        if set(unique_int) != set(range(len(unique_int))):
            mapping = {old: new for new, old in enumerate(sorted(unique_int))}
            print(f"[process] Remapping class labels: {mapping}")
            y_int = np.array([mapping[v] for v in y_int], dtype=int)

        return y_int, None

    threshold = float(np.median(y))
    print(f"[process] Regression target -> binary at median = {threshold:.4f}")
    y_bin = (y >= threshold).astype(int)
    return y_bin, None


def prepare_for_split(X, y):
    """Convert (X, y) to a form suitable for training."""
    y_out, mask = prepare_target(y)
    if mask is not None:
        X = X.loc[mask].reset_index(drop=True)
    return X, y_out

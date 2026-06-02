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


# ============================================================
# Fairness metrics
# ============================================================

def evaluate_fairness(y_true, y_pred, sensitive, name="sensitive"):
    """Print group-level accuracy and fairness metrics.

    Args:
        y_true: Ground-truth labels (binary 0/1).
        y_pred: Predicted labels (binary 0/1).
        sensitive: Array of sensitive-attribute values (e.g. sex, race).
        name: Human-readable name for the sensitive attribute.

    Metrics reported:
      - Per-group accuracy
      - Positive prediction rate  P(ŷ = 1 | group)
      - Statistical parity difference (max - min positive rate)
      - Equal opportunity difference  (TPR gap)
      - Disparate impact ratio  (min_rate / max_rate)
    """
    groups = sorted(np.unique(sensitive))
    n_groups = len(groups)
    if n_groups < 2:
        print(f"  [Fairness] {name}: only 1 group — fairness N/A")
        return

    print(f"\nFairness — {name}:")
    print(f"{'Group':>16s}  {'n':>6s}  {'Acc':>8s}  {'PosRate':>8s}")
    print("-" * 48)

    rates = {}
    tprs = {}
    min_sample = 50  # skip groups below this size for fairness metrics
    small_groups = []
    for g in groups:
        mask = sensitive == g
        n = mask.sum()
        acc = accuracy_score(y_true[mask], y_pred[mask])
        pos_rate = np.mean(y_pred[mask])
        pos_mask = y_true[mask] == 1
        tpr = (np.mean(y_pred[mask][pos_mask])
               if pos_mask.sum() > 0 else float("nan"))
        flag = " (!)" if n < min_sample else ""
        if n < min_sample:
            small_groups.append((str(g), n))
        rates[g] = pos_rate
        tprs[g] = tpr
        print(f"  {str(g):>16s}  {n:>6d}  {acc:>8.4f}  {pos_rate:>8.4f}{flag}")

    print("-" * 48)

    # Exclude small groups for fairness metrics
    large_rates = {g: v for g, v in rates.items()
                   if (sensitive == g).sum() >= min_sample}
    if small_groups:
        print(f"  (!) n<{min_sample}: {', '.join(f'{g}(n={n})' for g, n in small_groups)}")

    if len(large_rates) < 2:
        print(f"  Too few large groups for fairness comparison.")
        return

    # Statistical parity difference
    spd = max(large_rates.values()) - min(large_rates.values())
    print(f"  Statistical parity diff:  {spd:.4f}  (max - min pos rate)")

    # Disparate impact
    max_rate = max(large_rates.values())
    min_rate = min(large_rates.values())
    di = min_rate / max_rate if max_rate > 0 else float("nan")
    print(f"  Disparate impact ratio:   {di:.4f}  (<0.8 usually unfair)")

    # Equal opportunity difference
    large_tprs = {g: v for g, v in tprs.items()
                  if not np.isnan(v) and (sensitive == g).sum() >= min_sample}
    if len(large_tprs) >= 2:
        eod = max(large_tprs.values()) - min(large_tprs.values())
        print(f"  Equal opportunity diff:   {eod:.4f}  (TPR gap)")

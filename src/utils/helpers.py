"""Utility helpers for tree inspection and conversion."""

import numpy as np

from .nodes import SPLITLeaf, SPLITNode


def num_leaves(tree) -> int:
    """Return the number of leaves in a SPLIT tree."""
    if isinstance(tree, SPLITLeaf):
        return 1
    if isinstance(tree, SPLITNode):
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)
    return 0


def tree_to_dict(node, classes=None) -> dict:
    """Convert a SPLIT tree (SPLITNode/SPLITLeaf) into a nested dict.

    Args:
        node: Root of the tree (SPLITNode or SPLITLeaf).
        classes: Optional class label array used to map prediction indices
                 to actual class values. If None, raw prediction index is used.

    Returns:
        A dictionary with keys "feature"/"true"/"false" for internal nodes
        or "prediction" for leaves.
    """
    if isinstance(node, SPLITLeaf):
        pred = node.prediction
        if classes is not None:
            pred = classes[pred]
        return {"prediction": pred, "loss": node.loss}
    return {
        "feature": node.feature,
        "true": tree_to_dict(node.left_child, classes),
        "false": tree_to_dict(node.right_child, classes),
    }


def tree_to_sklearn(tree, feature_names, classes):
    """Convert a SPLIT tree to a scikit-learn-compatible tree dict.

    This facilitates integration with sklearn's tree visualization utilities.

    Args:
        tree: Root of the SPLIT tree.
        feature_names: List of feature name strings.
        classes: Class label array.

    Returns:
        A dict representation compatible with sklearn.tree.export_text style.
    """
    return tree_to_dict(tree, classes)


def are_trees_equal(tree1, tree2) -> bool:
    """Check structural equality of two SPLIT trees."""
    if isinstance(tree1, SPLITLeaf) and isinstance(tree2, SPLITLeaf):
        return tree1.prediction == tree2.prediction
    if isinstance(tree1, SPLITNode) and isinstance(tree2, SPLITNode):
        return (
            tree1.feature == tree2.feature
            and are_trees_equal(tree1.left_child, tree2.left_child)
            and are_trees_equal(tree1.right_child, tree2.right_child)
        )
    return False


def predict_sample(x, node, classes):
    """Recursively predict a single sample through a SPLIT tree.

    Args:
        x: 1-D numpy array of binary features.
        node: Current tree node (SPLITNode or SPLITLeaf).
        classes: Class label array for mapping prediction index to value.

    Returns:
        The predicted class label.
    """
    if isinstance(node, SPLITLeaf):
        return classes[node.prediction]
    if x[node.feature]:
        return predict_sample(x, node.left_child, classes)
    return predict_sample(x, node.right_child, classes)


def used_split_features(tree, feature_names=None):
    """Return feature names in tree split order (root first, level by level).

    Works on SPLIT trees (SPLITNode/SPLITLeaf) and CART trees (Node).

    Args:
        tree: Root of the tree.
        feature_names: Optional list mapping integer feature indices → names.

    Returns:
        List of feature identifiers in split order (duplicates allowed if a
        feature is used multiple times — common for binarized features from
        the same original column).
    """
    feats = _collect_feats_ordered(tree)
    if feature_names is not None:
        named = []
        for f in feats:
            if isinstance(f, str):
                named.append(f)
            elif 0 <= int(f) < len(feature_names):
                named.append(feature_names[int(f)])
            else:
                named.append(str(f))
        return named
    return [str(f) for f in feats]


def _collect_feats_ordered(node):
    """BFS-ordered list of feature identifiers from tree nodes."""
    if node is None:
        return []
    result = []
    # BFS queue: (node, level)
    from collections import deque
    queue = deque()
    queue.append(node)
    while queue:
        n = queue.popleft()
        # Check SPLIT-style internal node
        if hasattr(n, "left_child") and n.left_child is not None:
            result.append(n.feature)
            queue.append(n.left_child)
            queue.append(n.right_child)
        # Check CART-style internal node
        elif hasattr(n, "left") and n.left is not None:
            result.append(n.feature)
            queue.append(n.left)
            queue.append(n.right)
    return result


def predict_batch(X, tree, classes):
    """Predict a batch of samples through a SPLIT tree.

    Args:
        X: 2-D numpy array of binary features (n_samples, n_features).
        tree: Root of the SPLIT tree.
        classes: Class label array.

    Returns:
        1-D numpy array of predicted class labels.
    """
    return np.array([predict_sample(X[i], tree, classes) for i in range(len(X))])

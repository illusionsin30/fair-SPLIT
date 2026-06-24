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
    """Convert a tree into a nested dict."""
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
    """Convert a tree to a scikit-learn-compatible dict."""
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
    """Recursively predict a single sample."""
    if isinstance(node, SPLITLeaf):
        return classes[node.prediction]
    if x[node.feature]:
        return predict_sample(x, node.left_child, classes)
    return predict_sample(x, node.right_child, classes)


def used_split_features(tree, feature_names=None):
    """Return feature names in split order."""
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
    """Return a BFS-ordered list of feature identifiers."""
    if node is None:
        return []
    result = []
    from collections import deque
    queue = deque()
    queue.append(node)
    while queue:
        n = queue.popleft()
        if hasattr(n, "left_child") and n.left_child is not None:
            result.append(n.feature)
            queue.append(n.left_child)
            queue.append(n.right_child)
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


def split_leaf_id_sample(x, node, path=()):
    """Return a stable path-based leaf id for one SPLIT sample.

    Args:
        x: 1-D numpy array of binary features.
        node: Current SPLIT tree node.
        path: Tuple of branch labels accumulated from the root.

    Returns:
        Tuple representing the path to the reached leaf.
    """
    if isinstance(node, SPLITLeaf):
        return path
    if x[node.feature]:
        return split_leaf_id_sample(x, node.left_child, path + ("T",))
    return split_leaf_id_sample(x, node.right_child, path + ("F",))


def split_leaf_id_batch(X, tree):
    """Return path-based leaf ids for a batch of SPLIT samples.

    Args:
        X: 2-D numpy array of binary features.
        tree: Root of the SPLIT tree.

    Returns:
        List of tuple leaf ids.
    """
    return [split_leaf_id_sample(X[i], tree) for i in range(len(X))]


def cart_leaf_id_sample(row, node, path=()):
    """Return a stable path-based leaf id for one CART sample.

    Args:
        row: A pandas Series-like sample.
        node: Current CART node.
        path: Tuple of branch labels accumulated from the root.

    Returns:
        Tuple representing the path to the reached leaf.
    """
    if getattr(node, "left", None) is None:
        return path
    if node.is_numeric:
        go_left = row[node.feature] <= node.threshold
    else:
        go_left = row[node.feature] in node.left_categories
    if go_left:
        return cart_leaf_id_sample(row, node.left, path + ("L",))
    return cart_leaf_id_sample(row, node.right, path + ("R",))


def cart_leaf_id_batch(X, tree):
    """Return path-based leaf ids for a batch of CART samples.

    Args:
        X: DataFrame of CART input samples.
        tree: Root CART node.

    Returns:
        List of tuple leaf ids.
    """
    return [cart_leaf_id_sample(row, tree) for _, row in X.iterrows()]

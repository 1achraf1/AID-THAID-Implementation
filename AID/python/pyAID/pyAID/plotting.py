from __future__ import annotations

from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

from .tree import Node, assign_node_ids


def _layout_tree(node: Node, x=0, y=0, positions=None, leaf_positions=None):
    if positions is None:
        positions = {}
    if leaf_positions is None:
        leaf_positions = []

    if node.is_leaf:
        xpos = len(leaf_positions)
        positions[node._id] = (xpos, -y)
        leaf_positions.append(xpos)
        return positions, leaf_positions

    positions, leaf_positions = _layout_tree(node.left, x, y + 1, positions, leaf_positions)
    positions, leaf_positions = _layout_tree(node.right, x, y + 1, positions, leaf_positions)

    lx, ly = positions[node.left._id] if node.left else (x, -(y + 1))
    rx, ry = positions[node.right._id] if node.right else (x, -(y + 1))
    positions[node._id] = ((lx + rx) / 2, -y)
    return positions, leaf_positions


def plot_tree(root: Node, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Visualize the tree structure using a minimalist layout."""
    assign_node_ids(root)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    positions, _ = _layout_tree(root)

    def draw_edges(node: Node):
        if node.left:
            ax.plot(
                [positions[node._id][0], positions[node.left._id][0]],
                [positions[node._id][1], positions[node.left._id][1]],
                color="0.6",
            )
            draw_edges(node.left)
        if node.right:
            ax.plot(
                [positions[node._id][0], positions[node.right._id][0]],
                [positions[node._id][1], positions[node.right._id][1]],
                color="0.6",
            )
            draw_edges(node.right)

    draw_edges(root)

    def draw_nodes(node: Node):
        x, y = positions[node._id]
        label = (
            f"{node.feature_name or f'x{node.feature_index}'} ≤ {node.threshold:.3f}\n"
            f"n={node.n_samples}, mean={node.mean:.3f}"
            if not node.is_leaf
            else f"leaf\nn={node.n_samples}\nmean={node.mean:.3f}"
        )
        ax.scatter([x], [y], s=200, color="#2a9d8f" if node.is_leaf else "#264653")
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8)
        if node.left:
            draw_nodes(node.left)
        if node.right:
            draw_nodes(node.right)

    draw_nodes(root)
    ax.set_axis_off()
    ax.set_title("AID tree", fontsize=12)
    return ax


def plot_splits(
    X: np.ndarray,
    y: np.ndarray,
    node: Node,
    feature_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot scatter of the best split at a node."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    feat_name = feature_names[node.feature_index] if feature_names else f"x{node.feature_index}"
    ax.scatter(X[:, node.feature_index], y, alpha=0.6, s=20, color="#1d3557")
    ax.axvline(node.threshold, color="#e76f51", linestyle="--", label="threshold")
    ax.set_xlabel(feat_name)
    ax.set_ylabel("target")
    ax.legend()
    ax.set_title("Best split at node")
    return ax

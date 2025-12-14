from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict


@dataclass
class Node:
    """A simple binary tree node storing split meta-data and statistics."""

    depth: int
    n_samples: int
    sse: float
    mean: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    feature_name: Optional[str] = None
    gain: float = 0.0
    f_stat: Optional[float] = None
    parent_value: Optional[float] = None
    _id: int = field(default=0, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


    def predict_row(self, row) -> float:
        if self.is_leaf:
            return self.mean
        value = row[self.feature_index]
        branch = self.left if value <= self.threshold else self.right
        # Fall back to current mean if something is wrong.
        return branch.predict_row(row) if branch is not None else self.mean

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Remove non-serializable recursion markers.
        data["left"] = self.left.to_dict() if self.left else None
        data["right"] = self.right.to_dict() if self.right else None
        return data


def assign_node_ids(node: Node, start: int = 0) -> int:
    """Assign compact integer identifiers to nodes for plotting/export."""
    node._id = start
    next_id = start + 1
    if node.left:
        next_id = assign_node_ids(node.left, next_id)
    if node.right:
        next_id = assign_node_ids(node.right, next_id)
    return next_id

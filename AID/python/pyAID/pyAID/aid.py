from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .tree import Node
from .utils import SplitCandidate, ensure_numpy, find_best_split, sse_from_stats


class AIDRegressor:
    """Implementation of the historical AID regression algorithm.

    Parameters
    ----------
    R : int
        Minimum number of observations allowed in each child node.
    M : int
        Minimum number of observations required at a node to attempt a split.
    Q : int
        Maximum depth of the tree (root depth = 0).
    min_gain : float
        Minimum variance reduction required to accept a split.
    max_leaves : Optional[int]
        Optional cap on the total number of leaves.
    store_history : bool
        If True, keeps a log of evaluated splits for diagnostics.
    """

    def __init__(
        self,
        R: int = 5,
        M: int = 10,
        Q: int = 5,
        min_gain: float = 0.0,
        max_leaves: Optional[int] = None,
        store_history: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.R = R
        self.M = M
        self.Q = Q
        self.min_gain = min_gain
        self.max_leaves = max_leaves
        self.store_history = store_history
        self.random_state = random_state

        self.feature_names_: List[str] = []
        self.root_: Optional[Node] = None
        self.n_leaves_: int = 0
        self.history_: List[Dict[str, Any]] = []

    def fit(self, X, y) -> "AIDRegressor":
        X_arr, y_arr, feature_names = ensure_numpy(X, y)
        self.feature_names_ = feature_names

        self.n_leaves_ = 0
        self.history_ = []
        if self.random_state is not None:
            np.random.seed(self.random_state)

        total_sum = float(np.sum(y_arr))
        total_sum2 = float(np.sum(y_arr * y_arr))
        total_sse = sse_from_stats(total_sum, total_sum2, len(y_arr))
        mean_val = float(total_sum / len(y_arr))

        self.root_ = Node(
            depth=0,
            n_samples=len(y_arr),
            sse=total_sse,
            mean=mean_val,
            parent_value=None,
        )
        self._grow(self.root_, X_arr, y_arr, depth=0)
        return self

    def _can_split(self, node: Node, depth: int) -> bool:
        if depth >= self.Q:
            return False
        if node.n_samples < self.M:
            return False
        if self.max_leaves is not None and self.n_leaves_ >= self.max_leaves:
            return False
        return True

    def _grow(self, node: Node, X: np.ndarray, y: np.ndarray, depth: int) -> None:
        if not self._can_split(node, depth):
            self.n_leaves_ += 1
            return

        candidate = find_best_split(X, y, node.sse, self.R)
        if candidate is None or candidate.gain <= self.min_gain:
            self.n_leaves_ += 1
            return

        if len(candidate.left_indices) < self.R or len(candidate.right_indices) < self.R:
            self.n_leaves_ += 1
            return

        left_y = y[candidate.left_indices]
        right_y = y[candidate.right_indices]
        left_sse = candidate.left_sse
        right_sse = candidate.right_sse


        # Update node with split information.
        node.feature_index = candidate.feature_index
        node.feature_name = (
            self.feature_names_[candidate.feature_index]
            if self.feature_names_
            and candidate.feature_index < len(self.feature_names_)
            else None
        )
        node.threshold = candidate.threshold
        node.gain = candidate.gain
        node.f_stat = candidate.f_stat

        if self.store_history:
            self.history_.append(
                {
                    "depth": depth,
                    "feature": node.feature_name or f"x{node.feature_index}",
                    "threshold": node.threshold,
                    "gain": node.gain,
                    "f_stat": node.f_stat,
                    "n_left": len(candidate.left_indices),
                    "n_right": len(candidate.right_indices),
                }
            )

        node.left = Node(
            depth=depth + 1,
            n_samples=len(left_y),
            sse=left_sse,
            mean=float(np.mean(left_y)),
            parent_value=node.mean,
        )
        node.right = Node(
            depth=depth + 1,
            n_samples=len(right_y),
            sse=right_sse,
            mean=float(np.mean(right_y)),
            parent_value=node.mean,
        )

        self._grow(node.left, X[candidate.left_indices], left_y, depth + 1)
        self._grow(node.right, X[candidate.right_indices], right_y, depth + 1)

    def predict(self, X) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        preds = np.empty(X_arr.shape[0], dtype=float)
        for i, row in enumerate(X_arr):
            preds[i] = self.root_.predict_row(row)
        return preds

    def to_dict(self) -> Dict[str, Any]:
        if self.root_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.root_.to_dict()

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def summary(self) -> List[Dict[str, Any]]:
        """Return split history for diagnostics."""
        return list(self.history_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "R": self.R,
            "M": self.M,
            "Q": self.Q,
            "min_gain": self.min_gain,
            "max_leaves": self.max_leaves,
            "store_history": self.store_history,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "AIDRegressor":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key}")
            setattr(self, key, value)
        return self

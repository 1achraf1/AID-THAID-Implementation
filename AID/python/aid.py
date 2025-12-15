# =========================
# pyAID (optimized structure)
# Automatic Interaction Detection (AID) for regression
# Morgan & Sonquist (1963) - educational, variance/SSE reduction
# =========================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import json
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]


def _ensure_2d_float(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    return X


def _ensure_1d_float(y: ArrayLike) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y


def _sse_from_stats(sum_y: float, sum_y2: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return float(sum_y2 - (sum_y * sum_y) / n)


@dataclass
class _SplitCandidate:
    feature_index: int
    threshold: float
    gain: float
    f_stat: float
    left_idx: np.ndarray   # sample indices (global)
    right_idx: np.ndarray  # sample indices (global)


@dataclass
class _Node:
    depth: int
    n: int
    sse: float
    mean: float
    sum_y: float
    sum_y2: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    gain: float = 0.0
    f_stat: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class AIDRegressor:
    """
    AIDRegressor (Morgan & Sonquist, 1963) — regression tree by SSE reduction.

    Parameters
    ----------
    R : int
        Minimum size for each child node (min_child_size).
    M : int
        Minimum size of a node to attempt a split.
    Q : int
        Maximum depth (root depth = 0).
    min_gain : float
        Minimum SSE reduction required to accept a split.
    store_history : bool
        Store split history (for analysis/teaching).
    max_leaves : int
        Optional cap on number of leaves.
    presort : bool
        If True, presort each feature once at fit (often faster on large n).
    """

    def __init__(
        self,
        R: int = 5,
        M: int = 10,
        Q: int = 5,
        min_gain: float = 0.0,
        store_history: bool = False,
        max_leaves: int = 10**9,
        presort: bool = True,
    ):
        self.R = int(R)
        self.M = int(M)
        self.Q = int(Q)
        self.min_gain = float(min_gain)
        self.store_history = bool(store_history)
        self.max_leaves = int(max_leaves)
        self.presort = bool(presort)

        self.root_: Optional[_Node] = None
        self.history_: List[Dict[str, Any]] = []
        self._leaves = 0

        # fitted state
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._sorted_idx: Optional[List[np.ndarray]] = None  # per feature sorted sample indices
        self.feature_names_: Optional[List[str]] = None

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, X: ArrayLike, y: ArrayLike, feature_names: Optional[List[str]] = None) -> "AIDRegressor":
        Xn = _ensure_2d_float(X)
        yn = _ensure_1d_float(y)
        if Xn.shape[0] != yn.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self._X = Xn
        self._y = yn
        self.feature_names_ = feature_names if feature_names is not None else [f"x{i}" for i in range(Xn.shape[1])]

        if self.presort:
            self._sorted_idx = [
                np.argsort(self._X[:, j], kind="mergesort") for j in range(self._X.shape[1])
            ]
        else:
            self._sorted_idx = None

        # root stats (no copies)
        idx = np.arange(yn.size, dtype=int)
        sum_y = float(np.sum(yn))
        sum_y2 = float(np.sum(yn * yn))
        sse = _sse_from_stats(sum_y, sum_y2, int(yn.size))
        mean = float(sum_y / max(int(yn.size), 1))

        self.root_ = _Node(depth=0, n=int(yn.size), sse=sse, mean=mean, sum_y=sum_y, sum_y2=sum_y2)

        self.history_.clear()
        self._leaves = 0

        self.root_ = self._grow(self.root_, idx, depth=0)
        return self

    def _can_split(self, node: _Node, depth: int) -> bool:
        if depth >= self.Q:
            return False
        if node.n < self.M:
            return False
        if self._leaves >= self.max_leaves:
            return False
        # must be able to make 2 children of size >= R
        if node.n < 2 * self.R:
            return False
        return True

    # -----------------------------
    # Split search (indices-based)
    # -----------------------------
    def _find_best_split(self, idx: np.ndarray, parent: _Node) -> Optional[_SplitCandidate]:
        """
        Find best split across all features using SSE reduction.

        Works on global X/y but restricted to sample indices idx.
        No X/y submatrix copies are created.
        """
        assert self._X is not None and self._y is not None
        X = self._X
        y = self._y

        n = idx.size
        p = X.shape[1]
        if n < 2 * self.R:
            return None

        # mask for fast filtering (only if presort enabled)
        if self._sorted_idx is not None:
            in_node = np.zeros(y.size, dtype=bool)
            in_node[idx] = True

        best: Optional[_SplitCandidate] = None

        for j in range(p):
            if self._sorted_idx is not None:
                # THAID-like: presorted globally, then filter with node mask
                order_full = self._sorted_idx[j]
                relevant = order_full[in_node[order_full]]
            else:
                # fallback: sort just the node indices
                relevant = idx[np.argsort(X[idx, j], kind="mergesort")]

            if relevant.size < 2 * self.R:
                continue

            x_sorted = X[relevant, j]
            y_sorted = y[relevant]

            # candidate split positions where x changes
            diff_mask = x_sorted[:-1] != x_sorted[1:]
            if not np.any(diff_mask):
                continue

            pos = np.where(diff_mask)[0]  # split between pos and pos+1

            # enforce min_child_size
            left_n = pos + 1
            right_n = relevant.size - left_n
            ok = (left_n >= self.R) & (right_n >= self.R)
            pos = pos[ok]
            if pos.size == 0:
                continue

            csum_y = np.cumsum(y_sorted)
            csum_y2 = np.cumsum(y_sorted * y_sorted)
            total_sum = csum_y[-1]
            total_sum2 = csum_y2[-1]

            left_sum = csum_y[pos]
            left_sum2 = csum_y2[pos]
            right_sum = total_sum - left_sum
            right_sum2 = total_sum2 - left_sum2

            left_n_f = (pos + 1).astype(float)
            right_n_f = (relevant.size - (pos + 1)).astype(float)

            left_sse = left_sum2 - (left_sum * left_sum) / left_n_f
            right_sse = right_sum2 - (right_sum * right_sum) / right_n_f

            within = left_sse + right_sse
            gains = parent.sse - within

            # F-stat (descriptive): gain / (within/(n-2))
            denom = within / max(relevant.size - 2, 1)
            f_stats = np.where(denom > 0, gains / denom, 0.0)

            k = int(np.argmax(gains))
            if gains[k] <= 0:
                continue

            cut = int(pos[k])
            thr = float((x_sorted[cut] + x_sorted[cut + 1]) / 2.0)

            left_idx = relevant[: cut + 1]
            right_idx = relevant[cut + 1 :]

            cand = _SplitCandidate(
                feature_index=j,
                threshold=thr,
                gain=float(gains[k]),
                f_stat=float(f_stats[k]),
                left_idx=left_idx,
                right_idx=right_idx,
            )

            if best is None or cand.gain > best.gain:
                best = cand

        return best

    # -----------------------------
    # Tree growth (indices-based)
    # -----------------------------
    def _grow(self, node: _Node, idx: np.ndarray, depth: int) -> _Node:
        if not self._can_split(node, depth):
            self._leaves += 1
            return node

        cand = self._find_best_split(idx, node)
        if cand is None or cand.gain <= self.min_gain:
            self._leaves += 1
            return node

        assert self._y is not None
        y = self._y

        left_idx = cand.left_idx
        right_idx = cand.right_idx

        if left_idx.size < self.R or right_idx.size < self.R:
            self._leaves += 1
            return node

        # compute child stats without copying X (only y indexing)
        left_y = y[left_idx]
        right_y = y[right_idx]

        left_sum = float(np.sum(left_y))
        left_sum2 = float(np.sum(left_y * left_y))
        right_sum = float(np.sum(right_y))
        right_sum2 = float(np.sum(right_y * right_y))

        left_sse = _sse_from_stats(left_sum, left_sum2, int(left_y.size))
        right_sse = _sse_from_stats(right_sum, right_sum2, int(right_y.size))

        node.feature_index = cand.feature_index
        node.threshold = cand.threshold
        node.gain = cand.gain
        node.f_stat = cand.f_stat

        if self.store_history:
            self.history_.append(
                dict(
                    depth=depth,
                    feature_index=cand.feature_index,
                    feature_name=self.feature_names_[cand.feature_index] if self.feature_names_ else f"x{cand.feature_index}",
                    threshold=cand.threshold,
                    gain=cand.gain,
                    f_stat=cand.f_stat,
                    parent_n=node.n,
                    parent_mean=node.mean,
                    parent_sse=node.sse,
                    n_left=int(left_y.size),
                    mean_left=float(left_sum / left_y.size),
                    sse_left=float(left_sse),
                    n_right=int(right_y.size),
                    mean_right=float(right_sum / right_y.size),
                    sse_right=float(right_sse),
                )
            )

        node.left = _Node(
            depth=depth + 1,
            n=int(left_y.size),
            sse=float(left_sse),
            mean=float(left_sum / left_y.size),
            sum_y=left_sum,
            sum_y2=left_sum2,
        )
        node.right = _Node(
            depth=depth + 1,
            n=int(right_y.size),
            sse=float(right_sse),
            mean=float(right_sum / right_y.size),
            sum_y=right_sum,
            sum_y2=right_sum2,
        )

        node.left = self._grow(node.left, left_idx, depth + 1)
        node.right = self._grow(node.right, right_idx, depth + 1)
        return node

    # -----------------------------
    # Predict (batch traversal)
    # -----------------------------
    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model is not fitted.")
        Xn = _ensure_2d_float(X)

        preds = np.empty(Xn.shape[0], dtype=float)

        # stack of (node, indices_of_rows_to_route)
        stack: List[tuple[_Node, np.ndarray]] = [(self.root_, np.arange(Xn.shape[0], dtype=int))]

        while stack:
            node, rows = stack.pop()
            if rows.size == 0:
                continue

            # leaf
            if node.feature_index is None or node.threshold is None or node.left is None or node.right is None:
                preds[rows] = node.mean
                continue

            j = node.feature_index
            thr = node.threshold
            xcol = Xn[rows, j]
            go_left = xcol <= thr

            left_rows = rows[go_left]
            right_rows = rows[~go_left]

            stack.append((node.left, left_rows))
            stack.append((node.right, right_rows))

        return preds

    # -----------------------------
    # Utilities
    # -----------------------------
    def summary(self) -> str:
        if self.root_ is None:
            return "AIDRegressor(not fitted)"
        n_splits = len(self.history_) if self.store_history else "N/A"
        return (
            f"AIDRegressor(R={self.R}, M={self.M}, Q={self.Q}, min_gain={self.min_gain}, presort={self.presort})\n"
            f"Root: n={self.root_.n}, mean={self.root_.mean:.6f}, sse={self.root_.sse:.6f}\n"
            f"Splits stored: {n_splits}"
        )

    def _node_to_dict(self, node: Optional[_Node]) -> Optional[Dict[str, Any]]:
        if node is None:
            return None
        fname = None
        if node.feature_index is not None and self.feature_names_ is not None:
            fname = self.feature_names_[node.feature_index]
        return {
            "depth": node.depth,
            "n": node.n,
            "sum_y": node.sum_y,
            "sum_y2": node.sum_y2,
            "sse": node.sse,
            "mean": node.mean,
            "feature_index": node.feature_index,
            "feature_name": fname,
            "threshold": node.threshold,
            "gain": node.gain,
            "f_stat": node.f_stat,
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right),
        }

    def to_json(self) -> str:
        if self.root_ is None:
            raise ValueError("Model is not fitted.")
        return json.dumps(self._node_to_dict(self.root_), ensure_ascii=False)

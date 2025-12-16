#AID
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Sequence, Tuple
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


def _compute_sse(sum_y: float, sum_y2: float, n: int) -> float:
    """SSE = sum(y^2) - (sum(y)^2)/n"""
    if n <= 0:
        return 0.0
    return float(sum_y2 - (sum_y * sum_y) / n)


def _resolve_max_features(max_features, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        key = max_features.lower()
        if key == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if key == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features string must be one of: None, 'sqrt', 'log2'")
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("max_features float must be in (0, 1].")
        return max(1, int(np.ceil(max_features * n_features)))
    # int-like
    k = int(max_features)
    if k <= 0:
        raise ValueError("max_features must be >= 1")
    return min(k, n_features)


@dataclass
class _SplitCandidate:
    feature_idx: int
    split_value: float
    gain: float
    f_stat: float
    left_indices: np.ndarray   # global indices
    right_indices: np.ndarray  # global indices


class AIDNode:
    __slots__ = (
        "depth", "n_samples", "sse", "prediction", "sum_y", "sum_y2",
        "split_feature_idx", "split_value", "gain", "f_stat",
        "left", "right"
    )

    def __init__(self, depth: int, n_samples: int, sse: float, prediction: float,
                 sum_y: float, sum_y2: float):
        self.depth = depth
        self.n_samples = n_samples
        self.sse = sse
        self.prediction = prediction
        self.sum_y = sum_y
        self.sum_y2 = sum_y2

        self.split_feature_idx: Optional[int] = None
        self.split_value: Optional[float] = None
        self.gain: float = 0.0
        self.f_stat: Optional[float] = None
        self.left: Optional["AIDNode"] = None
        self.right: Optional["AIDNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.split_feature_idx is None


class AID:
    """
    AID regressor using SSE reduction (least squares) splits.

    Parameters
    ----------
    min_samples_leaf : int, default=5
        Minimum samples per leaf (AID parameter R).
    min_samples_split : int, default=10
        Minimum samples in a node to attempt a split (AID parameter M).
    max_depth : int, default=5
        Maximum depth (root depth = 0) (AID parameter Q).
    min_gain : float, default=0.0
        Minimum SSE reduction required to accept a split.
    max_leaves : int, default=10**9
        Optional cap on number of leaves.
    presort : bool, default=True
        If True, presort each feature once at fit and filter via node mask.
    store_history : bool, default=False
        Store split history.
    max_features : None | int | float | {"sqrt","log2"}, default=None
        Feature subsampling per node (like typical trees). If None uses all features.
    random_state : int | None, default=0
        RNG seed for feature subsampling (only relevant if max_features < n_features).
    """

    def __init__(
        self,
        min_samples_leaf: int = 5,
        min_samples_split: int = 10,
        max_depth: int = 5,
        min_gain: float = 0.0,
        store_history: bool = False,
        max_leaves: int = 10**9,
        presort: bool = True,
        max_features=None,
        random_state: Optional[int] = 0,
    ):
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_depth = int(max_depth)
        self.min_gain = float(min_gain)
        self.store_history = bool(store_history)
        self.max_leaves = int(max_leaves)
        self.presort = bool(presort)
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[AIDNode] = None
        self.history_: List[Dict[str, Any]] = []
        self._n_leaves = 0

        # fitted state
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._sorted_indices: Optional[List[np.ndarray]] = None  # per feature sorted sample indices
        self.feature_names_: Optional[List[str]] = None
        self.n_features_: int = 0

        self._rng = np.random.default_rng(random_state)

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, X: ArrayLike, y: ArrayLike, feature_names: Optional[List[str]] = None) -> "AID":
        X_arr, y_arr = self._validate_input(X, y)

        self._X = X_arr
        self._y = y_arr
        self.n_features_ = X_arr.shape[1]
        self.feature_names_ = feature_names if feature_names is not None else [f"X{i}" for i in range(X_arr.shape[1])]

        if self.presort:
            self._presort_numeric_features()
        else:
            self._sorted_indices = None

        indices = np.arange(y_arr.size, dtype=int)
        sum_y = float(np.sum(y_arr))
        sum_y2 = float(np.sum(y_arr * y_arr))
        sse = _compute_sse(sum_y, sum_y2, int(y_arr.size))
        prediction = float(sum_y / max(int(y_arr.size), 1))

        self.root_ = AIDNode(depth=0, n_samples=int(y_arr.size), sse=sse,
                             prediction=prediction, sum_y=sum_y, sum_y2=sum_y2)

        self.history_.clear()
        self._n_leaves = 0

        self.root_ = self._build_tree(self.root_, indices, depth=0)
        return self

    def _validate_input(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = _ensure_2d_float(X)
        y_arr = _ensure_1d_float(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Shape mismatch: X has {X_arr.shape[0]} samples, y has {y_arr.shape[0]}")

        if np.any(np.isnan(X_arr)) or np.any(np.isinf(X_arr)):
            raise ValueError("X contains NaN or Inf")

        if np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
            raise ValueError("y contains NaN or Inf")

        return X_arr, y_arr

    def _presort_numeric_features(self):
        assert self._X is not None
        self._sorted_indices = [
            np.argsort(self._X[:, j], kind="mergesort") for j in range(self.n_features_)
        ]

    def _should_stop(self, node: AIDNode, depth: int) -> bool:
        if depth >= self.max_depth:
            return True
        if node.n_samples < self.min_samples_split:
            return True
        if self._n_leaves >= self.max_leaves:
            return True
        if node.n_samples < 2 * self.min_samples_leaf:
            return True
        return False

    # -----------------------------
    # Split search (indices-based)
    # -----------------------------
    def _feature_subset(self) -> np.ndarray:
        k = _resolve_max_features(self.max_features, self.n_features_)
        if k >= self.n_features_:
            return np.arange(self.n_features_, dtype=int)
        # deterministic given RNG state
        return self._rng.choice(self.n_features_, size=k, replace=False)

    def _find_best_split(self, indices: np.ndarray, parent_node: AIDNode) -> Optional[_SplitCandidate]:
        assert self._X is not None and self._y is not None
        X = self._X
        y = self._y

        n = indices.size
        if n < 2 * self.min_samples_leaf:
            return None

        # mask for fast filtering (only if presort enabled)
        if self._sorted_indices is not None:
            in_node = np.zeros(y.size, dtype=bool)
            in_node[indices] = True

        best_candidate: Optional[_SplitCandidate] = None
        features = self._feature_subset()

        for feature_idx in features:
            if self._sorted_indices is not None:
                sorted_full = self._sorted_indices[feature_idx]
                relevant = sorted_full[in_node[sorted_full]]
            else:
                relevant = indices[np.argsort(X[indices, feature_idx], kind="mergesort")]

            if relevant.size < 2 * self.min_samples_leaf:
                continue

            x_sorted = X[relevant, feature_idx]
            y_sorted = y[relevant]

            # candidate split positions where x changes
            diff_mask = x_sorted[:-1] != x_sorted[1:]
            if not np.any(diff_mask):
                continue

            split_positions = np.where(diff_mask)[0]  # split between pos and pos+1

            # enforce min_samples_leaf
            left_n = split_positions + 1
            right_n = relevant.size - left_n
            valid_mask = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
            split_positions = split_positions[valid_mask]
            if split_positions.size == 0:
                continue

            csum_y = np.cumsum(y_sorted)
            csum_y2 = np.cumsum(y_sorted * y_sorted)
            total_sum = csum_y[-1]
            total_sum2 = csum_y2[-1]

            left_sum = csum_y[split_positions]
            left_sum2 = csum_y2[split_positions]
            right_sum = total_sum - left_sum
            right_sum2 = total_sum2 - left_sum2

            left_n_float = (split_positions + 1).astype(float)
            right_n_float = (relevant.size - (split_positions + 1)).astype(float)

            left_sse = left_sum2 - (left_sum * left_sum) / left_n_float
            right_sse = right_sum2 - (right_sum * right_sum) / right_n_float

            within_sse = left_sse + right_sse
            gains = parent_node.sse - within_sse

            best_idx = int(np.argmax(gains))
            best_gain = float(gains[best_idx])
            if best_gain <= 0:
                continue

            # "F-stat" descriptive (gain divided by within/(n-2))
            denom = float(within_sse[best_idx] / max(relevant.size - 2, 1))
            f_stat = float(best_gain / denom) if denom > 0 else float("inf")

            split_pos = int(split_positions[best_idx])
            split_value = float((x_sorted[split_pos] + x_sorted[split_pos + 1]) / 2.0)

            left_indices = relevant[: split_pos + 1]
            right_indices = relevant[split_pos + 1 :]

            candidate = _SplitCandidate(
                feature_idx=int(feature_idx),
                split_value=split_value,
                gain=best_gain,
                f_stat=f_stat,
                left_indices=left_indices,
                right_indices=right_indices,
            )

            if best_candidate is None or candidate.gain > best_candidate.gain:
                best_candidate = candidate

        return best_candidate

    # -----------------------------
    # Tree growth (indices-based)
    # -----------------------------
    def _build_tree(self, node: AIDNode, indices: np.ndarray, depth: int) -> AIDNode:
        if self._should_stop(node, depth):
            self._n_leaves += 1
            return node

        candidate = self._find_best_split(indices, node)
        if candidate is None or candidate.gain <= self.min_gain:
            self._n_leaves += 1
            return node

        assert self._y is not None
        y = self._y

        left_indices = candidate.left_indices
        right_indices = candidate.right_indices

        if left_indices.size < self.min_samples_leaf or right_indices.size < self.min_samples_leaf:
            self._n_leaves += 1
            return node

        left_y = y[left_indices]
        right_y = y[right_indices]

        left_sum = float(np.sum(left_y))
        left_sum2 = float(np.sum(left_y * left_y))
        right_sum = float(np.sum(right_y))
        right_sum2 = float(np.sum(right_y * right_y))

        left_sse = _compute_sse(left_sum, left_sum2, int(left_y.size))
        right_sse = _compute_sse(right_sum, right_sum2, int(right_y.size))

        node.split_feature_idx = candidate.feature_idx
        node.split_value = candidate.split_value
        node.gain = candidate.gain
        node.f_stat = candidate.f_stat

        if self.store_history:
            self.history_.append(
                dict(
                    depth=depth,
                    feature_idx=candidate.feature_idx,
                    feature_name=self.feature_names_[candidate.feature_idx] if self.feature_names_ else f"X{candidate.feature_idx}",
                    split_value=candidate.split_value,
                    gain=candidate.gain,
                    f_stat=candidate.f_stat,
                    parent_n=node.n_samples,
                    parent_mean=node.prediction,
                    parent_sse=node.sse,
                    n_left=int(left_y.size),
                    mean_left=float(left_sum / left_y.size),
                    sse_left=float(left_sse),
                    n_right=int(right_y.size),
                    mean_right=float(right_sum / right_y.size),
                    sse_right=float(right_sse),
                )
            )

        node.left = AIDNode(
            depth=depth + 1,
            n_samples=int(left_y.size),
            sse=float(left_sse),
            prediction=float(left_sum / left_y.size),
            sum_y=left_sum,
            sum_y2=left_sum2,
        )
        node.right = AIDNode(
            depth=depth + 1,
            n_samples=int(right_y.size),
            sse=float(right_sse),
            prediction=float(right_sum / right_y.size),
            sum_y=right_sum,
            sum_y2=right_sum2,
        )

        node.left = self._build_tree(node.left, left_indices, depth + 1)
        node.right = self._build_tree(node.right, right_indices, depth + 1)
        return node

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model not fitted")
        X_arr = _ensure_2d_float(X)

        predictions = np.empty(X_arr.shape[0], dtype=float)
        stack: List[Tuple[AIDNode, np.ndarray]] = [(self.root_, np.arange(X_arr.shape[0], dtype=int))]

        while stack:
            node, rows = stack.pop()
            if rows.size == 0:
                continue

            if node.is_leaf:
                predictions[rows] = node.prediction
                continue

            feature_idx = int(node.split_feature_idx)  # type: ignore[arg-type]
            split_value = float(node.split_value)      # type: ignore[arg-type]
            x_col = X_arr[rows, feature_idx]
            goes_left = x_col <= split_value

            left_rows = rows[goes_left]
            right_rows = rows[~goes_left]

            stack.append((node.left, left_rows))   # type: ignore[arg-type]
            stack.append((node.right, right_rows)) # type: ignore[arg-type]

        return predictions

    # -----------------------------
    # Score / utilities
    # -----------------------------
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_arr = _ensure_1d_float(y)
        y_pred = self.predict(X)

        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    def mse(self, X: ArrayLike, y: ArrayLike) -> float:
        y_arr = _ensure_1d_float(y)
        y_pred = self.predict(X)
        return float(np.mean((y_arr - y_pred) ** 2))

    def summary(self) -> str:
        if self.root_ is None:
            return "AID(not fitted)"
        n_splits = len(self.history_) if self.store_history else "N/A"
        return (
            f"AID(min_samples_leaf={self.min_samples_leaf}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_depth={self.max_depth}, min_gain={self.min_gain}, presort={self.presort}, "
            f"max_features={self.max_features}, random_state={self.random_state})\n"
            f"Root: n={self.root_.n_samples}, mean={self.root_.prediction:.6f}, sse={self.root_.sse:.6f}\n"
            f"Splits stored: {n_splits}"
        )

    def print_tree(self, max_depth: Optional[int] = None):
        if self.root_ is None:
            print("Model not fitted")
            return

        def print_node(node: AIDNode, depth: int = 0):
            if max_depth is not None and depth > max_depth:
                return

            indent = "  " * depth

            if node.is_leaf:
                print(f"{indent}Leaf: mean={node.prediction:.3f}, sse={node.sse:.3f}, n={node.n_samples}")
            else:
                feature_name = self.feature_names_[node.split_feature_idx] if self.feature_names_ else f"X{node.split_feature_idx}"
                print(f"{indent}{feature_name} <= {node.split_value:.6f} (gain={node.gain:.6f}, F={node.f_stat:.2f}, n={node.n_samples})")
                print_node(node.left, depth + 1)   # type: ignore[arg-type]
                print_node(node.right, depth + 1)  # type: ignore[arg-type]

        print_node(self.root_, 0)

    def to_dict(self) -> Dict[str, Any]:
        if self.root_ is None:
            raise ValueError("Model not fitted")

        def node_to_dict(node: AIDNode) -> Dict[str, Any]:
            d = {
                "depth": node.depth,
                "n_samples": node.n_samples,
                "prediction": node.prediction,
                "sse": node.sse,
            }
            if node.is_leaf:
                d["is_leaf"] = True
            else:
                d["is_leaf"] = False
                d["split_feature_idx"] = node.split_feature_idx
                d["split_feature_name"] = self.feature_names_[node.split_feature_idx] if self.feature_names_ else f"X{node.split_feature_idx}"
                d["split_value"] = node.split_value
                d["gain"] = node.gain
                d["f_stat"] = node.f_stat
                d["left"] = node_to_dict(node.left)   # type: ignore[arg-type]
                d["right"] = node_to_dict(node.right) # type: ignore[arg-type]
            return d

        return {
            "params": {
                "min_samples_leaf": self.min_samples_leaf,
                "min_samples_split": self.min_samples_split,
                "max_depth": self.max_depth,
                "min_gain": self.min_gain,
                "presort": self.presort,
                "max_features": self.max_features,
                "random_state": self.random_state,
            },
            "tree": node_to_dict(self.root_),
            "history": self.history_ if self.store_history else None,
        }

    def to_json(self, path: str):
        obj = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

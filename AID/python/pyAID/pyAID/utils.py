from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def ensure_numpy(X, y) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert inputs to contiguous numpy arrays and extract feature names."""
    feature_names: List[str]
    if hasattr(X, "to_numpy"):
        feature_names = list(getattr(X, "columns", []))
        X_arr = X.to_numpy(dtype=float, copy=False)
    else:
        feature_names = [f"x{i}" for i in range(np.asarray(X).shape[1])]
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    return np.ascontiguousarray(X_arr, dtype=float), y_arr, feature_names


def sse_from_stats(sum_y: float, sum_y2: float, n: int) -> float:
    """Compute sum of squared errors using aggregated statistics."""
    if n <= 0:
        return 0.0
    return float(sum_y2 - (sum_y * sum_y) / n)


@dataclass
class SplitCandidate:
    feature_index: int
    threshold: float
    gain: float
    f_stat: float
    left_indices: np.ndarray
    right_indices: np.ndarray
    left_sse: float
    right_sse: float



def find_best_split(
    X: np.ndarray, y: np.ndarray, parent_sse: float, min_child_size: int
) -> Optional[SplitCandidate]:
    """Find the best binary split across all features using variance reduction."""
    n_samples, n_features = X.shape
    best: Optional[SplitCandidate] = None

    for j in range(n_features):
        x = X[:, j]
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        y_sorted = y[order]

        diff_mask = np.diff(x_sorted) != 0
        if not np.any(diff_mask):
            continue

        split_positions = np.nonzero(diff_mask)[0]
        split_positions = split_positions[
            (split_positions + 1 >= min_child_size)
            & (n_samples - (split_positions + 1) >= min_child_size)
        ]
        if split_positions.size == 0:
            continue

        csum_y = np.cumsum(y_sorted)
        csum_y2 = np.cumsum(y_sorted * y_sorted)
        total_sum = csum_y[-1]
        total_sum2 = csum_y2[-1]

        left_n = split_positions + 1
        right_n = n_samples - left_n

        left_sum = csum_y[split_positions]
        left_sum2 = csum_y2[split_positions]
        right_sum = total_sum - left_sum
        right_sum2 = total_sum2 - left_sum2

        left_sse = left_sum2 - (left_sum * left_sum) / left_n
        right_sse = right_sum2 - (right_sum * right_sum) / right_n
        gains = parent_sse - (left_sse + right_sse)



        # F-statistic for a binary split (df1 = 1, df2 = n - 2):
        # F = (BSS/1) / (SSE/(n-2)) where BSS = parent_sse - (left_sse + right_sse)
        within = left_sse + right_sse  # SSE after split
        within_df = n_samples - 2
        mse = np.zeros_like(within, dtype=float)
        if within_df > 0:
            mse = within / within_df

        f_stats = np.zeros_like(gains, dtype=float)
        mask = mse > 0
        f_stats[mask] = gains[mask] / mse[mask]



        best_idx = int(np.argmax(gains))
        if gains[best_idx] <= 0:
            continue

        pos = split_positions[best_idx]
        threshold = (x_sorted[pos] + x_sorted[pos + 1]) * 0.5
        selected = SplitCandidate(
            feature_index=j,
            threshold=float(threshold),
            gain=float(gains[best_idx]),
            f_stat=float(f_stats[best_idx]),
            left_indices=order[: pos + 1],
            right_indices=order[pos + 1 :],
            left_sse=float(left_sse[best_idx]),
            right_sse=float(right_sse[best_idx]),
        )

        if best is None or selected.gain > best.gain:
            best = selected

    return best

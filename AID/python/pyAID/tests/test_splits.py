import numpy as np

from pyAID.utils import find_best_split, sse_from_stats


def test_best_split_on_simple_step():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    parent_sse = sse_from_stats(np.sum(y), np.sum(y * y), len(y))

    candidate = find_best_split(X, y, parent_sse=parent_sse, min_child_size=1)
    assert candidate is not None
    assert candidate.feature_index == 0
    assert 1.0 < candidate.threshold < 2.0
    assert candidate.gain > 0

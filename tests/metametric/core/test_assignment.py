from scipy.optimize import linear_sum_assignment
import numpy as np

from metametric.core._assignment import iterative_max_matching


def test_iterative_max_matching():
    cost = np.random.rand(5, 10)
    total, matches = list(iterative_max_matching(cost))[-1]
    matches.sort(key=lambda x: x[0])
    row_idx, col_idx = linear_sum_assignment(cost, maximize=True)
    assert np.isclose(total, cost[row_idx, col_idx].sum().item())
    matches_from_sp = [(i.item(), j.item(), cost[i, j].item()) for i, j in zip(row_idx, col_idx)]
    matches_from_sp.sort(key=lambda x: x[0])
    assert matches == matches_from_sp

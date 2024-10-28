from abc import abstractmethod
from typing import TypeVar, Generic
from collections.abc import Sequence, Collection

import numpy as np
import scipy.optimize as spo

from metametric.core.constraint import MatchingConstraint

T = TypeVar("T")


class MatchingProblem(Generic[T]):
    """A matching between two collections of objects."""

    def __init__(self, x: Sequence[T], y: Sequence[T], gram_matrix: np.ndarray):
        self.x = x
        self.y = y
        self.gram_matrix = gram_matrix

    @abstractmethod
    def solve(self) -> tuple[float, Collection[tuple[int, int, float]]]:
        """Solves the matching problem."""
        raise NotImplementedError


class AssignmentProblem(MatchingProblem[T]):
    def __init__(self, x: Sequence[T], y: Sequence[T], gram_matrix: np.ndarray, constraint: MatchingConstraint):
        super().__init__(x, y, gram_matrix)
        self.constraint = constraint

    def solve(self) -> tuple[float, Collection[tuple[int, int, float]]]:
        m = self.gram_matrix
        if self.constraint == MatchingConstraint.ONE_TO_ONE:
            row_idx, col_idx = spo.linear_sum_assignment(
                cost_matrix=m,
                maximize=True,
            )
            total = m[row_idx, col_idx].sum()
            matching = [(i.item(), j.item(), m[i, j].item()) for i, j in zip(row_idx, col_idx)]
            return total, matching
        if self.constraint == MatchingConstraint.ONE_TO_MANY:
            total = m.max(axis=0).sum().item()
            selected_x = m.argmax(axis=0)
            matching = [(selected_x[j].item(), j, m[selected_x[j], j].item()) for j in range(m.shape[1])]
            return total, matching
        if self.constraint == MatchingConstraint.MANY_TO_ONE:
            total = m.max(axis=1).sum().item()
            selected_y = m.argmax(axis=1)
            matching = [(i, selected_y[i].item(), m[i, selected_y[i]].item()) for i in range(m.shape[0])]
            return total, matching
        if self.constraint == MatchingConstraint.MANY_TO_MANY:
            total = m.sum().item()
            matching = [(i, j, m[i, j].item()) for i in range(m.shape[0]) for j in range(m.shape[1])]
            return total, matching
        raise ValueError(f"Invalid constraint: {self.constraint}")

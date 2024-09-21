from abc import abstractmethod
from typing import Sequence, Type, TypeVar, Union, Generic, Tuple

import numpy as np
import scipy.optimize as spo

from metametric.core.constraint import MatchingConstraint

T = TypeVar("T")


class Matching(Generic[T]):
    def __init__(self, x: Sequence[T], y: Sequence[T], matches: Sequence[Tuple[int, int, float]]):
        self.x = x
        self.y = y
        self._matches = matches

    def matches(self) -> Sequence[Tuple[T, T, float]]:
        return [(self.x[i], self.y[j], score) for i, j, score in self._matches]


class MatchingProblem(Generic[T]):
    """A matching between two collections of objects."""

    def __init__(self, x: Sequence[T], y: Sequence[T], gram_matrix: np.ndarray):
        self.x = x
        self.y = y
        self.gram_matrix = gram_matrix

    @abstractmethod
    def solve(self) -> Tuple[float, Matching[T]]:
        """Solves the matching problem."""
        raise NotImplementedError


class AssignmentProblem(MatchingProblem[T]):
    def __init__(self, x: Sequence[T], y: Sequence[T], gram_matrix: np.ndarray, constraint: MatchingConstraint):
        super().__init__(x, y, gram_matrix)
        self.constraint = constraint

    def solve(self) -> Tuple[float, Matching[T]]:
        m = self.gram_matrix
        if self.constraint == MatchingConstraint.ONE_TO_ONE:
            row_idx, col_idx = spo.linear_sum_assignment(
                cost_matrix=m,
                maximize=True,
            )
            total = m[row_idx, col_idx].sum()
            matches = [(i, j, m[i, j].item()) for i, j in zip(row_idx, col_idx)]
            return total, Matching(self.x, self.y, matches)
        if self.constraint == MatchingConstraint.ONE_TO_MANY:
            total = m.max(axis=0).sum().item()
            selected_x = m.argmax(axis=0)
            matches = [(selected_x[j].item(), j, m[selected_x[j], j].item()) for j in range(m.shape[1])]
            return total, Matching(self.x, self.y, matches)
        if self.constraint == MatchingConstraint.MANY_TO_ONE:
            total = m.max(axis=1).sum().item()
            selected_y = m.argmax(axis=1)
            matches = [(i, selected_y[i].item(), m[i, selected_y[i]].item()) for i in range(m.shape[0])]
            return total, Matching(self.x, self.y, matches)
        if self.constraint == MatchingConstraint.MANY_TO_MANY:
            total = m.sum().item()
            matches = [(i, j, m[i, j].item()) for i in range(m.shape[0]) for j in range(m.shape[1])]
            return total, Matching(self.x, self.y, matches)
        raise ValueError(f"Invalid constraint: {self.constraint}")

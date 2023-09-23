"""Metric derivation with alignment constraints."""
import enum
from enum import Enum
from typing import Collection, TypeVar, Union

import numpy as np
import scipy.optimize as spo

from autometric.core.metric import Metric

T = TypeVar("T")


class AlignmentConstraint(Enum):
    """Alignment constraints for the alignment metric."""

    ONE_TO_ONE = enum.auto()
    ONE_TO_MANY = enum.auto()
    MANY_TO_ONE = enum.auto()
    MANY_TO_MANY = enum.auto()

    @staticmethod
    def from_str(s: str) -> "AlignmentConstraint":
        return {
            "<->": AlignmentConstraint.ONE_TO_ONE,
            "<-": AlignmentConstraint.ONE_TO_MANY,
            "->": AlignmentConstraint.MANY_TO_ONE,
            "~": AlignmentConstraint.MANY_TO_MANY,
            "1:1": AlignmentConstraint.ONE_TO_ONE,
            "1:*": AlignmentConstraint.ONE_TO_MANY,
            "*:1": AlignmentConstraint.MANY_TO_ONE,
            "*:*": AlignmentConstraint.MANY_TO_MANY,
        }[s]


class AlignmentMetric(Metric[Collection[T]]):
    """A metric derived using some alignment constraints."""

    def __init__(self, inner: Metric[T], constraint: Union[str, AlignmentConstraint] = AlignmentConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = AlignmentConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        """Score two collections of objects.

        Args:
            x (`Collection[T]`): The first collection.
            y (`Collection[T]`): The second collection.

        Returns:
            `float`: The score of the alignment.
        """
        # TODO: alternative implementation when the inner metric is discrete
        return solve_alignment(
            self.inner.gram_matrix(x, y),
            self.constraint,
        )

    def score_self(self, x: Collection[T]) -> float:
        """Score a collection of objects with itself."""
        if self.constraint == AlignmentConstraint.MANY_TO_MANY:
            return self.inner.gram_matrix(x, x).sum()
        else:
            return sum(self.inner.score_self(u) for u in x)


def solve_alignment(gram_matrix: np.ndarray, constraint: AlignmentConstraint) -> float:
    """Solve the alignment problem.

    Args:
        gram_matrix (`np.ndarray`): The gram matrix of the inner metric.
        constraint (`AlignmentConstraint`): The alignment constraint.

    Returns:
        `float`: The score of the alignment.
    """
    if constraint == AlignmentConstraint.ONE_TO_ONE:
        row_idx, col_idx = spo.linear_sum_assignment(
            cost_matrix=gram_matrix,
            maximize=True,
        )
        return gram_matrix[row_idx, col_idx].sum()
    if constraint == AlignmentConstraint.ONE_TO_MANY:
        return gram_matrix.max(axis=0).sum()
    if constraint == AlignmentConstraint.MANY_TO_ONE:
        return gram_matrix.max(axis=1).sum()
    if constraint == AlignmentConstraint.MANY_TO_MANY:
        return gram_matrix.sum()

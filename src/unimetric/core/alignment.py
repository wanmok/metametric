"""Metric derivation with alignment constraints."""
import enum
from enum import Enum
from typing import Collection, TypeVar

import numpy as np
import scipy.optimize as spo

from unimetric.core.metric import Metric

T = TypeVar("T")


class AlignmentConstraint(Enum):
    """Alignment constraints for the alignment metric."""

    ONE_TO_ONE = enum.auto()
    ONE_TO_MANY = enum.auto()
    MANY_TO_ONE = enum.auto()
    MANY_TO_MANY = enum.auto()


class AlignmentMetric(Metric[Collection[T]]):
    """A metric derived using some alignment constraints."""

    def __init__(self, inner: Metric[T], constraint: AlignmentConstraint = AlignmentConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        """Score two collections of objects.

        Parameters
        ----------
        x : Collection[T]
        y : Collection[T]

        Returns
        -------
        float
            The score of the two collections.
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

    Parameters
    ----------
    gram_matrix : np.ndarray
        The gram matrix of the inner metric.
    constraint : AlignmentConstraint
        The alignment constraint.

    Returns
    -------
    float
        The score of the alignment.
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

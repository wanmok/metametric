import enum
from enum import Enum
from typing import Collection, TypeVar

import numpy as np
import scipy.optimize as spo

from unimetric.metric import Metric

T = TypeVar("T")


class AlignmentConstraint(Enum):
    """Alignment constraints for the alignment metric."""

    OneToOne = enum.auto()
    OneToMany = enum.auto()
    ManyToOne = enum.auto()
    ManyToMany = enum.auto()


class AlignmentMetric(Metric[Collection[T]]):
    def __init__(self, inner: Metric[T], constraint: AlignmentConstraint = AlignmentConstraint.OneToOne):
        self.inner = inner
        self.constraint = constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        # TODO: alternative implementation when the inner metric is discrete
        return solve_alignment(
            self.inner.gram_matrix(x, y),
            self.constraint,
        )

    def score_self(self, x: Collection[T]) -> float:
        if self.constraint == AlignmentConstraint.ManyToMany:
            return self.inner.gram_matrix(x, x).sum()
        else:
            return sum(self.inner.score_self(u) for u in x)


def solve_alignment(gram_matrix: np.ndarray, constraint: AlignmentConstraint) -> float:
    if constraint == AlignmentConstraint.OneToOne:
        row_idx, col_idx = spo.linear_sum_assignment(
            cost_matrix=gram_matrix,
            maximize=True,
        )
        return gram_matrix[row_idx, col_idx].sum()
    if constraint == AlignmentConstraint.OneToMany:
        return gram_matrix.max(axis=0).sum()
    if constraint == AlignmentConstraint.ManyToOne:
        return gram_matrix.max(axis=1).sum()
    if constraint == AlignmentConstraint.ManyToMany:
        return gram_matrix.sum()

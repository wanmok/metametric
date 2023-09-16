from typing import Collection, Generic, Mapping, TypeVar, Iterator, Tuple
from enum import Enum
import numpy as np
import scipy.optimize as spo

from unimetric.metric import Metric, DiscreteMetric


T = TypeVar('T')


class AlignmentConstraint(Enum):
    """
    Alignment constraints for the alignment metric.
    """
    ONE_TO_ONE = 0
    ONE_TO_MANY = 1
    MANY_TO_ONE = 2
    MANY_TO_MANY = 3


class AlignmentMetric(Metric[Collection[T]]):
    def __init__(self, inner: Metric[T], constraint: AlignmentConstraint = AlignmentConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        # TODO: alternative implementation when the inner metric is discrete
        return solve_alignment(
            self.inner.gram_matrix(x, y),
            self.constraint,
        )

    def score_self(self, x: Collection[T]) -> float:
        if self.constraint == AlignmentConstraint.MANY_TO_MANY:
            return self.inner.gram_matrix(x, x).sum()
        else:
            return sum(self.inner.score_self(u) for u in x)


def solve_alignment(gram_matrix: np.ndarray, constraint: AlignmentConstraint) -> float:
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

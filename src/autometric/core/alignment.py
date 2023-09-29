"""Metric derivation with alignment constraints."""
from dataclasses import is_dataclass
from typing import Collection, TypeVar, Union, Sequence, Type

import numpy as np
import scipy.optimize as spo

from autometric.core.constraint import AlignmentConstraint
from autometric.core.graph import Graph, _reachability_matrix
from autometric.core._ilp import MatchingProblem
from autometric.core.metric import Metric, DiscreteMetric

T = TypeVar("T")


class SetAlignmentMetric(Metric[Collection[T]]):
    """A metric derived from the alignment of two sets."""

    def __init__(self, inner: Metric[T], constraint: Union[str, AlignmentConstraint] = AlignmentConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = AlignmentConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        """Score two sets of objects."""
        if isinstance(self.inner, DiscreteMetric) and self.constraint == AlignmentConstraint.ONE_TO_ONE:
            return len(set(x) & set(y))
        else:
            m = self.inner.gram_matrix(x, y)
            if self.constraint == AlignmentConstraint.ONE_TO_ONE:
                row_idx, col_idx = spo.linear_sum_assignment(
                    cost_matrix=m,
                    maximize=True,
                )
                return m[row_idx, col_idx].sum()
            if self.constraint == AlignmentConstraint.ONE_TO_MANY:
                return m.max(axis=0).sum()
            if self.constraint == AlignmentConstraint.MANY_TO_ONE:
                return m.max(axis=1).sum()
            if self.constraint == AlignmentConstraint.MANY_TO_MANY:
                return m.sum()

    def score_self(self, x: Collection[T]) -> float:
        """Score a set of objects with itself."""
        if self.constraint == AlignmentConstraint.MANY_TO_MANY:
            return self.inner.gram_matrix(x, x).sum()
        elif self.constraint == AlignmentConstraint.ONE_TO_ONE:
            return sum(self.inner.score_self(u) for u in x)
        else:
            return self.score(x, x)


class SequenceAlignmentMetric(Metric[Sequence[T]]):
    """A metric derived from the alignment of two sequences."""

    def __init__(self, inner: Metric[T], constraint: Union[str, AlignmentConstraint] = AlignmentConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = AlignmentConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def score(self, x: Sequence[T], y: Sequence[T]) -> float:
        m = self.inner.gram_matrix(x, y)
        f = np.zeros([m.shape[0] + 1, m.shape[1] + 1])
        for i in range(m.shape[0] + 1):
            for j in range(m.shape[1] + 1):
                if i == 0 or j == 0:
                    f[i, j] = 0
                else:
                    f[i, j] = max(f[i - 1, j - 1] + m[i - 1, j - 1], f[i - 1, j], f[i, j - 1])
                    if self.constraint == AlignmentConstraint.ONE_TO_MANY:
                        f[i, j] = max(f[i, j], f[i, j - 1] + m[i - 1, j - 1])
                    elif self.constraint == AlignmentConstraint.MANY_TO_ONE:
                        f[i, j] = max(f[i, j], f[i - 1, j] + m[i - 1, j - 1])
                    elif self.constraint == AlignmentConstraint.MANY_TO_MANY:
                        f[i, j] = max(f[i, j], f[i, j - 1] + m[i - 1, j - 1],
                                      f[i - 1, j] + m[i - 1, j - 1])
        return f[-1, -1].item()

    def score_self(self, x: Sequence[T]) -> float:
        if self.constraint == AlignmentConstraint.ONE_TO_ONE:
            return sum(self.inner.score_self(u) for u in x)
        else:
            return self.score(x, x)


class GraphAlignmentMetric(Metric[Graph[T]]):
    """A metric derived from the alignment of two graphs (including trees, DAGs, and general graphs)."""
    def __init__(self, inner: Metric[T], constraint: Union[str, AlignmentConstraint] = AlignmentConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = AlignmentConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def score(self, x: Graph[T], y: Graph[T]) -> float:
        x_nodes = list(x.nodes())
        y_nodes = list(y.nodes())
        gram_matrix = self.inner.gram_matrix(x_nodes, y_nodes)
        x_reach = _reachability_matrix(x)
        y_reach = _reachability_matrix(y)

        problem = MatchingProblem(x_nodes, y_nodes, gram_matrix, has_vars=False)
        problem.add_matching_constraint(self.constraint)
        problem.add_monotonicity_constraint(x_reach, y_reach)
        return problem.solve()


class LatentSetAlignmentMetric(Metric[Collection[T]]):
    """A metric derived to support aligning latent variables defined in structures."""

    def __init__(
            self,
            cls: Type[T],
            inner: Metric[T],
            constraint: Union[str, AlignmentConstraint] = AlignmentConstraint.ONE_TO_ONE
    ):
        if is_dataclass(cls):
            self.cls = cls
        else:
            raise ValueError(f"{cls} has to be a dataclass.")
        self.inner = inner
        self.constraint = AlignmentConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        """Score two collections of objects."""
        x = list(x)
        y = list(y)
        gram_matrix = self.inner.gram_matrix(x, y)
        problem = MatchingProblem(x, y, gram_matrix, has_vars=True)
        problem.add_matching_constraint(self.constraint)
        problem.add_variable_matching_constraint()
        problem.add_latent_variable_constraint(self.cls)
        return problem.solve()

    def score_self(self, x: Collection[T]) -> float:
        """Score a collection of objects with itself."""
        return SetAlignmentMetric(self.inner, self.constraint).score_self(x)


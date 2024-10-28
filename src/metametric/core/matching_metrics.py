"""Metric derivation with matching constraints."""

from dataclasses import is_dataclass
from typing import TypeVar, Union
from collections.abc import Collection, Sequence
from collections import defaultdict

import numpy as np

from metametric.core._ilp import ILPMatchingProblem
from metametric.core.constraint import MatchingConstraint
from metametric.core.graph import Graph, _reachability_matrix
from metametric.core.matching import Match, Matching, Path
from metametric.core.metric import DiscreteMetric, Metric
from metametric.core._problem import AssignmentProblem


C = TypeVar("C")
T = TypeVar("T")


def _matching_from_triples(
    original_x: C,
    original_y: C,
    score: float,
    x: Sequence[T],
    y: Sequence[T],
    matches: Collection[tuple[int, int, float]],
) -> Matching:
    def _matching():
        yield Match(Path(), original_x, Path(), original_y, score)
        for i, j, s in matches:
            yield Match(Path().prepend(i), x[i], Path().prepend(j), y[j], s)

    return Matching(_matching())


class SetMatchingMetric(Metric[Collection[T]]):
    """A metric derived from the matching of two sets."""

    def __init__(self, inner: Metric[T], constraint: Union[str, MatchingConstraint] = MatchingConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = MatchingConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def compute(self, x: Collection[T], y: Collection[T]) -> tuple[float, Matching]:
        """Score two sets of objects."""
        original_x, original_y = x, y
        x, y = list(x), list(y)
        x_is_empty = len(x) == 0
        y_is_empty = len(y) == 0
        if x_is_empty and y_is_empty:
            return 1.0, Matching([Match(Path(), x, Path(), y, 1.0)])
        elif x_is_empty or y_is_empty:
            return 0.0, Matching([])
        elif isinstance(self.inner, DiscreteMetric) and self.constraint == MatchingConstraint.ONE_TO_ONE:
            intersection = set(x) & set(y)
            score = len(intersection)

            def _matching():
                yield Match(Path(), x, Path(), y, score)
                x_indices, y_indices = defaultdict(list), defaultdict(list)
                for i, u in enumerate(x):
                    x_indices[u].append(i)
                for j, v in enumerate(y):
                    y_indices[v].append(j)
                for k in intersection:
                    for i, j in zip(x_indices[k], y_indices[k]):
                        yield Match(Path().prepend(i), x[i], Path().prepend(j), y[j], 1.0)

            return score, Matching(_matching())
        else:
            m = self.inner.gram_matrix(x, y)
            score, triples = AssignmentProblem(x, y, m, self.constraint).solve()
            return score, _matching_from_triples(original_x, original_y, score, x, y, triples)

    def score_self(self, x: Collection[T]) -> float:
        """Score a set of objects with itself."""
        x = list(x)
        if len(x) == 0:
            return 1.0
        elif self.constraint == MatchingConstraint.MANY_TO_MANY:
            return self.inner.gram_matrix(x, x).sum()
        elif self.constraint == MatchingConstraint.ONE_TO_ONE:
            return sum(self.inner.score_self(u) for u in x)
        else:
            return self.score(x, x)


class SequenceMatchingMetric(Metric[Sequence[T]]):
    """A metric derived from the matching of two sequences."""

    def __init__(self, inner: Metric[T], constraint: Union[str, MatchingConstraint] = MatchingConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = MatchingConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def compute(self, x: Sequence[T], y: Sequence[T]) -> tuple[float, Matching]:
        m = self.inner.gram_matrix(x, y)
        f = np.zeros([m.shape[0] + 1, m.shape[1] + 1])
        for i in range(m.shape[0] + 1):
            for j in range(m.shape[1] + 1):
                if i == 0 or j == 0:
                    f[i, j] = 0
                else:
                    f[i, j] = max(f[i - 1, j - 1] + m[i - 1, j - 1], f[i - 1, j], f[i, j - 1])
                    if self.constraint == MatchingConstraint.ONE_TO_MANY:
                        f[i, j] = max(f[i, j], f[i, j - 1] + m[i - 1, j - 1])
                    elif self.constraint == MatchingConstraint.MANY_TO_ONE:
                        f[i, j] = max(f[i, j], f[i - 1, j] + m[i - 1, j - 1])
                    elif self.constraint == MatchingConstraint.MANY_TO_MANY:
                        f[i, j] = max(f[i, j], f[i, j - 1] + m[i - 1, j - 1], f[i - 1, j] + m[i - 1, j - 1])
        return f[-1, -1].item(), Matching([])  # TODO: implement matching

    def score_self(self, x: Sequence[T]) -> float:
        if self.constraint == MatchingConstraint.ONE_TO_ONE:
            return sum(self.inner.score_self(u) for u in x)
        else:
            return self.score(x, x)


class GraphMatchingMetric(Metric[Graph[T]]):
    """A metric derived from the matching of two graphs (including trees, DAGs, and general graphs)."""

    def __init__(self, inner: Metric[T], constraint: Union[str, MatchingConstraint] = MatchingConstraint.ONE_TO_ONE):
        self.inner = inner
        self.constraint = MatchingConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def compute(self, x: Graph[T], y: Graph[T]) -> tuple[float, Matching]:
        x_nodes = list(x.nodes())
        y_nodes = list(y.nodes())
        gram_matrix = self.inner.gram_matrix(x_nodes, y_nodes)
        x_reach = _reachability_matrix(x)
        y_reach = _reachability_matrix(y)

        problem = ILPMatchingProblem(x_nodes, y_nodes, gram_matrix, has_vars=False)
        problem.add_matching_constraint(self.constraint)
        problem.add_monotonicity_constraint(x_reach, y_reach)
        score, matches = problem.solve()
        return score, _matching_from_triples(x, y, score, x_nodes, y_nodes, matches)


class LatentSetMatchingMetric(Metric[Collection[T]]):
    """A metric derived to support matching latent variables defined in structures."""

    def __init__(
        self,
        cls: type[T],
        inner: Metric[T],
        constraint: Union[str, MatchingConstraint] = MatchingConstraint.ONE_TO_ONE,
    ):
        if is_dataclass(cls):
            self.cls = cls
        else:
            raise ValueError(f"{cls} has to be a dataclass.")
        self.inner = inner
        self.constraint = MatchingConstraint.from_str(constraint) if isinstance(constraint, str) else constraint

    def compute(self, x: Collection[T], y: Collection[T]) -> tuple[float, Matching]:
        """Score two collections of objects."""
        original_x, original_y = x, y
        x = list(x)
        y = list(y)

        x_is_empty = len(x) == 0
        y_is_empty = len(y) == 0

        if x_is_empty and y_is_empty:
            return 1.0, Matching([Match(Path(), x, Path(), y, 1.0)])
        elif x_is_empty or y_is_empty:
            return 0.0, Matching([])

        gram_matrix = self.inner.gram_matrix(x, y)
        problem = ILPMatchingProblem(x, y, gram_matrix, has_vars=True)
        problem.add_matching_constraint(self.constraint)
        problem.add_variable_matching_constraint()
        problem.add_latent_variable_constraint(self.cls)
        score, matches = problem.solve()
        return score, _matching_from_triples(original_x, original_y, score, x, y, matches)

    def score_self(self, x: Collection[T]) -> float:
        """Score a collection of objects with itself."""
        return SetMatchingMetric(self.inner, self.constraint).score_self(x)

"""Metrics for ranking problems."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from jaxtyping import Float

from metametric.core.matching import Matching
from metametric.core.metric import Metric, DiscreteMetric, ParameterizedMetric
from metametric.core._assignment import iterative_max_matching


T = TypeVar("T")


class WeightedRankingMetric(ParameterizedMetric[Sequence[tuple[T, float]], Float[np.ndarray, "k"]]):
    """A metric derived from the ranking of a set of objects.

    Note that the ranking is assumed to be in descending order, and the weight attached
    to each element is NOT the score from the ranking function:
    Instead, it is a weight to that element when computing the metric.
    """

    def __init__(self, inner: Metric[T], max_k: int = 100):
        self.inner = inner
        self.max_k = max_k

    def compute(
        self, x: Sequence[tuple[T, float]], y: Sequence[tuple[T, float]]
    ) -> tuple[Float[np.ndarray, "k"], Matching]:
        x_trunc = x[: self.max_k]
        y_dict = {v: v_score for v, v_score in y}

        if isinstance(self.inner, DiscreteMetric):
            match = np.zeros(self.max_k)
            for k, (u, u_score) in enumerate(x_trunc):
                match[k] = y_dict.get(u, 0.0) * u_score
            match_sum = match.cumsum()
            return match_sum, Matching([])  # TODO: implement matching

        else:  # Full iterative Hungarian matching
            gram_matrix = self.inner.gram_matrix([t[0] for t in x_trunc], [t[0] for t in y])
            x_weight = np.array(t[1] for t in x_trunc)
            y_weight = np.array(t[1] for t in y)
            gram_matrix *= x_weight[:, np.newaxis] * y_weight[np.newaxis, :]
            x_trunc_len = len(x_trunc)
            match_sum = np.zeros(self.max_k)
            for k, (total, matches) in enumerate(iterative_max_matching(gram_matrix)):
                match_sum[k] = total
            match_sum[x_trunc_len:] = match_sum[x_trunc_len - 1]
            return match_sum, Matching([])  # TODO: implement matching

    def score_self(self, x: Sequence[tuple[T, float]]) -> Float[np.ndarray, "k"]:
        x_trunc = x[: self.max_k]
        self_match = np.array([self.inner.score_self(u) * u_score * u_score for u, u_score in x_trunc]).cumsum()
        r = np.zeros(self.max_k)
        r[: len(self_match)] = self_match
        r[len(self_match) :] = self_match[-1]
        return r


class RankingMetric(ParameterizedMetric[Sequence[T], Float[np.ndarray, "k"]]):
    """A metric derived from the ranking of a set of objects.

    Note that the ranking is assumed to be in descending order.
    """

    def __init__(self, inner: Metric[T], max_k: int = 100):
        self.weighted = WeightedRankingMetric(inner, max_k)

    def compute(self, x: Sequence[T], y: Sequence[T]) -> tuple[Float[np.ndarray, "k"], Matching]:
        x_with_score = [(u, 1.0) for u in x]
        y_with_score = [(v, 1.0) for v in y]
        return self.weighted.compute(x_with_score, y_with_score)

    def score_self(self, x: Sequence[T]) -> Float[np.ndarray, "k"]:
        x_with_score = [(u, 1.0) for u in x]
        return self.weighted.score_self(x_with_score)

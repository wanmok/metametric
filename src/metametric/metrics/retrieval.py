"""Metric definitions for retrieval tasks."""

from typing import TypeVar

from metametric.core.normalizers import PrecisionAtK, NormalizedParametrizedMetric, RecallAtK
from metametric.core.ranking_metrics import RankingMetric
from metametric.core.metric import DiscreteMetric

T = TypeVar("T")

predicted = [
    ("a", 0.4),
    ("b", 0.3),
    ("c", 0.2),
    ("d", 0.1),
]

reference = [
    ("c", 1.0),
    ("d", 1.0),
    ("e", 1.0),
]


def sort_by_score(x: list[tuple[str, float]]) -> list[str]:
    """Sort a list of (str, float) tuples by the float value."""
    return [u for u, _ in sorted(x, key=lambda t: t[1], reverse=True)]


rm = RankingMetric(DiscreteMetric(str), max_k=10).contramap(sort_by_score)
rmp = NormalizedParametrizedMetric(rm, PrecisionAtK())
rmr = NormalizedParametrizedMetric(rm, RecallAtK())

z, _ = rm.compute(predicted, reference)
pk = rmp.compute(predicted, reference)
rk = rmr.compute(predicted, reference)

print(z)

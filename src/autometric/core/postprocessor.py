"""Metric aggregator for computing metrics on a batch of predictions and references."""
from typing import TypeVar, Protocol, Set, Optional, Dict, Sequence, List, Callable, Generic, Collection
from enum import Enum, auto
from dataclasses import dataclass

from autometric.core.aggregator import Aggregator
from autometric.core.metric import Metric
from autometric.core.normalizers import Normalizer


def _compute_normalized_metrics(
        normalizers: Collection[Normalizer],
        sxy: float,
        sxx: float,
        syy: float
) -> Dict[str, float]:
    return {
        normalizer.name: normalizer.normalize(sxy, sxx, syy)
        for normalizer in normalizers
    }


class Postprocessor(Protocol):

    def compute(self, agg: Aggregator) -> Dict[str, float]:
        """Compute the metrics from the aggregator."""
        raise NotImplementedError()


class MacroAverage(Postprocessor):
    def __init__(self, normalizers: Collection[Normalizer]):
        self.normalizers = normalizers

    def compute(self, agg: Aggregator) -> Dict[str, float]:
        n = len(agg)
        metrics_per_sample = [
            _compute_normalized_metrics(self.normalizers, sxy, sxx, syy)
            for sxy, sxx, syy in zip(agg.match, agg.pred, agg.ref)
        ]
        metrics = {
            normalizer.name: sum(metric[normalizer.name] for metric in metrics_per_sample) / n
            for normalizer in self.normalizers
        }
        return metrics


class MicroAverage(Postprocessor):
    def __init__(self, normalizers: Collection[Normalizer]):
        self.normalizers = normalizers

    def compute(self, agg: Aggregator) -> Dict[str, float]:
        sxy_total = sum(agg.match)
        sxx_total = sum(agg.pred)
        syy_total = sum(agg.ref)
        metrics = {
            name: value
            for name, value in _compute_normalized_metrics(self.normalizers, sxy_total, sxx_total, syy_total).items()
        }
        return metrics


class JoinedPostprocessor(Postprocessor):
    def __init__(self, metric_families: Dict[str, Postprocessor[T]]):
        self.metric_families = metric_families

    def compute(self, agg: Aggregator) -> Dict[str, float]:
        return {
            f"{prefix}-{name}": value
            for prefix, family in self.metric_families.items()
            for name, value in family.compute(agg).items()
        }


class PostprocessorWithExtra(Postprocessor):
    def __init__(self, extra: Callable[[Dict[str, float]], Dict[str, float]]):
        self.extra = extra

    def compute(self, agg: Aggregator) -> Dict[str, float]:
        metrics = self.compute(agg)
        metrics.update(self.extra(metrics))
        return metrics

"""Metric aggregator for computing metrics on a batch of predictions and references."""
from typing import Protocol, Dict, Callable, Collection

from autometric.core.state import MetricState
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


class Reduction(Protocol):

    def compute(self, agg: MetricState) -> Dict[str, float]:
        """Compute the metrics from the aggregator."""
        raise NotImplementedError()

    def with_extra(self, extra: Callable[[Dict[str, float]], Dict[str, float]]):
        return ReductionWithExtra(self, extra)


class MacroAverage(Reduction):
    def __init__(self, normalizers: Collection[Normalizer]):
        self.normalizers = normalizers

    def compute(self, agg: MetricState) -> Dict[str, float]:
        n = len(agg)
        metrics_per_sample = [
            _compute_normalized_metrics(self.normalizers, sxy, sxx, syy)
            for sxy, sxx, syy in zip(agg.matches, agg.preds, agg.refs)
        ]
        metrics = {
            normalizer.name: sum(metric[normalizer.name] for metric in metrics_per_sample) / n
            for normalizer in self.normalizers
        }
        return metrics


class MicroAverage(Reduction):
    def __init__(self, normalizers: Collection[Normalizer]):
        self.normalizers = normalizers

    def compute(self, agg: MetricState) -> Dict[str, float]:
        sxy_total = sum(agg.match)
        sxx_total = sum(agg.pred)
        syy_total = sum(agg.ref)
        metrics = {
            name: value
            for name, value in _compute_normalized_metrics(self.normalizers, sxy_total, sxx_total, syy_total).items()
        }
        return metrics


class MultipleReductions(Reduction):
    def __init__(self, reductions: Dict[str, Reduction]):
        self.reductions = reductions

    def compute(self, agg: MetricState) -> Dict[str, float]:
        return {
            (f"{prefix}-{name}" if name != "" else prefix): value
            for prefix, family in self.reductions.items()
            for name, value in family.compute(agg).items()
        }


class ReductionWithExtra(Reduction):
    def __init__(self, original: Reduction, extra: Callable[[Dict[str, float]], Dict[str, float]]):
        self.original = original
        self.extra = extra

    def compute(self, agg: MetricState) -> Dict[str, float]:
        metrics = self.original.compute(agg)
        metrics.update(self.extra(metrics))
        return metrics

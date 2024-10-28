"""Metric aggregator for computing metrics on a batch of predictions and references."""

from typing import Callable, Optional, Protocol
from collections.abc import Collection

from metametric.core.normalizers import Normalizer
from metametric.core.state import SingleMetricState


def _compute_normalized_metrics(
    normalizers: Collection[Optional[Normalizer]], sxy: float, sxx: float, syy: float
) -> dict[str, float]:
    normalized_metrics = {
        normalizer.name: normalizer.normalize(sxy, sxx, syy) for normalizer in normalizers if normalizer is not None
    }
    if None in normalizers:
        normalized_metrics[""] = sxy
    return normalized_metrics


class Reduction(Protocol):
    """Describes how a collection of metric results are reduced.

    Examples include macro-averaging and micro-averaging.
    """

    def compute(self, state: SingleMetricState) -> dict[str, float]:
        """Compute the metrics from the aggregator."""
        raise NotImplementedError()

    def with_extra(self, extra: Callable[[dict[str, float]], dict[str, float]]):
        return ReductionWithExtra(self, extra)


class MacroAverage(Reduction):
    """Macro-average reduction."""

    def __init__(self, normalizers: Collection[Optional[Normalizer]]):
        self.normalizers = normalizers
        self.normalizer_names = [normalizer.name for normalizer in normalizers if normalizer is not None]
        if None in normalizers:
            self.normalizer_names.append("")

    def compute(self, state: SingleMetricState) -> dict[str, float]:
        n = len(state)
        metrics_per_sample = [
            _compute_normalized_metrics(self.normalizers, sxy, sxx, syy)
            for sxy, sxx, syy in zip(state.matches, state.preds, state.refs)
        ]
        metrics = {name: sum(metric[name] for metric in metrics_per_sample) / n for name in self.normalizer_names}
        return metrics


class MicroAverage(Reduction):
    """Micro-average reduction."""

    def __init__(self, normalizers: Collection[Optional[Normalizer]]):
        self.normalizers = normalizers

    def compute(self, state: SingleMetricState) -> dict[str, float]:
        sxy_total = sum(state.matches)
        sxx_total = sum(state.preds)
        syy_total = sum(state.refs)
        metrics = {
            name: value
            for name, value in _compute_normalized_metrics(self.normalizers, sxy_total, sxx_total, syy_total).items()
        }
        return metrics


class MultipleReductions(Reduction):
    """A collection of multiple reductions."""

    def __init__(self, reductions: dict[str, Reduction]):
        self.reductions = reductions

    def compute(self, state: SingleMetricState) -> dict[str, float]:
        return {
            (f"{prefix}-{name}" if name != "" else prefix): value
            for prefix, family in self.reductions.items()
            for name, value in family.compute(state).items()
        }


class ReductionWithExtra(Reduction):
    """Equip a downstream function after reduction is computed."""

    def __init__(self, original: Reduction, extra: Callable[[dict[str, float]], dict[str, float]]):
        self.original = original
        self.extra = extra

    def compute(self, state: SingleMetricState) -> dict[str, float]:
        metrics = self.original.compute(state)
        metrics.update(self.extra(metrics))
        return metrics

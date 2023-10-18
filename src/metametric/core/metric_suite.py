"""Metric suites are collections of metrics that are computed together."""
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Protocol, Sequence, TypeVar

from metametric.core.metric import Metric
from metametric.core.reduction import Reduction
from metametric.core.state import (MetricState, MultipleMetricStates,
                                   SingleMetricState)

T = TypeVar("T", contravariant=True)


class Aggregator(Protocol[T]):
    """Metric aggregator for computing metrics on a stream of predictions and references."""

    @property
    def state(self) -> MetricState[T]:
        """The internal state of the aggregator."""
        raise NotImplementedError()

    def update_single(self, pred: T, ref: T) -> None:
        """Update the aggregator with a single prediction and its reference."""
        self.state.update_single(pred, ref)

    def update_batch(self, preds: Sequence[T], refs: Sequence[T]):
        """Update the aggregator with a batch of predictions and their references."""
        self.state.update_batch(preds, refs)

    def reset(self) -> None:
        """Reset the aggregator to its initialization state."""
        self.state.reset()

    def compute(self) -> Dict[str, float]:
        """Compute the metrics from the aggregator."""
        raise NotImplementedError()


class MetricSuite(Protocol[T]):
    """Metric suites are collections of metrics that are computed together."""
    def new(self) -> Aggregator[T]:
        """Create a new aggregator for the metric suite."""
        raise NotImplementedError()

    def with_extra(self, extra: Callable[[Dict[str, float]], Dict[str, float]]) -> "MetricSuite[T]":
        """Add extra metrics to the metric suite."""
        return MetricSuiteWithExtra(self, extra)


@dataclass
class MetricFamily(MetricSuite[T]):
    """A collection of metrics that shared the same internal state.

    For example, the precision, recall, and F-1 of event detection should be computed together within one family.
    """
    def __init__(self, metric: Metric[T], reduction: Reduction):
        self.metric = metric
        self.reduction = reduction

    def new(self) -> Aggregator[T]:
        return MetricFamilyAggregator(self)


class MetricFamilyAggregator(Aggregator[T]):
    """Aggregator for a metric family."""
    def __init__(self, family: MetricFamily[T]):
        self.family = family
        self._state = SingleMetricState(family.metric)

    @property
    def state(self) -> MetricState[T]:
        return self._state

    def compute(self) -> Dict[str, float]:
        return self.family.reduction.compute(self._state)


class MultipleMetricFamilies(MetricSuite[T]):
    """A collection of metric families, whose internal states are separate."""
    def __init__(
            self,
            families: Dict[str, MetricSuite[T]]
    ):
        self.families = families

    def new(self) -> Aggregator[T]:
        return MultipleMetricFamiliesAggregator(self)


class MultipleMetricFamiliesAggregator(Aggregator[T]):
    """Aggregator for multiple metric families."""
    def __init__(self, coll: MultipleMetricFamilies[T]):
        self.aggs: Dict[str, Aggregator[T]] = {
            name: family.new()
            for name, family in coll.families.items()
        }

    @cached_property
    def state(self) -> MetricState[T]:
        return MultipleMetricStates({
            name: agg.state
            for name, agg in self.aggs.items()
        })

    def compute(self) -> Dict[str, float]:
        metrics = {
            f"{name}-{key}" if key != "" else name: value
            for name, agg in self.aggs.items()
            for key, value in agg.compute().items()
        }
        return metrics


class MetricSuiteWithExtra(MetricSuite[T]):
    """A metric suite with extra metrics."""
    def __init__(self, original: MetricSuite[T], extra: Callable[[Dict[str, float]], Dict[str, float]]):
        self.original = original
        self.extra = extra

    def new(self) -> Aggregator[T]:
        return WithExtraAggregator(self)


class WithExtraAggregator(Aggregator[T]):
    """Aggregator for a metric suite with extra metrics."""
    def __init__(self, coll: MetricSuiteWithExtra[T]):
        self.agg = coll.original.new()
        self.extra = coll.extra

    @property
    def state(self) -> MetricState[T]:
        return self.agg.state

    def compute(self) -> Dict[str, float]:
        metrics = self.agg.compute()
        metrics.update(self.extra(metrics))
        return metrics

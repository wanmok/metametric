from dataclasses import dataclass
from typing import TypeVar, Dict, Sequence, Protocol, Callable
from functools import cached_property
from autometric.core.metric import Metric
from autometric.core.reduction import MetricState, Reduction
from autometric.core.state import StateFactory, MultipleMetricStates, DefaultStateFactory

T = TypeVar("T")


class Aggregator(Protocol[T]):

    @property
    def state(self) -> MetricState[T]:
        raise NotImplementedError()

    def update_single(self, pred: T, ref: T) -> None:
        """Update the aggregator with a single prediction and its reference."""
        self.state.update_single(pred, ref)

    def update_batch(self, preds: Sequence[T], refs: Sequence[T]):
        """Update the aggregator with a batch of predictions and their references."""
        self.state.update_batch(preds, refs)

    def reset(self) -> None:
        self.state.reset()

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError()


class MetricCollection(Protocol[T]):
    def new(self, factory: StateFactory) -> Aggregator[T]:
        raise NotImplementedError()

    def with_extra(self, extra: Callable[[Dict[str, float]], Dict[str, float]]) -> "MetricCollection[T]":
        return MetricCollectionWithExtra(self, extra)


@dataclass
class MetricFamily(MetricCollection[T]):
    def __init__(self, metric: Metric[T], reduction: Reduction):
        self.metric = metric
        self.reduction = reduction

    def new(self, factory: StateFactory) -> Aggregator[T]:
        return MetricFamilyAggregator(self, factory)


class MetricFamilyAggregator(Aggregator[T]):
    def __init__(self, family: MetricFamily[T], factory: StateFactory):
        self.family = family
        self._state = factory.new(family.metric)

    @property
    def state(self) -> MetricState[T]:
        return self._state

    def compute(self) -> Dict[str, float]:
        return self.family.reduction.compute(self.state)


class MultipleMetricFamilies(MetricCollection[T]):
    def __init__(
            self,
            families: Dict[str, MetricFamily[T]]
    ):
        self.families = families

    def new(self, factory: StateFactory) -> Aggregator[T]:
        return MultipleMetricFamiliesAggregator(self, factory)


class MultipleMetricFamiliesAggregator(Aggregator[T]):
    def __init__(self, coll: MultipleMetricFamilies[T], factory: StateFactory):
        self.aggs: Dict[str, Aggregator[T]] = {
            name: family.new(factory)
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


class MetricCollectionWithExtra(MetricCollection[T]):
    def __init__(self, original: MetricCollection[T], extra: Callable[[Dict[str, float]], Dict[str, float]]):
        self.original = original
        self.extra = extra

    def new(self, factory: StateFactory) -> Aggregator[T]:
        return WithExtraAggregator(self, factory)


class WithExtraAggregator(Aggregator[T]):
    def __init__(self, coll: MetricCollectionWithExtra[T], factory: StateFactory):
        self.agg = coll.original.new(factory)
        self.extra = coll.extra

    @property
    def state(self) -> MetricState[T]:
        return self.agg.state

    def compute(self) -> Dict[str, float]:
        metrics = self.agg.compute()
        metrics.update(self.extra(metrics))
        return metrics

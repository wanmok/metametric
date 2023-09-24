from dataclasses import dataclass
from typing import TypeVar, Dict, Sequence, Protocol, Callable
from autometric.core.metric import Metric
from autometric.core.reduction import Aggregator, Reduction

T = TypeVar("T")


class MetricCollectionAggregator(Protocol[T]):
    def update(self, pred: T, ref: T) -> None:
        """Update the aggregator with a single prediction and its reference."""
        raise NotImplementedError()

    def update_batch(self, pred: Sequence[T], ref: Sequence[T]):
        """Update the aggregator with a batch of predictions and their references."""
        for p, r in zip(pred, ref):
            self.update(p, r)

    def reset(self) -> None:
        raise NotImplementedError()

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError()


class MetricCollection(Protocol[T]):
    def new(self) -> MetricCollectionAggregator[T]:
        raise NotImplementedError()

    def with_extra(self, extra: Callable[[Dict[str, float]], Dict[str, float]]) -> "MetricCollection[T]":
        return MetricCollectionWithExtra(self, extra)



@dataclass
class MetricFamily(MetricCollection[T]):
    def __init__(self, metric: Metric[T], postprocessor: Reduction):
        self.metric = metric
        self.postprocessor = postprocessor

    def new(self) -> MetricCollectionAggregator[T]:
        return MetricFamilyAggregator(self)


class MetricFamilyAggregator(MetricCollectionAggregator[T]):
    def __init__(self, family: MetricFamily[T]):
        self.family = family
        self.agg = Aggregator(self.family.metric)

    def update(self, pred: T, ref: T) -> None:
        self.agg.update(pred, ref)

    def reset(self) -> None:
        self.agg.reset()

    def compute(self) -> Dict[str, float]:
        return self.family.postprocessor.compute(self.agg)


class MultipleMetricFamilies(MetricCollection[T]):
    def __init__(
            self,
            families: Dict[str, MetricFamily[T]]
    ):
        self.families = families

    def new(self) -> MetricCollectionAggregator[T]:
        return MultipleMetricFamiliesAggregator(self)


class MultipleMetricFamiliesAggregator(MetricCollectionAggregator[T]):
    def __init__(self, coll: MultipleMetricFamilies[T]):
        self.aggs = {
            name: family.new()
            for name, family in coll.families.items()
        }

    def update(self, pred: T, ref: T) -> None:
        for agg in self.aggs.values():
            agg.update(pred, ref)

    def reset(self) -> None:
        for agg in self.aggs.values():
            agg.reset()

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

    def new(self) -> MetricCollectionAggregator[T]:
        return MetricCollectionWithExtraAggregator(self)


class MetricCollectionWithExtraAggregator(MetricCollectionAggregator[T]):
    def __init__(self, coll: MetricCollectionWithExtra[T]):
        self.agg = coll.original.new()
        self.extra = coll.extra

    def update(self, pred: T, ref: T) -> None:
        self.agg.update(pred, ref)

    def reset(self) -> None:
        self.agg.reset()

    def compute(self) -> Dict[str, float]:
        metrics = self.agg.compute()
        metrics.update(self.extra(metrics))
        return metrics

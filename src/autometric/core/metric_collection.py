from dataclasses import dataclass
from typing import Generic, TypeVar, Dict, Sequence, Protocol, Callable, Optional
from autometric.core.metric import Metric
from autometric.core.postprocessor import Aggregator, Postprocessor

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


@dataclass
class MetricFamily(MetricCollection[T]):
    def __init__(self, metric: Metric[T], postprocessor: Postprocessor[T]):
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


class JoinedMetricCollection(MetricCollection[T]):
    def __init__(
            self,
            families: Dict[str, MetricFamily[T]],
            extra: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None):
        self.families = families
        self.extra = extra

    def new(self) -> MetricCollectionAggregator[T]:
        return JoinedMetricCollectionAggregator(self)


class JoinedMetricCollectionAggregator(MetricCollectionAggregator[T]):
    def __init__(self, coll: JoinedMetricCollection[T]):
        self.aggs = {
            name: family.new()
            for name, family in coll.families.items()
        }
        self.extra = coll.extra

    def update(self, pred: T, ref: T) -> None:
        for agg in self.aggs.values():
            agg.update(pred, ref)

    def reset(self) -> None:
        for agg in self.aggs.values():
            agg.reset()

    def compute(self) -> Dict[str, float]:
        metrics = {
            name: agg.compute()
            for name, agg in self.aggs.items()
        }
        if self.extra is not None:
            metrics = self.extra(metrics)
        return metrics
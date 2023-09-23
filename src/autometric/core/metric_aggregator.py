from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Set, Optional, Dict, Sequence, List, Callable
from enum import Enum, auto

from autometric.core.metric import Metric
from autometric.core.normalizers import Normalizer, Precision, Recall, FScore

T = TypeVar("T")


class MetricAggregator(Generic[T], ABC):
    @abstractmethod
    def update(self, pred: T, ref: T) -> None:
        raise NotImplementedError()

    def update_batch(self, pred: Sequence[T], ref: Sequence[T]):
        for p, r in zip(pred, ref):
            self.update(p, r)

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()


class Averaging(Enum):
    """Averaging methods."""

    MICRO = auto()
    MACRO = auto()

    @staticmethod
    def from_str(s: str) -> "Averaging":
        return {
            "micro": Averaging.MICRO,
            "macro": Averaging.MACRO,
        }[s]


class SingleMetricAggregator(MetricAggregator[T]):
    def __init__(
            self,
            metric: Metric[T],
            averaging: Optional[Set[Averaging]] = None,
            normalizers: Optional[List[Normalizer]] = None,
    ):
        self.metric = metric
        self.averaging = averaging or {Averaging.MACRO, Averaging.MICRO}
        self.normalizers = normalizers or [Precision(), Recall(), FScore()]
        self.pred = []
        self.ref = []
        self.match = []

    def update(self, pred: T, ref: T):
        sxx = self.metric.score_self(pred)
        syy = self.metric.score_self(ref)
        sxy = self.metric.score(pred, ref)
        self.pred.append(sxx)
        self.ref.append(syy)
        self.match.append(sxy)


    def _compute_normalized_metrics(self, sxy: float, sxx: float, syy: float) -> Dict[str, float]:
        return {
            normalizer.name(): normalizer.normalize(sxy, sxx, syy)
            for normalizer in self.normalizers
        }

    def compute(self) -> Dict[str, float]:
        n = len(self.match)
        if n == 0:
            return {}
        metrics = {}
        if Averaging.MICRO in self.averaging:
            sxy_total = sum(self.match)
            sxx_total = sum(self.pred)
            syy_total = sum(self.ref)
            metrics |= {
                f"micro-{name}": value
                for name, value in self._compute_normalized_metrics(sxy_total, sxx_total, syy_total).items()
            }
        if Averaging.MACRO in self.averaging:
            metrics_per_sample = [
                self._compute_normalized_metrics(sxy, sxx, syy)
                for sxy, sxx, syy in zip(self.match, self.pred, self.ref)
            ]
            metrics |= {
                f"macro-{normalizer.name()}": sum(metric[normalizer.name()] for metric in metrics_per_sample) / n
                for normalizer in self.normalizers
            }
        return metrics

    def reset(self):
        self.pred = []
        self.ref = []
        self.match = []


class MetricAggregatorCollection(MetricAggregator[T]):
    def __init__(
            self,
            aggregators: Dict[str, MetricAggregator[T]],
            extra: Optional[Callable[[Dict[str, float]], Dict[str, float]]],
    ):
        self.aggregators = aggregators
        self.extra = extra

    def update(self, pred: T, ref: T):
        for agg in self.aggregators.values():
            agg.update(pred, ref)

    def update_batch(self, pred: Sequence[T], ref: Sequence[T]):
        for agg in self.aggregators.values():
            agg.update_batch(pred, ref)

    def compute(self) -> Dict[str, float]:
        metrics = {
            f"{name}-{key}": value
            for name, agg in self.aggregators.items()
            for key, value in agg.compute().items()
        }
        if self.extra is None:
            return metrics
        else:
            extra_metrics = self.extra(metrics)
            return {**metrics, **extra_metrics}

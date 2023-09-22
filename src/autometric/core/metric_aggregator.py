"""Metric aggregator for computing metrics on a batch of predictions and references."""
from typing import TypeVar, Generic, Set, Optional, Dict, Sequence, List
from enum import Enum, auto

from autometric.core.metric import Metric
from autometric.core.normalizers import Normalizer, Precision, Recall, FScore

T = TypeVar("T")


class Averaging(Enum):
    """Averaging methods."""

    MICRO = auto()
    MACRO = auto()


class MetricAggregator(Generic[T]):
    """Metric aggregator for computing metrics on a batch of predictions and references.

    The aggregator can compute multiple metrics at once, and supports multiple averaging methods.
    """

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

    def _update(self, pred: T, ref: T):
        """Update the aggregator with a single prediction and its reference."""
        sxx = self.metric.score_self(pred)
        syy = self.metric.score_self(ref)
        sxy = self.metric.score(pred, ref)
        self.pred.append(sxx)
        self.ref.append(syy)
        self.match.append(sxy)

    def update(self, pred: T, ref: T):
        """Wrapper that allows for polymorphism."""
        self._update(pred, ref)

    def update_batch(self, preds: Sequence[T], refs: Sequence[T]):
        """Update the aggregator with a batch of predictions and their references."""
        for p, r in zip(preds, refs):
            self._update(p, r)

    def _compute_normalized_metrics(self, sxy: float, sxx: float, syy: float) -> Dict[str, float]:
        return {normalizer.name: normalizer.normalize(sxy, sxx, syy) for normalizer in self.normalizers}

    def compute(self) -> Dict[str, float]:
        """Compute the metrics."""
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
                f"macro-{normalizer.name}": sum(metric[normalizer.name] for metric in metrics_per_sample) / n
                for normalizer in self.normalizers
            }
        return metrics

    def reset(self):
        """Reset the aggregator."""
        self.pred = []
        self.ref = []
        self.match = []

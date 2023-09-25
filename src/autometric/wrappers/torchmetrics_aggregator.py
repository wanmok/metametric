"""TorchMetrics wrapper for autometric."""
from typing import TypeVar, Optional, Set, List, Any, Sequence

from torchmetrics import Metric as TorchMetric

from autometric.core.metric import Metric
from autometric.core.state import MetricState
from autometric.core.normalizers import Normalizer

T = TypeVar("T")


class TorchMetricsStateFactory(TorchMetric, MetricState[T]):
    """TorchMetrics wrapper for autometric."""

    is_differentiable = False

    def __init__(self, metric: Metric[T]):
        TorchMetric.__init__(self)
        self.metric = metric

        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("refs", [], dist_reduce_fx="cat")
        self.add_state("match", [], dist_reduce_fx="cat")

    def update_single(self, pred: T, ref: T) -> None:
        pass

    def update(self, preds: Sequence[T], refs: Sequence[T]) -> None:
        """Update the aggregator with a batch of predictions and their references."""
        MetricAggregator.update_batch(self, preds=preds, refs=refs)

    def compute(self) -> Any:
        """Compute the metrics."""
        return MetricAggregator.compute(self)

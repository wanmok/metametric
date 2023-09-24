"""TorchMetrics wrapper for autometric."""
from typing import TypeVar, Optional, Set, List, Any, Sequence

from torchmetrics import Metric as TorchMetric

from autometric.core.metric import Metric
from autometric.core.metric_aggregator import MetricAggregator, Averaging
from autometric.core.normalizers import Normalizer

T = TypeVar("T")


class TorchMetricsMetricAggregator(TorchMetric, MetricAggregator[T]):
    """TorchMetrics wrapper for autometric."""

    is_differentiable = False

    def __init__(
        self,
        metric: Metric[T],
        averaging: Optional[Set[Averaging]] = None,
        normalizers: Optional[List[Normalizer]] = None,
    ):
        TorchMetric.__init__(self)
        MetricAggregator.__init__(self, metric=metric, averaging=averaging, normalizers=normalizers)

        self.add_state("preds", [], dist_reduce_fx="cat")
        self.add_state("refs", [], dist_reduce_fx="cat")
        self.add_state("match", [], dist_reduce_fx="cat")

    def update(self, preds: Sequence[T], refs: Sequence[T]) -> None:
        """Update the aggregator with a batch of predictions and their references."""
        MetricAggregator.update_batch(self, preds=preds, refs=refs)

    def compute(self) -> Any:
        """Compute the metrics."""
        return MetricAggregator.compute(self)

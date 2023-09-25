"""Tests TorchMetrics wrapper for autometric."""
from pytest import approx

from autometric.core.alignment import AlignmentMetric, AlignmentConstraint
from autometric.core.metric import DiscreteMetric
from autometric.core.metric_aggregator import Averaging
from autometric.core.normalizers import Precision, Recall, FScore
from autometric.wrappers.torchmetrics_aggregator import TorchMetricsMetricAggregator


def test_torchmetrics_metric_aggregator():
    """Basic test for metric aggregator that computes precision, recall, and F-score."""
    a = [[0, 1], [2], [1, 2]]
    b = [[0, 1, 2, 3], [2, 3], [1, 2, 3]]
    metric_aggregator = TorchMetricsMetricAggregator(
        AlignmentMetric(DiscreteMetric(int), AlignmentConstraint.ONE_TO_ONE),
        averaging={Averaging.MACRO, Averaging.MICRO},
        normalizers=[Precision(), Recall(), FScore(), FScore(0.5), FScore(2)],
    )
    metric_aggregator.update_single(a, b)
    metrics = metric_aggregator.compute()

    assert metrics["micro-precision"] == approx(1.0, abs=0.01)
    assert metrics["micro-recall"] == approx(0.55, abs=0.01)
    assert metrics["micro-f1"] == approx(0.71, abs=0.01)
    assert metrics["micro-f0.5"] == approx(0.86, abs=0.01)
    assert metrics["micro-f2"] == approx(0.61, abs=0.01)
    assert metrics["macro-precision"] == approx(1.0, abs=0.01)
    assert metrics["macro-recall"] == approx(0.55, abs=0.01)
    assert metrics["macro-f1"] == approx(0.71, abs=0.01)
    assert metrics["macro-f0.5"] == approx(0.86, abs=0.01)
    assert metrics["macro-f2"] == approx(0.61, abs=0.01)

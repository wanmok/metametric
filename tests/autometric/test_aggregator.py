from autometric.core.alignment import AlignmentMetric, AlignmentConstraint
from autometric.core.metric import DiscreteMetric
from autometric.core.reduction import MetricAggregator, Averaging
from autometric.core.normalizers import Precision, FScore, Recall


def test_metric_aggregator():
    a = [[0, 1], [2], [1, 2]]
    b = [[0, 1, 2, 3], [2, 3], [1, 2, 3]]
    metric_aggregator = MetricAggregator(
        AlignmentMetric(DiscreteMetric(int), AlignmentConstraint.ONE_TO_ONE),
        averaging={Averaging.MACRO, Averaging.MICRO},
        normalizers=[Precision(), Recall(), FScore(), FScore(0.5), FScore(2)],
    )
    metric_aggregator.update_batch(a, b)
    metrics = metric_aggregator.compute()
    for k, v in metrics.items():
        print(f"{k}: {v}")
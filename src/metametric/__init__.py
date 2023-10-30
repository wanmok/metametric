"""The core functionality of metametric package."""

__version__ = "0.1.0"


from metametric.core.metric import Metric, Variable
from metametric.core.constraint import MatchingConstraint
from metametric.core.metric_suite import MetricFamily, MetricSuite, Aggregator

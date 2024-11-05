"""The core functionality of metametric package."""

import importlib.metadata


__version__ = importlib.metadata.version(__package__ or __name__)


from metametric.core.constraint import MatchingConstraint  # noqa: F401
from metametric.core.metric import Metric, Variable  # noqa: F401
from metametric.core.metric_suite import MetricFamily, MetricSuite  # noqa: F401
from metametric.core.path import Path  # noqa: F401
from metametric.core.reduction import Reduction  # noqa: F401

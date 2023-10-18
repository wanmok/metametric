"""Defines the states of metric aggregators."""
from typing import Dict, Protocol, Sequence, TypeVar

from metametric.core.metric import Metric

T = TypeVar("T", contravariant=True)


class MetricState(Protocol[T]):
    """Encapsulates the state of a metric aggregator."""

    def update_single(self, pred: T, ref: T) -> None:
        """Update the aggregator with a single prediction and its reference."""
        raise NotImplementedError()

    def update_batch(self, preds: Sequence[T], refs: Sequence[T]) -> None:
        """Update the aggregator with a batch of predictions and their references."""
        for p, r in zip(preds, refs):
            self.update_single(p, r)

    def reset(self) -> None:
        """Reset the aggregator."""
        raise NotImplementedError()

    def __len__(self):
        """Return the number of predictions."""
        raise NotImplementedError()


class SingleMetricState(MetricState[T]):
    """Encapsulates the state of a single metric aggregator."""
    def __init__(self, metric: Metric[T]):
        self.metric = metric
        self.preds = []
        self.refs = []
        self.matches = []

    def update_single(self, pred: T, ref: T) -> None:
        """Update the aggregator with a single prediction and its reference."""
        sxx = self.metric.score_self(pred)
        syy = self.metric.score_self(ref)
        sxy = self.metric.score(pred, ref)
        self.preds.append(sxx)
        self.refs.append(syy)
        self.matches.append(sxy)

    def reset(self) -> None:
        """Reset the aggregator to its initialization state."""
        self.preds = []
        self.refs = []
        self.matches = []

    def __len__(self):
        """Returns the number of prediction-reference pairs aggregated."""
        return len(self.matches)


class MultipleMetricStates(MetricState[T]):
    """Encapsulates the state of multiple metric aggregators."""
    def __init__(self, states: Dict[str, MetricState[T]]):
        self.states = states

    def update_single(self, pred: T, ref: T) -> None:
        for state in self.states.values():
            state.update_single(pred, ref)

    def update_batch(self, preds: Sequence[T], refs: Sequence[T]) -> None:
        for state in self.states.values():
            state.update_batch(preds, refs)

    def reset(self) -> None:
        for state in self.states.values():
            state.reset()

    def __len__(self):
        """Returns the number of prediction-reference pairs aggregated."""
        return len(list(self.states.values())[0])

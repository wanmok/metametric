from typing import Generic, TypeVar, Sequence, Protocol, Dict

from autometric.core.metric import Metric


T = TypeVar("T")


class MetricState(Protocol[T]):

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
        self.preds = []
        self.refs = []
        self.matches = []

    def __len__(self):
        return len(self.matches)


class MultipleMetricStates(MetricState[T]):
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
        return len(list(self.states.values())[0])


class StateFactory(Protocol):

    def new(self, metric: Metric[T]) -> MetricState[T]:
        """Create a new aggregator."""
        raise NotImplementedError()


class DefaultStateFactory(StateFactory):

    def new(self, metric: Metric[T]) -> MetricState[T]:
        return SingleMetricState(metric)

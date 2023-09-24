from typing import Generic, TypeVar, Sequence

from autometric.core.metric import Metric


T = TypeVar("T")


class Aggregator(Generic[T]):

    def __init__(self, metric: Metric[T]):
        self.metric = metric
        self.pred = []
        self.ref = []
        self.match = []

    def update(self, pred: T, ref: T) -> None:
        """Update the aggregator with a single prediction and its reference."""
        sxx = self.metric.score_self(pred)
        syy = self.metric.score_self(ref)
        sxy = self.metric.score(pred, ref)
        self.pred.append(sxx)
        self.ref.append(syy)
        self.match.append(sxy)

    def update_batch(self, pred: Sequence[T], ref: Sequence[T]):
        """Update the aggregator with a batch of predictions and their references."""
        for p, r in zip(pred, ref):
            self.update(p, r)

    def reset(self) -> None:
        self.pred = []
        self.ref = []
        self.match = []

    def __len__(self):
        return len(self.match)



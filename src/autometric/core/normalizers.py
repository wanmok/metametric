from abc import abstractmethod
from typing import TypeVar, Generic
from enum import Enum, auto
from autometric.core.metric import Metric


T = TypeVar("T")


class Normalizer:
    """A metric that normalizes another metric."""

    @abstractmethod
    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        raise NotImplementedError()

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()


class Jaccard(Normalizer):
    """Jaccard metric."""
    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        return score_xy / (score_xx + score_yy - score_xy)

    def name(self) -> str:
        return "jaccard"


class Precision(Normalizer):
    """Precision metric."""
    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        return score_xy / score_xx

    def name(self) -> str:
        return "precision"


class Recall(Normalizer):
    """Recall metric."""
    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        return score_xy / score_yy

    def name(self) -> str:
        return "recall"


class FScore(Normalizer):
    """F-score metric."""
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        return (1 + self.beta ** 2) * score_xy / ((self.beta ** 2) * score_yy + score_xx) if score_xy > 0.0 else 0.0

    def name(self) -> str:
        if self.beta == 1.0:
            return "f1"
        else:
            return f"f{self.beta}"


class NormalizingMetric(Metric[T]):
    def __init__(self, inner: Metric[T], normalizer: Normalizer):
        self.inner = inner
        self.normalizer = normalizer

    def score(self, x: T, y: T) -> float:
        """Score two objects."""
        sxy = self.inner.score(x, y)
        sxx = self.inner.score_self(x)
        syy = self.inner.score_self(y)
        return self.normalizer.normalize(sxy, sxx, syy)

    def score_self(self, x: T) -> float:
        return 1.0

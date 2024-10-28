"""Normalizers to normalize metrics as normalized metrics."""

from typing import Optional, Protocol, TypeVar, runtime_checkable

from metametric.core.matching import Matching, Match, Path
from metametric.core.metric import Metric

T = TypeVar("T")


@runtime_checkable
class Normalizer(Protocol):
    """A metric that normalizes another metric."""

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric.

        Args:
            score_xy (`float`): The score between two objects.
            score_xx (`float`): The score of the first object with itself, usually on the prediction side.
            score_yy (`float`): The score of the second object with itself, usually on the reference side.

        Returns:
            `float`: The normalized score.
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """Get the name of the normalizer."""
        raise NotImplementedError()

    @staticmethod
    def from_str(s: str) -> Optional["Normalizer"]:
        if s == "none":
            return None
        if s == "jaccard":
            return Jaccard()
        elif s == "precision":
            return Precision()
        elif s == "recall":
            return Recall()
        elif s == "dice":
            return FScore()
        elif s.startswith("f"):
            return FScore(beta=float(s[1:]))
        else:
            raise ValueError(f"Unknown normalizer {s}")


class Jaccard(Normalizer):
    """Jaccard metric."""

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using Jaccard metric."""
        return score_xy / (score_xx + score_yy - score_xy)

    @property
    def name(self) -> str:
        """Get the name of the normalizer."""
        return "jaccard"


class Precision(Normalizer):
    """Precision metric."""

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using precision metric."""
        return score_xy / score_xx

    @property
    def name(self) -> str:
        """Get the name of the normalizer."""
        return "precision"


class Recall(Normalizer):
    """Recall metric."""

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using recall metric."""
        return score_xy / score_yy

    @property
    def name(self) -> str:
        """Get the name of the normalizer."""
        return "recall"


class FScore(Normalizer):
    """F-score metric."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using F-score metric."""
        return (1 + self.beta**2) * score_xy / ((self.beta**2) * score_yy + score_xx) if score_xy > 0.0 else 0.0

    @property
    def name(self) -> str:
        """Get the name of the FScore based on `beta`."""
        if self.beta == 1.0:
            return "f1"
        else:
            b = int(self.beta) if self.beta.is_integer() else self.beta
            return f"f{b}"


class NormalizedMetric(Metric[T]):
    """A wrapper for the metric that normalizes another metric.

    This ensures that applying a [`Normalizer`] to a [`Metric`] is also a [`Metric`].
    """

    def __init__(self, inner: Metric[T], normalizer: Normalizer):
        self.inner = inner
        self.normalizer = normalizer

    def compute(self, x: T, y: T) -> tuple[float, Matching]:
        """Score two objects."""
        sxy, inner_matching = self.inner.compute(x, y)
        sxx = self.inner.score_self(x)
        syy = self.inner.score_self(y)
        normalized_score = self.normalizer.normalize(sxy, sxx, syy)

        def _matching():
            for match in inner_matching:
                if match.pred_path.is_root() and match.ref_path.is_root():
                    yield Match(Path(), x, Path(), y, normalized_score)
                else:
                    yield match

        return normalized_score, Matching(_matching())

    def score_self(self, x: T) -> float:
        """Score an object with itself."""
        return 1.0

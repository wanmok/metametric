"""Normalizers to normalize metrics as normalized metrics."""

from typing import Optional, Protocol, TypeVar, runtime_checkable

from jaxtyping import Float
import numpy as np

from metametric.core.matching import Matching, Match, Path
from metametric.core.metric import Metric, ParameterizedMetric

T = TypeVar("T")
RI = TypeVar("RI", contravariant=True)
RO = TypeVar("RO", covariant=True)


@runtime_checkable
class Normalizer(Protocol[RI, RO]):
    """A metric that normalizes another metric."""

    name: str

    def normalize(self, score_xy: RI, score_xx: RI, score_yy: RI) -> RO:
        """Normalize the metric.

        Args:
            score_xy (`float`): The score between two objects.
            score_xx (`float`): The score of the first object with itself, usually on the prediction side.
            score_yy (`float`): The score of the second object with itself, usually on the reference side.

        Returns:
            `float`: The normalized score.
        """
        raise NotImplementedError()

    @staticmethod
    def from_str(s: str) -> Optional["Normalizer"]:
        if s == "none":
            return None
        if s == "jaccard" or s == "j":
            return Jaccard()
        elif s == "precision" or s == "p":
            return Precision()
        elif s == "precision@k" or s == "p@k":
            return PrecisionAtK()
        elif s == "recall@k" or s == "r@k":
            return RecallAtK()
        elif s == "ranking_average_precision" or s == "ranking_ap":
            return RankingAveragePrecision()
        elif s == "recall" or s == "r":
            return Recall()
        elif s == "dice" or s == "f":
            return FScore()
        elif s.startswith("f"):
            return FScore(beta=float(s[1:]))
        else:
            raise ValueError(f"Unknown normalizer {s}")


class Jaccard(Normalizer[float, float]):
    """Jaccard metric."""

    name = "jaccard"

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using Jaccard metric."""
        return score_xy / (score_xx + score_yy - score_xy)


class Precision(Normalizer[float, float]):
    """Precision metric."""

    name = "precision"

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using precision metric."""
        return score_xy / score_xx


class Recall(Normalizer[float, float]):
    """Recall metric."""

    name = "recall"

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using recall metric."""
        return score_xy / score_yy


class FScore(Normalizer[float, float]):
    """F-score metric."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        if self.beta == 1.0:
            self.name = "f1"
        else:
            b = int(self.beta) if self.beta.is_integer() else self.beta
            self.name = f"f{b}"

    def normalize(self, score_xy: float, score_xx: float, score_yy: float) -> float:
        """Normalize the metric using F-score metric."""
        return (1 + self.beta**2) * score_xy / ((self.beta**2) * score_yy + score_xx) if score_xy > 0.0 else 0.0


class PrecisionAtK(Normalizer[Float[np.ndarray, "k"], Float[np.ndarray, "k"]]):
    """Precision@k metric."""

    name = "precision@k"

    def normalize(
        self, score_xy: Float[np.ndarray, "k"], score_xx: Float[np.ndarray, "k"], score_yy: Float[np.ndarray, "k"]
    ) -> Float[np.ndarray, "k"]:
        """Normalize the metric using precision@k metric."""
        return score_xy / score_xx


class RecallAtK(Normalizer[Float[np.ndarray, "k"], Float[np.ndarray, "k"]]):
    """Recall@k metric."""

    name = "recall@k"

    def normalize(
        self, score_xy: Float[np.ndarray, "k"], score_xx: Float[np.ndarray, "k"], score_yy: Float[np.ndarray, "k"]
    ) -> Float[np.ndarray, "k"]:
        """Normalize the metric using recall@k metric."""
        return score_xy / score_yy


class RankingAveragePrecision(Normalizer[Float[np.ndarray, "k"], float]):
    """Average precision metric."""

    name = "ranking_average_precision"

    def normalize(
        self, score_xy: Float[np.ndarray, "k"], score_xx: Float[np.ndarray, "k"], score_yy: Float[np.ndarray, "k"]
    ) -> float:
        """Normalize the metric using average precision metric."""
        p = score_xy / score_xx
        r = score_xy / score_yy
        dr = np.diff(r, prepend=0.0)
        return np.dot(p, dr).item()


class NormalizedParametrizedMetric(ParameterizedMetric[T, RO]):
    """A wrapper for the parameterized metric that normalizes another metric.

    This ensures that applying a [`Normalizer`] to a [`ParameterizedMetric`] is also a [`ParameterizedMetric`].
    """

    def __init__(self, inner: ParameterizedMetric[T, RI], normalizer: Normalizer[RI, RO]):
        self.inner = inner
        self.normalizer = normalizer

    def compute(self, x: T, y: T) -> tuple[RO, Matching]:
        """Score two objects."""
        sxy, inner_matching = self.inner.compute(x, y)
        sxx = self.inner.score_self(x)
        syy = self.inner.score_self(y)
        normalized_score = self.normalizer.normalize(sxy, sxx, syy)

        def _matching():
            for match in inner_matching:
                if match.pred_path.is_root() and match.ref_path.is_root() and isinstance(normalized_score, float):
                    yield Match(Path(), x, Path(), y, normalized_score)
                else:
                    yield match

        return normalized_score, Matching(_matching())


class NormalizedMetric(NormalizedParametrizedMetric[T, float], Metric[T]):
    """A wrapper for the metric that normalizes another metric.

    This ensures that applying a [`Normalizer`] to a [`Metric`] is also a [`Metric`].
    """

    def __init__(self, inner: Metric[T], normalizer: Normalizer[float, float]):
        super().__init__(inner, normalizer)

    def score_self(self, x: T) -> float:
        """Score an object with itself."""
        return 1.0

"""Metric interface and implementations for commonly used metrics."""

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from functools import reduce
from operator import mul
from typing import (
    Callable,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    Union,
    get_origin,
    runtime_checkable,
)
from collections.abc import Sequence

import numpy as np

from metametric.core.matching import Matching, Match, Path

S = TypeVar("S", contravariant=True)
T = TypeVar("T", contravariant=True)
U = TypeVar("U")


class Metric(Generic[T]):
    r"""The basic metric interface.

    Here a *metric* is defined as a function $\phi: T \times T \to \mathbb{R}_{\ge 0}$ that takes two objects and
    returns a non-negative number that quantifies their similarity.
    It follows the common usage in machine learning and NLP literature, as in the phrase "evaluation metrics".
    This is *not* the metric in the mathematical sense, where it is a generalization of *distances*.
    """

    @abstractmethod
    def compute(self, x: T, y: T) -> tuple[float, Matching]:
        r"""Scores two objects using this metric, and returns the score and a matching object."""
        raise NotImplementedError

    def score(self, x: T, y: T) -> float:
        r"""Scores two objects using this metric: $\phi(x, y)$."""
        return self.compute(x, y)[0]

    def score_self(self, x: T) -> float:
        r"""Scores an object against itself: $\phi(x, x)$.

        In many cases there is a faster way to compute this than the general pair case.
        In such cases, please override this function.
        """
        return self.score(x, x)

    def gram_matrix(self, xs: Sequence[T], ys: Sequence[T]) -> np.ndarray:
        r"""Computes the Gram matrix of the metric given two collections of objects.

        Args:
            xs: A collection of objects $\{x_1, \ldots, x_n\}$.
            ys: A collection of objects $\{y_1, \ldots, y_m\}$.

        Returns:
            A Gram matrix $G$ where $G = \begin{bmatrix} \phi(x_1, y_1) & \cdots & \phi(x_1, y_m) \\
                                \vdots & \ddots & \vdots \\
                                \phi(x_n, y_1) & \cdots & \phi(x_n, y_m) \end{bmatrix}$.
        """
        return np.array([[self.score(x, y) for y in ys] for x in xs])

    def contramap(self, f: Callable[[S], T]) -> "Metric[S]":
        r"""Returns a new metric $\phi^\prime$ by first preprocessing the objects by a given function $f: S \to T$.

        \[ \phi^\prime(x, y) = \phi(f(x), f(y)) \]

        Args:
            f: A preprocessing function.

        Returns:
            A new metric $\phi^\prime$.

        """
        return ContramappedMetric(self, f)

    @staticmethod
    def from_function(f: Callable[[T, T], float]) -> "Metric[T]":
        r"""Create a metric from a function $f: T \times T \to \mathbb{R}_{\ge 0}$.

        Args:
            f (`Callable[[T, T], float]`):
                A function that takes two objects and returns a float.
                This is the function that derives the metric.

        Returns:
            `Metric[T]`: A metric that uses the function to score two objects.
        """
        return MetricFromFunction(f)


class MetricFromFunction(Metric[T]):
    """A metric wrapped from a function."""

    def __init__(self, f: Callable[[T, T], float]):
        self.f = f

    def compute(self, x: T, y: T) -> tuple[float, Matching]:
        """Score two objects."""
        score = self.f(x, y)

        def _matching():
            yield Match(Path(), x, Path(), y, score)

        return self.f(x, y), Matching(_matching())


class ContramappedMetric(Metric[S]):
    """A metric contramapped by a function."""

    def __init__(self, inner: Metric[T], f: Callable[[S], T]):
        self.inner = inner
        self.f = f

    def compute(self, x: S, y: S) -> tuple[float, Matching]:
        """Score two objects."""
        return self.inner.compute(self.f(x), self.f(y))

    def score_self(self, x: S) -> float:
        """Scores an object against itself."""
        return self.inner.score_self(self.f(x))


class DiscreteMetric(Metric[T]):
    """A metric for discrete objects."""

    def __init__(self, cls: type[T]):
        if getattr(cls, "__eq__", None) is None:
            raise ValueError("Class must implement __eq__")

    def compute(self, x: T, y: T) -> tuple[float, Matching]:
        """Score two objects."""
        if x == y:
            return 1.0, Matching([Match(Path(), x, Path(), y, 1.0)])
        return 0.0, Matching([])

    def score_self(self, x: T) -> float:
        """Scores an object against itself."""
        return 1.0


class ProductMetric(Metric[T]):
    """A metric that is the product of other metrics."""

    def __init__(self, cls: type[T], field_metrics: dict[str, Metric]):
        if not is_dataclass(cls):
            raise ValueError(f"{cls} has to be a dataclass.")
        self.field_metrics = field_metrics

    def compute(self, x: T, y: T) -> tuple[float, Matching]:
        """Score two objects."""
        field_scores = {
            fld: self.field_metrics[fld].compute(getattr(x, fld), getattr(y, fld)) for fld in self.field_metrics.keys()
        }
        total_score = reduce(mul, (s for s, _ in field_scores.values()), 1.0)

        def _matching():
            yield Match(Path(), x, Path(), y, total_score)
            for fld, (s, matching) in field_scores.items():
                for m in matching.matches:
                    yield Match(m.pred_path.prepend(fld), m.pred, m.ref_path.prepend(fld), m.ref, m.score)

        return total_score, Matching(_matching())


class UnionMetric(Metric[T]):
    """A metric that is the union of other metrics."""

    def __init__(self, cls: type[T], case_metrics: dict[type, Metric]):
        if get_origin(cls) is not Union:
            raise ValueError(f"{cls} has to be a union.")
        self.case_metrics = case_metrics

    def compute(self, x: T, y: T) -> tuple[float, Matching]:
        """Score two objects."""
        x_type = type(x)
        y_type = type(y)
        if x_type != y_type:
            return 0.0, Matching([])
        return self.case_metrics[x_type].compute(x, y)

    def score_self(self, x: T) -> float:
        """Scores an object against itself."""
        return 1.0


@dataclass(eq=True, frozen=True)
class Variable:
    """A variable in latent matchings."""

    name: str

    latent_metric: ClassVar[Metric["Variable"]] = Metric.from_function(lambda x, y: 1.0)


@runtime_checkable
class HasMetric(Protocol[T]):
    """Protocol for classes that have a metric."""

    metric: Metric[T]


@runtime_checkable
class HasLatentMetric(Protocol[T]):
    """Protocol for classes that have a latent metric."""

    latent_metric: Metric[T]

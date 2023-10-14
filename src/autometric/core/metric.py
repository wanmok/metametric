"""Metric interface and implementations for commonly used metrics."""
from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from functools import reduce
from operator import mul
from typing import (Callable, ClassVar, Collection, Dict, Generic, Protocol,
                    Type, TypeVar, Union, get_origin, runtime_checkable)

import numpy as np

S = TypeVar("S", contravariant=True)
T = TypeVar("T", contravariant=True)
U = TypeVar("U")


class Metric(Generic[T]):
    """Metric interface."""

    @abstractmethod
    def score(self, x: T, y: T) -> float:
        """Score two objects."""
        raise NotImplementedError()

    def score_self(self, x: T) -> float:
        """Scores an object against itself.

        In many cases there is a faster way to compute this than the general pair case.
        In such cases, please override this function.
        """
        return self.score(x, x)

    def gram_matrix(self, xs: Collection[T], ys: Collection[T]) -> np.ndarray:
        """Compute the gram matrix of the metric."""
        return np.array([[self.score(x, y) for y in ys] for x in xs])

    def contramap(self, f: Callable[[S], T]) -> "Metric[S]":
        """Returns a new metric by first preprocess the objects by a given function."""
        return ContramappedMetric(self, f)

    @staticmethod
    def from_function(f: Callable[[T, T], float]) -> "Metric[T]":
        """Create a metric from a function.

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

    def score(self, x: T, y: T) -> float:
        """Score two objects."""
        return self.f(x, y)


class ContramappedMetric(Metric[S]):
    """A metric contramapped by a function."""

    def __init__(self, inner: Metric[T], f: Callable[[S], T]):
        self.inner = inner
        self.f = f

    def score(self, x: S, y: S) -> float:
        """Score two objects."""
        return self.inner.score(self.f(x), self.f(y))

    def score_self(self, x: S) -> float:
        """Scores an object against itself."""
        return self.inner.score_self(self.f(x))


class DiscreteMetric(Metric[T]):
    """A metric for discrete objects."""

    def __init__(self, cls: Type[T]):
        if getattr(cls, "__eq__", None) is None:
            raise ValueError("Class must implement __eq__")

    def score(self, x: T, y: T) -> float:
        """Score two objects."""
        return float(x == y)

    def score_self(self, x: T) -> float:
        """Scores an object against itself."""
        return 1.0


class ProductMetric(Metric[T]):
    """A metric that is the product of other metrics."""

    def __init__(self, cls: Type[T], field_metrics: Dict[str, Metric]):
        if not is_dataclass(cls):
            raise ValueError(f"{cls} has to be a dataclass.")
        self.field_metrics = field_metrics

    def score(self, x: T, y: T) -> float:
        """Score two objects."""
        return reduce(
            mul, (self.field_metrics[fld].score(getattr(x, fld), getattr(y, fld)) for fld in self.field_metrics.keys())
        )


class UnionMetric(Metric[T]):
    """A metric that is the union of other metrics."""

    def __init__(self, cls: Type[T], case_metrics: Dict[type, Metric]):
        if get_origin(cls) is not Union:
            raise ValueError(f"{cls} has to be a union.")
        self.case_metrics = case_metrics

    def score(self, x: T, y: T) -> float:
        """Score two objects."""
        x_type = type(x)
        y_type = type(y)
        if x_type != y_type:
            return 0.0
        return self.case_metrics[x_type].score(x, y)

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

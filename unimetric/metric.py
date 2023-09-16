from abc import abstractmethod
from dataclasses import is_dataclass
from functools import reduce
from operator import mul
from typing import Callable, Dict, Generic, Sequence, Set, Type, TypeVar, Collection, Union, get_origin
import numpy as np

T = TypeVar('T', contravariant=True)
U = TypeVar('U')


class Metric(Generic[T]):

    @abstractmethod
    def score(self, x: T, y: T) -> float:
        raise NotImplementedError()

    def score_self(self, x: T) -> float:
        """
        Scores an object against itself.
        In many cases there is a faster way to compute this than the general pair case.
        In such cases, please override this function.
        """
        return self.score(x, x)

    def gram_matrix(self, xs: Collection[T], ys: Collection[T]) -> np.ndarray:
        return np.array([
            [self.score(x, y) for y in ys]
            for x in xs
        ])

    @staticmethod
    def from_function(f: Callable[[T, T], float]) -> "Metric[T]":
        return MetricFromFunction(f)


class MetricFromFunction(Metric[T]):
    def __init__(self, f: Callable[[T, T], float]):
        self.f = f

    def score(self, x: T, y: T) -> float:
        return self.f(x, y)


class ContramappedMetric(Metric[T]):
    def __init__(self, inner: Metric[U], f: Callable[[T], U]):
        self.inner = inner
        self.f = f

    def score(self, x: T, y: T) -> float:
        return self.inner.score(self.f(x), self.f(y))

    def score_self(self, x: T) -> float:
        return self.inner.score_self(self.f(x))


class DiscreteMetric(Metric[T]):
    def __init__(self, cls: Type[T]):
        if getattr(cls, "__eq__", None) is None:
            raise ValueError("Class must implement __eq__")

    def score(self, x: T, y: T) -> float:
        return float(x == y)

    def score_self(self, x: T) -> float:
        return 1.0


class Jaccard(Metric[T]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        sxy = self.inner.score(x, y)
        sxx = self.inner.score_self(x)
        syy = self.inner.score_self(y)
        return sxy / (sxx + syy - sxy)

    def score_self(self, x: T) -> float:
        return 1.0


class Precision(Metric[T]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        sxy = self.inner.score(x, y)
        sxx = self.inner.score_self(x)
        return sxy / sxx

    def score_self(self, x: T) -> float:
        return 1.0


class Recall(Metric[T]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        sxy = self.inner.score(x, y)
        syy = self.inner.score_self(y)
        return sxy / syy

    def score_self(self, x: T) -> float:
        return 1.0


class FScore(Metric[T]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        sxy = self.inner.score(x, y)
        sxx = self.inner.score_self(x)
        syy = self.inner.score_self(y)
        p = sxy / sxx
        r = sxy / syy
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)

    def score_self(self, x: T) -> float:
        return 1.0


class ProductMetric(Metric[T]):
    def __init__(self, cls: type, field_metrics: Dict[str, Metric]):
        if not is_dataclass(cls):
            raise ValueError(f"{cls} has to be a dataclass.")
        self.field_metrics = field_metrics

    def score(self, x: T, y: T) -> float:
        return reduce(
            mul,
            (
                self.field_metrics[fld].score(getattr(x, fld), getattr(y, fld))
                for fld in self.field_metrics.keys()
            )
        )


class UnionMetric(Metric[T]):
    def __init__(self, cls: type, case_metrics: Dict[type, Metric]):
        if not get_origin(cls) is Union:
            raise ValueError(f"{cls} has to be a union.")
        self.case_metrics = case_metrics

    def score(self, x: T, y: T) -> float:
        x_type = type(x)
        y_type = type(y)
        if x_type != y_type:
            return 0.0
        return self.case_metrics[x_type].score(x, y)

    def score_self(self, x: T) -> float:
        return 1.0

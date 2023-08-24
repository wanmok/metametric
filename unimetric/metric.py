from abc import abstractmethod
from functools import reduce
from operator import mul
from typing import Callable, Dict, Generic, Set, Type, TypeVar
import numpy as np
import scipy.optimize as spo

T = TypeVar('T', contravariant=True)
U = TypeVar('U')


class Metric(Generic[T]):

    @abstractmethod
    def score(self, x: T, y: T) -> float:
        raise NotImplementedError()

    def score_self(self, x: T) -> float:
        """
        Score an object against itself.
        In a lot of cases there is a faster way to compute this than the general pair case.
        In such cases, please override this function.
        """
        return self.score(x, x)


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


class ProductMetric(Metric[T]):
    def __init__(self, cls: type, field_metrics: Dict[str, Metric]):
        if getattr(cls, "__dataclass_fields__", None) is None:
            raise ValueError("Class must be a dataclass")
        self.field_metrics = field_metrics

    def score(self, x: T, y: T) -> float:
        return reduce(
            mul,
            (
                self.field_metrics[fld].score(getattr(x, fld), getattr(y, fld))
                for fld in self.field_metrics.keys()
            )
        )


class AlignmentMetric(Metric[Set[U]]):
    def __init__(self, inner: Metric[U]):
        self.inner = inner

    def score(self, x: Set[U], y: Set[U]) -> float:
        if isinstance(self.inner, DiscreteMetric):
            return len(x & y)
        # else, we need to solve the assignment problem
        m = np.array([
            [
                self.inner.score(u, v)
                for v in y
            ]
            for u in x
        ])
        row_idx, col_idx = spo.linear_sum_assignment(
            cost_matrix=m,
            maximize=True,
        )
        return m[row_idx, col_idx].sum()

    def score_self(self, x: Set[U]) -> float:
        return sum(self.inner.score_self(u) for u in x)


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

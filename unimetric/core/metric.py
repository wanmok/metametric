from abc import abstractmethod
from functools import reduce
from operator import mul
from typing import Dict, Generic, Set, Type, TypeVar
import networkx as nx

T = TypeVar('T', contravariant=True)


class Metric(Generic[T]):
    @abstractmethod
    def score(self, x: T, y: T) -> float:
        raise NotImplementedError()


class DiscreteMetric(Metric[T]):
    def __init__(self, cls: Type[T]):
        if getattr(cls, "__eq__", None) is None:
            raise ValueError("Class must implement __eq__")

    def score(self, x: T, y: T) -> float:
        return float(x == y)


class ProductMetric(Metric[T]):
    def __init__(self, cls: Type[T], field_metrics: Dict[str, Metric]):
        if getattr(cls, "__dataclass_fields__", None) is None:
            raise ValueError("Class must be a dataclass")
        self.field_names = cls.__dataclass_fields__.keys()
        self.field_metrics = field_metrics

    def score(self, x: T, y: T) -> float:
        return reduce(
            mul,
            (
                self.field_metrics[fld].score(getattr(x, fld), getattr(y, fld))
                for fld in self.field_names
            )
        )


class AlignmentMetric(Metric[Set[T]]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        # TODO(tongfei): if inner metric is discrete, use direct counting
        g = nx.Graph()
        g.add_nodes_from(((0, u) for u in x), bipartite=0)
        g.add_nodes_from(((1, v) for v in y), bipartite=1)
        g.add_weighted_edges_from(
            ((0, u), (1, v), -self.inner.score(u, v))
            for u in x
            for v in y
        )
        matching = nx.bipartite.minimum_weight_full_matching(g)
        return -sum(
            g[(0, u)][matching[(0, u)]]['weight']
            for u in x
            if (0, u) in matching
        )


class Jaccard(Metric[T]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        sxy = self.inner.score(x, y)
        sxx = self.inner.score(x, x)
        syy = self.inner.score(y, y)
        return sxy / (sxx + syy - sxy)


class Dice(Metric[T]):
    def __init__(self, inner: Metric[T]):
        self.inner = inner

    def score(self, x: T, y: T) -> float:
        sxy = self.inner.score(x, y)
        sxx = self.inner.score(x, x)
        syy = self.inner.score(y, y)
        p = sxy / sxx
        r = sxy / syy
        return 2 * p * r / (p + r)

"""This module contains the domain-specific language (DSL) for defining metrics.

It is recommended to import this module as `mm` (short for metametric) for brevity:
```py
import metametric.dsl as mm
```
"""
import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Callable,
    Collection,
    Dict,
    Generic,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_origin,
    TYPE_CHECKING,
)

if sys.version_info >= (3, 10):
    from types import EllipsisType as Ell
else:
    if TYPE_CHECKING:
        from builtins import ellipsis as Ell
    else:
        Ell = type(...)

from metametric.core.constraint import MatchingConstraint
from metametric.core.matching import (
    GraphMatchingMetric,
    LatentSetMatchingMetric,
    SequenceMatchingMetric,
    SetMatchingMetric,
)
from metametric.core.decorator import derive_metric
from metametric.core.graph import Graph
from metametric.core.metric import ContramappedMetric, DiscreteMetric, Metric, ProductMetric, UnionMetric
from metametric.core.metric_suite import MetricFamily, MetricSuite, MultipleMetricFamilies
from metametric.core.normalizers import NormalizedMetric, Normalizer
from metametric.core.reduction import MacroAverage, MicroAverage, MultipleReductions, Reduction

T = TypeVar("T", contravariant=True)
S = TypeVar("S")

DslConfig = Union[
    Type[T],
    Tuple[Type[T], Union[MatchingConstraint, str]],
    Tuple[Type[T], Union[MatchingConstraint, str], Union[Normalizer, str, None]],
]


@dataclass
class _Config(Generic[T]):
    cls: Type[T]
    constraint: MatchingConstraint = MatchingConstraint.ONE_TO_ONE
    normalizer: Optional[Normalizer] = None

    @classmethod
    def standardize(cls, config: DslConfig[T]) -> "_Config[T]":
        if isinstance(config, tuple):
            if len(config) == 2:
                t, constraint = config
                normalizer = None
            elif len(config) == 3:
                t, constraint, normalizer = config
                if isinstance(normalizer, str):
                    normalizer = Normalizer.from_str(normalizer)
            else:
                raise ValueError(f"Invalid config: {config}")
        else:
            t = config
            constraint = MatchingConstraint.ONE_TO_ONE
            normalizer = None
        if isinstance(constraint, str):
            constraint = MatchingConstraint.from_str(constraint)
        return cls(t, constraint, normalizer)


def from_func(func: Callable[[T, T], float]) -> Metric[T]:
    """Create a metric from a binary function.

    Args:
        func: A binary function that takes two objects of type `T` and returns a float.

    Returns:
        A `Metric` object that wraps around the given function.
    """
    return Metric.from_function(func)


def preprocess(f: Callable[[S], T], m: Metric[T]) -> Metric[S]:
    r"""Returns a `Metric` object by first preprocess the input by some function then apply an inner metric.

    This is the `contramap` operation on the metric functor.

    \[ \phi(x, y) = m(f(x), f(y)) \]

    Args:
        f: A function that preprocesses the input to be compared.
        m: The inner metric to be applied on the preprocessed input.

    Returns:
        A `Metric` object that preprocesses the input by $f$ then apply $m$ on the preprocessed input.
    """
    return ContramappedMetric(m, f)


class auto:
    """Automatically derive a similarity from a type."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Metric[T]:
        """Automatically derive a similarity from a type, optionally with a constraint and a normalizer.

        Examples:
            `mm.auto[AMR]`: Derives a similarity for type `AMR`.

            `mm.auto[AMR, '<->', 'f1']`: Derives a similarity for type `AMR` with 1:1 constraint and $F_1$ normalizer.

        Args:
            config: A type, optionally with a constraint and optionally a normalizer.

        Returns:
            A similarity over the given type.

        """
        cfg = _Config.standardize(config)
        return derive_metric(cfg.cls, constraint=cfg.constraint)


class discrete:
    """Returns a discrete similarity over a type with `__eq__` implemented."""
    def __class_getitem__(cls, t: Type[T]) -> Metric[T]:
        r"""Returns a discrete similarity over a type with `__eq__` implemented.

        \[ \delta(x, y) = \begin{cases} 1 & \textrm{if} \ x = y \\
        0 & \textrm{if} \  x \neq y \end{cases} \]

        Args:
            t: The type to be compared.

        Returns:
            A discrete similarity $\delta$ over the given type.
        """
        return DiscreteMetric(t)


class dataclass:
    """Constructs a similarity for a dataclass `T`."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Callable[[Dict[str, Union[Ell, Metric]]], Metric[T]]:
        r"""Constructs a similarity for a dataclass `T`.

        It takes the product of the metrics for each of its fields defined in `field_metrics: Dict[str, Metric[Any]]`.

        \[ \phi(x, y) = \prod_{(f, m_f) \in M} m_f(x.\!f, y.\!f) \]

        Examples:
            Consider the following dataclass:
            ```py
            @dataclass
            class Relation:
                type: str
                subj: Mention
                obj: Mention
            ```

            To derive a similarity for `Relation`:
            ```py
            mm.dataclass[Relation]({
                "type": mm.discrete[str],
                "subj": mm.auto[Mention],
                "obj": mm.auto[Mention],
            })
            ```
            One could just write `...` for automatically derived metrics:
            ```py
            mm.dataclass[Relation]({
                "type": ...,
                "subj": ...,
                "obj": ...,
            })
            ```
            And this could be simplified to just `mm.auto[Relation]`.

        Args:
            config: A type, optionally with a constraint and optionally a normalizer.

        Returns:
            A function that takes a dictionary of metrics for each field and returns a metric over the dataclass.
        """
        cfg = _Config.standardize(config)
        assert is_dataclass(cfg.cls)

        def product_metric(field_metrics: Dict[str, Union[Ell, Metric]]) -> Metric[T]:
            field_types = {fld.name: fld.type for fld in fields(cfg.cls)}
            field_metrics_no_ell: Dict[str, Metric] = {
                fld: (auto[field_types[fld], cfg.constraint] if metric is ... else metric)
                for fld, metric in field_metrics.items()
            }
            return ProductMetric(cls=cfg.cls, field_metrics=field_metrics_no_ell)

        return product_metric



class union:
    """Constructs a similarity for a union type `T`."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Callable[[Dict[Type, Union[Ell, Metric]]], Metric[T]]:
        r"""Constructs a similarity for a union type `T`.

        It selects the metric of the union cases defined in `case_metrics: Dict[type, Metric[Any]]`.

        \[ \phi(x, y) = \sum_{(c, m_c) \in M} \mathbb{1}_{x \in c}\mathbb{1}_{y \in c} m_c(x, y) \]

        Args:
            config: A type, optionally with a constraint and optionally a normalizer.

        Returns:
            A function that takes a dictionary of metrics for each case and returns a metric over the union type.

        """
        cfg = _Config.standardize(config)

        def union_metric(case_metrics: Dict[Type, Union[Ell, Metric]]) -> Metric[T]:
            assert get_origin(cfg.cls) is Union
            case_metrics_no_ell: Dict[type, Metric] = {
                case: (auto[case, cfg.constraint] if metric is ... else metric)
                for case, metric in case_metrics.items()
            }
            return UnionMetric(cls=cfg.cls, case_metrics=case_metrics_no_ell)

        return union_metric


class set_matching:
    """Constructs a set matching metric for a collection type `Collection[T]`."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Collection[T]]]:
        r"""Constructs a set matching metric for a collection type `Collection[T]` given an internal metric `Metric[T]`.

        The unnormalized version is given by

        \[ \Sigma(X, Y) = \max_{M^\diamond} \sum_{(u, v) \in M^\diamond} \phi_T(u, v) \]

        where $M^\diamond$ is a matching between $X$ and $Y$ according to the specified matching constraint.

        It could be further normalized by a normalizer $\mathsf{N}$ specified in the config.

        Examples:
            ```py
            mm.set_matching[EntitySet, '<->', 'f1'](mm.auto[Entity])
            ```

        Args:
            config: The internal element type `T`, optionally with a matching constraint and optionally a normalizer.

        Returns:
            A function that takes an internal element metric returns a metric over the collection type.
        """
        cfg = _Config.standardize(config)

        def matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Collection[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = SetMatchingMetric(inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return matching_metric


class sequence_matching:
    """Constructs a sequence matching metric for a collection type `Sequence[T]`."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Sequence[T]]]:
        r"""Constructs a sequence matching metric for a sequence `Sequence[T]` given an internal metric `Metric[T]`.

        The unnormalized version is given by

        \[ \Sigma(X, Y) = \max_{M^\diamond} \sum_{(u, v) \in M^\diamond} \phi_T(u, v) \]

        such that for all $(u, v) \in M^\diamond$ and $(u^\prime, v^\prime) \in M^\diamond$,
        $u \le u^\prime$ if and only if $v \le v^\prime$.
        Here $\le$ is the total order of **precedence relation** on sequence elements in $X$ and $Y$,
        and $M^\diamond$ is a matching between $X$ and $Y$ according to the specified matching constraint.

        It could be further normalized by a normalizer $\mathsf{N}$ specified in the config.

        Args:
            config: The internal element type `T`, optionally with a matching constraint and optionally a normalizer.

        Returns:
            A function that takes an internal element metric returns a metric over the sequence type.
        """
        cfg = _Config.standardize(config)

        def matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Sequence[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = SequenceMatchingMetric(inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return matching_metric


class graph_matching:
    """Constructs a graph matching metric for a graph type that conforms to `networkx.Digraph`."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Graph[T]]]:
        r"""Constructs a graph matching metric for a graph type `Graph[T]` given an internal metric `Metric[T]`.

        Here `Graph[T]` is a minimal protocol that implements a graph: `networkx.Digraph` is a conforming instantiation.

        The unnormalized version is given by

        \[ \Sigma(X, Y) = \max_{M^\diamond} \sum_{(u, v) \in M^\diamond} \phi_T(u, v) \]

        such that for all $(u, v) \in M^\diamond$ and $(u^\prime, v^\prime) \in M^\diamond$,
        $u \le u^\prime$ if and only if $v \le v^\prime$.
        Here $\le$ is the pre-order of **reachability relation** on graph vertices in $X$ and $Y$,
        and $M^\diamond$ is a matching between $X$ and $Y$ according to the specified matching constraint.

        It could be further normalized by a normalizer $\mathsf{N}$ specified in the config.

        Args:
            config: The internal element type `T`, optionally with a matching constraint and optionally a normalizer.

        Returns:
            A function that takes an internal element metric returns a metric over the sequence type.
        """
        cfg = _Config.standardize(config)

        def matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Graph[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = GraphMatchingMetric(inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return matching_metric



class latent_set_matching:
    """Constructs a set matching metric for a collection type `Collection[T]` where `T` contains `Variable`s."""
    def __class_getitem__(cls, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Collection[T]]]:
        r"""Constructs a latent set matching metric for a collection type `Collection[T]` where `T` has `Variable`s.

        The unnormalized version is given by

        \[ \Sigma(X, Y) = \max_{M^{\leftrightarrow}_V, M^\diamond} \sum_{(u, v) \in M^\diamond} \phi_T(u, v) \]

        where $M^\diamond$ is a matching between $X$ and $Y$ according to the specified matching constraint, and
        $M^\leftrightarrow_V$ is a one-to-one matching between the variables in $X$ and $Y$.

        It could be further normalized by a normalizer $\mathsf{N}$ specified in the config.

        Examples:
            ```py
            mm.latent_set_matching[Prop, '<->', 'f1'](...)
            ```

        Args:
            config: The internal element type `T`, optionally with a matching constraint and optionally a normalizer.

        Returns:
            A function that takes an internal element metric returns a metric over the collection type.
        """
        cfg = _Config.standardize(config)

        def latent_matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Collection[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = LatentSetMatchingMetric(cfg.cls, inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return latent_matching_metric


class normalize:
    """Normalizes a metric with a normalizer."""
    def __class_getitem__(self, normalizer: Union[Normalizer, str]) -> Callable[[Metric[T]], Metric[T]]:
        """Normalizes a metric with a normalizer.

        The normalizer can be a `Normalizer` object or
        a string in `none`, `precision`, `recall`, `dice`, `jaccard`, or `f{beta}` that specifies a normalizer.

        Args:
            normalizer: A `Normalizer` object or a string defined above.

        Returns:
            A function that normalizes a metric by the given normalizer.
        """
        if isinstance(normalizer, str):
            normalizer_obj = Normalizer.from_str(normalizer)

        def normalize(metric: Metric[T]) -> Metric[T]:
            return metric if normalizer_obj is None else NormalizedMetric(metric, normalizer_obj)

        return normalize


def macro_average(normalizers: Collection[Union[Normalizer, str]]) -> Reduction:
    """Constructs a macro-average reduction strategy from a collection of normalizers."""
    normalizer_objs = [
        Normalizer.from_str(normalizer) if isinstance(normalizer, str) else normalizer
        for normalizer in normalizers
    ]
    return MacroAverage(normalizer_objs)


def micro_average(normalizers: Collection[Union[Normalizer, str]]) -> Reduction:
    """Constructs a micro-average reduction strategy from a collection of normalizers."""
    normalizer_objs = [
        Normalizer.from_str(normalizer) if isinstance(normalizer, str) else normalizer
        for normalizer in normalizers
    ]
    return MicroAverage(normalizer_objs)


def family(metric: Metric[T], reduction: Union[Reduction, Dict[str, Reduction]]) -> MetricFamily[T]:
    """Constructs a metric family from a metric and a reduction strategy."""
    if isinstance(reduction, dict):
        reduction = MultipleReductions(reduction)
    return MetricFamily(metric, reduction)


def suite(collection: Dict[str, MetricSuite[T]]) -> MetricSuite[T]:
    """Combines a collection of metric suites into a single metric suite."""
    return MultipleMetricFamilies(collection)

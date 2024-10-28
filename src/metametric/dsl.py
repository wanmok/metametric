"""This module contains the domain-specific language (DSL) for defining metrics."""

import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_origin,
    TYPE_CHECKING,
)
from collections.abc import Collection, Sequence

if sys.version_info >= (3, 10):
    from types import EllipsisType as Ell
else:
    if TYPE_CHECKING:
        from builtins import ellipsis as Ell
    else:
        Ell = type(...)

from metametric.core.constraint import MatchingConstraint
from metametric.core.matching_metrics import (
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
from metametric.core.matching import Matching, Match, Hook


T = TypeVar("T", contravariant=True)
S = TypeVar("S")

DslConfig = Union[
    type[T],
    tuple[type[T], Union[MatchingConstraint, str]],
    tuple[type[T], Union[MatchingConstraint, str], Union[Normalizer, str, None]],
]


@dataclass
class _Config(Generic[T]):
    cls: type[T]
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
    """Create a metric from a binary function."""
    return Metric.from_function(func)


def preprocess(func: Callable[[S], T], m: Metric[T]) -> Metric[S]:
    """Preprocess the input by some function then apply a metric.

    This is the `contramap` operation on the metric functor.
    """
    return ContramappedMetric(m, func)


class _Auto:
    def __getitem__(self, config: DslConfig[T]) -> Metric[T]:
        cfg = _Config.standardize(config)
        return derive_metric(cfg.cls, constraint=cfg.constraint)


auto = _Auto()


class _Discrete:
    def __getitem__(self, t: type[T]) -> Metric[T]:
        return DiscreteMetric(t)


discrete = _Discrete()


class _DataClass:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[dict[str, Union[Ell, Metric]]], Metric[T]]:
        cfg = _Config.standardize(config)
        assert is_dataclass(cfg.cls)

        def product_metric(field_metrics: dict[str, Union[Ell, Metric]]) -> Metric[T]:
            field_types = {fld.name: fld.type for fld in fields(cfg.cls)}
            field_metrics_no_ell: dict[str, Metric] = {
                fld: (auto[field_types[fld], cfg.constraint] if metric is ... else metric)  # pyright: ignore
                for fld, metric in field_metrics.items()
            }
            return ProductMetric(cls=cfg.cls, field_metrics=field_metrics_no_ell)

        return product_metric


dataclass = _DataClass()


class _Union:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[dict[type, Union[Ell, Metric]]], Metric[T]]:
        cfg = _Config.standardize(config)

        def union_metric(case_metrics: dict[type, Union[Ell, Metric]]) -> Metric[T]:
            assert get_origin(cfg.cls) is Union
            case_metrics_no_ell: dict[type, Metric] = {
                case: (auto[case, cfg.constraint] if metric is ... else metric) for case, metric in case_metrics.items()
            }
            return UnionMetric(cls=cfg.cls, case_metrics=case_metrics_no_ell)

        return union_metric


union = _Union()


class _SetMatching:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Collection[T]]]:
        cfg = _Config.standardize(config)

        def matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Collection[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = SetMatchingMetric(inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return matching_metric


set_matching = _SetMatching()


class _SequenceMatching:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Sequence[T]]]:
        cfg = _Config.standardize(config)

        def matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Sequence[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = SequenceMatchingMetric(inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return matching_metric


sequence_matching = _SequenceMatching()


class _GraphMatching:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Graph[T]]]:
        cfg = _Config.standardize(config)

        def matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Graph[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = GraphMatchingMetric(inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return matching_metric


graph_matching = _GraphMatching()


class _LatentSetMatching:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Collection[T]]]:
        cfg = _Config.standardize(config)

        def latent_matching_metric(inner: Union[Ell, Metric[T]]) -> Metric[Collection[T]]:
            if inner is ...:
                inner = auto[cfg.cls, cfg.constraint]
            match = LatentSetMatchingMetric(cfg.cls, inner, constraint=cfg.constraint)
            if cfg.normalizer is not None:
                match = NormalizedMetric(match, cfg.normalizer)
            return match

        return latent_matching_metric


latent_set_matching = _LatentSetMatching()


class _Normalize:
    def __getitem__(self, normalizer: Union[Normalizer, str]) -> Callable[[Metric[T]], Metric[T]]:
        if isinstance(normalizer, str):
            normalizer_obj = Normalizer.from_str(normalizer)
        else:
            normalizer_obj = None

        def _normalize(metric: Metric[T]) -> Metric[T]:
            return metric if normalizer_obj is None else NormalizedMetric(metric, normalizer_obj)

        return _normalize


normalize = _Normalize()


def macro_average(normalizers: Collection[Union[Normalizer, str]]) -> Reduction:
    """Macro-average reduction."""
    normalizer_objs = [
        Normalizer.from_str(normalizer) if isinstance(normalizer, str) else normalizer for normalizer in normalizers
    ]
    return MacroAverage(normalizer_objs)


def micro_average(normalizers: Collection[Union[Normalizer, str]]) -> Reduction:
    """Micro-average reduction."""
    normalizer_objs = [
        Normalizer.from_str(normalizer) if isinstance(normalizer, str) else normalizer for normalizer in normalizers
    ]
    return MicroAverage(normalizer_objs)


def family(metric: Metric[T], reduction: Union[Reduction, dict[str, Reduction]]) -> MetricFamily[T]:
    """Creates a metric family."""
    if isinstance(reduction, dict):
        reduction = MultipleReductions(reduction)
    return MetricFamily(metric, reduction)


def suite(collection: dict[str, MetricSuite[T]]) -> MetricSuite[T]:
    """Creates a metric suite."""
    return MultipleMetricFamilies(collection)


__all__ = [
    "from_func",
    "preprocess",
    "auto",
    "discrete",
    "dataclass",
    "union",
    "set_matching",
    "sequence_matching",
    "graph_matching",
    "latent_set_matching",
    "normalize",
    "macro_average",
    "micro_average",
    "family",
    "suite",
    "Metric",
    "Match",
    "Matching",
    "Hook",
]

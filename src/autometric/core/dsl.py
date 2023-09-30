"""This module contains the domain-specific language (DSL) for defining metrics."""
from dataclasses import dataclass, fields, is_dataclass
from typing import (Callable, Collection, Dict, Generic, Optional, Sequence,
                    Tuple, Type, TypeVar, Union, get_origin)

from autometric.core.alignment import (AlignmentConstraint,
                                       GraphAlignmentMetric,
                                       LatentSetAlignmentMetric,
                                       SequenceAlignmentMetric,
                                       SetAlignmentMetric)
from autometric.core.decorator import derive_metric
from autometric.core.graph import Graph
from autometric.core.metric import (ContramappedMetric, DiscreteMetric, Metric,
                                    ProductMetric, UnionMetric, WrappedMetric)
from autometric.core.metric_suite import (MetricFamily, MetricSuite,
                                          MultipleMetricFamilies)
from autometric.core.normalizers import NormalizedMetric, Normalizer
from autometric.core.reduction import (MacroAverage, MicroAverage,
                                       MultipleReductions, Reduction)

T = TypeVar("T")
S = TypeVar("S")

Ell = type(...)  # `ellipsis` is private


DslConfig = Union[
    Type[T],
    Tuple[Type[T], Union[AlignmentConstraint, str]],
    Tuple[Type[T], Union[AlignmentConstraint, str], Union[Normalizer, str, None]],
]

@dataclass
class _Config(Generic[T]):
    cls: Type[T]
    constraint: AlignmentConstraint = AlignmentConstraint.ONE_TO_ONE
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
            constraint = AlignmentConstraint.ONE_TO_ONE
            normalizer = None
        if isinstance(constraint, str):
            constraint = AlignmentConstraint.from_str(constraint)
        return cls(t, constraint, normalizer)


def from_func(func: Callable[[T, T], float]) -> Metric[T]:
    """Create a metric from a binary function."""
    return Metric.from_function(func)


def preprocess(func: Callable[[S], T], m: Metric[T]) -> Metric[S]:
    """Preprocess the input by some function then apply a metric.

    This is the `contramap` operation on the metric functor.
    """
    return ContramappedMetric(m, func)


def wrapped(metrics: Tuple[Metric[T], ...], f: Callable[[float, ...], float]) -> Metric[T]:
    """Execute an additional downstream function after metrics has been computed.

    This is useful to compute a weighted sum of multiple metrics.
    """
    return WrappedMetric(metrics, f)


class _Auto:
    def __getitem__(self, config: DslConfig[T]) -> Metric[T]:
        config = _Config.standardize(config)
        return derive_metric(config.cls, constraint=config.constraint)


auto = _Auto()


class _Discrete:
    def __getitem__(self, t: Type[T]) -> Metric[T]:
        return DiscreteMetric(t)


discrete = _Discrete()


class _DataClass:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Dict[str, Union[Ell, Metric]]], Metric[T]]:
        config = _Config.standardize(config)
        assert is_dataclass(config.cls)

        def product_metric(field_metrics: Dict[str, Union[Ell, Metric]]) -> Metric[T]:
            field_types = {fld.name: fld.type for fld in fields(config.cls)}
            field_metrics = {
                fld: (auto[field_types[fld], config.constraint] if metric is ... else metric)
                for fld, metric in field_metrics.items()
            }
            return ProductMetric(cls=config.cls, field_metrics=field_metrics)

        return product_metric


dataclass = _DataClass()


class _Union:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Dict[Type, Union[Ell, Metric]]], Metric[T]]:
        config = _Config.standardize(config)

        def union_metric(case_metrics: Dict[Type, Union[Ell, Metric]]) -> Metric[T]:
            assert get_origin(config.cls) is Union
            case_metrics = {
                case: (auto[case, config.constraint] if metric is ... else metric)
                for case, metric in case_metrics.items()
            }
            return UnionMetric(cls=config.cls, case_metrics=case_metrics)

        return union_metric


union = _Union()


class _SetAlignment:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Collection[T]]]:
        config = _Config.standardize(config)

        def alignment_metric(inner: Union[Ell, Metric[T]]) -> Metric[Collection[T]]:
            if inner is ...:
                inner = auto[config.cls, config.constraint]
            match = SetAlignmentMetric(inner, constraint=config.constraint)
            if config.normalizer is not None:
                match = NormalizedMetric(match, config.normalizer)
            return match

        return alignment_metric


set_alignment = _SetAlignment()


class _SequenceAlignment:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Sequence[T]]]:
        config = _Config.standardize(config)

        def alignment_metric(inner: Union[Ell, Metric[T]]) -> Metric[Sequence[T]]:
            if inner is ...:
                inner = auto[config.cls, config.constraint]
            match = SequenceAlignmentMetric(inner, constraint=config.constraint)
            if config.normalizer is not None:
                match = NormalizedMetric(match, config.normalizer)
            return match

        return alignment_metric


sequence_alignment = _SequenceAlignment()


class _GraphAlignment:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Graph[T]]]:
        config = _Config.standardize(config)

        def alignment_metric(inner: Union[Ell, Metric[T]]) -> Metric[Graph[T]]:
            if inner is ...:
                inner = auto[config.cls, config.constraint]
            match = GraphAlignmentMetric(inner, constraint=config.constraint)
            if config.normalizer is not None:
                match = NormalizedMetric(match, config.normalizer)
            return match

        return alignment_metric


graph_alignment = _GraphAlignment()


class _LatentSetAlignment:
    def __getitem__(self, config: DslConfig[T]) -> Callable[[Union[Ell, Metric[T]]], Metric[Collection[T]]]:
        config = _Config.standardize(config)
        def latent_alignment_metric(inner: Union[Ell, Metric[T]]) -> Metric[Collection[T]]:
            if inner is ...:
                inner = auto[config.cls, config.constraint]
            match = LatentSetAlignmentMetric(config.cls, inner, constraint=config.constraint)
            if config.normalizer is not None:
                match = NormalizedMetric(match, config.normalizer)
            return match

        return latent_alignment_metric


latent_set_alignment = _LatentSetAlignment()


class _Normalize:
    def __getitem__(self, normalizer: Union[Normalizer, str]) -> Callable[[Metric[T]], Metric[T]]:
        if isinstance(normalizer, str):
            normalizer = Normalizer.from_str(normalizer)

        def normalize(metric: Metric[T]) -> Metric[T]:
            return NormalizedMetric(metric, normalizer)

        return normalize


normalize = _Normalize()


class _MacroAverage:
    def __call__(self, normalizers: Collection[Union[Normalizer, str]]) -> Reduction:
        normalizers = [
            Normalizer.from_str(normalizer) if isinstance(normalizer, str) else normalizer
            for normalizer in normalizers
        ]
        return MacroAverage(normalizers)


macro_average = _MacroAverage()


class _MicroAverage:
    def __call__(self, normalizers: Collection[Union[Normalizer, str]]) -> Reduction:
        normalizers = [
            Normalizer.from_str(normalizer) if isinstance(normalizer, str) else normalizer
            for normalizer in normalizers
        ]
        return MicroAverage(normalizers)


micro_average = _MicroAverage()


class _Family:
    def __call__(self, metric: Metric[T], reduction: Union[Reduction, Dict[str, Reduction]]) -> MetricFamily[T]:
        if isinstance(reduction, dict):
            reduction = MultipleReductions(reduction)
        return MetricFamily(metric, reduction)


family = _Family()


class _Suite:
    def __call__(self, collection: Dict[str, MetricSuite[T]]) -> MetricSuite[T]:
        return MultipleMetricFamilies(collection)


suite = _Suite()

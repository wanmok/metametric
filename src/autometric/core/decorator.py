"""Decorator for deriving metrics from dataclasses."""
from dataclasses import fields, is_dataclass
from typing import (
    Literal,
    get_args,
    get_origin,
    Collection,
    Annotated,
    Union,
    Protocol,
    runtime_checkable,
    Callable,
    TypeVar,
    Type,
)

from autometric.core.alignment import AlignmentConstraint, AlignmentMetric
from autometric.core.latent_alignment import dataclass_has_variable, LatentAlignmentMetric
from autometric.core.metric import Metric, ProductMetric, DiscreteMetric, UnionMetric
from autometric.core.normalizers import NormalizedMetric, Normalizer

NormalizerLiteral = Literal["none", "jaccard", "dice", "f1"]
ConstraintLiteral = Literal["<->", "<-", "->", "~", "1:1", "1:*", "*:1", "*:*"]

T = TypeVar("T", contravariant=True)


@runtime_checkable
class HasMetric(Protocol[T]):
    """Protocol for classes that have a metric."""

    metric: Metric[T]


@runtime_checkable
class HasLatentMetric(Protocol[T]):
    """Protocol for classes that have a latent metric."""

    latent_metric: Metric[T]


def derive_metric(cls: Type, constraint: AlignmentConstraint) -> Metric:
    """Derive a unified metric from any type.

    Args:
        cls (`Type`): The type to derive the metric from.
        constraint (`AlignmentConstraint`): The alignment constraint to use.

    Returns:
        `Metric`: The derived metric.
    """
    # if the type is annotated with a metric instance, use the metric annotation
    if get_origin(cls) is Annotated:
        metric = get_args(cls)[1]
        if isinstance(metric, Metric):
            return metric

    # if an explicit metric is defined, use it
    if isinstance(cls, HasMetric):
        return cls.metric

    cls_origin = get_origin(cls)
    if isinstance(cls, HasLatentMetric):
        return cls.latent_metric

    # derive product metric from dataclass
    elif is_dataclass(cls):
        return ProductMetric(
            cls=cls, field_metrics={fld.name: derive_metric(fld.type, constraint=constraint) for fld in fields(cls)}
        )

    # derive union metric from unions
    elif cls_origin is Union:
        return UnionMetric(
            cls=cls, case_metrics={case: derive_metric(case, constraint=constraint) for case in get_args(cls)}
        )

    # derive alignment metric from collections
    elif cls_origin is not None and isinstance(cls_origin, type) and issubclass(cls_origin, Collection):
        elem_type = get_args(cls)[0]
        inner_metric = derive_metric(elem_type, constraint=constraint)
        if dataclass_has_variable(elem_type):
            return LatentAlignmentMetric(
                cls=elem_type,
                inner=inner_metric,
                constraint=constraint,
            )
        else:
            return AlignmentMetric(
                inner=inner_metric,
                constraint=constraint,
            )

    # derive discrete metric from equality
    elif getattr(cls, "__eq__", None) is not None:
        return DiscreteMetric(cls=cls)

    else:
        raise ValueError(f"Could not derive metric from type {cls}.")


def autometric(
    normalizer: Union[NormalizerLiteral, Normalizer] = "none",
    constraint: ConstraintLiteral = "<->",
) -> Callable[[Type], Type]:
    """Decorate a dataclass to have corresponding metric derived.

    Args:
        normalizer (`Union[NormalizerLiteral, Normalizer]`, defaults to "none"):
            The normalizer to use.
        constraint (`ConstraintLiteral`, defaults to "<->"):
            The alignment constraint to use.

    Returns:
        `Callable[[Type], Type]`: The class decorator.
    """

    def class_decorator(cls: Type) -> Type:
        alignment_constraint = {
            "<->": AlignmentConstraint.ONE_TO_ONE,
            "<-": AlignmentConstraint.ONE_TO_MANY,
            "->": AlignmentConstraint.MANY_TO_ONE,
            "~": AlignmentConstraint.MANY_TO_MANY,
            "1:1": AlignmentConstraint.ONE_TO_ONE,
            "1:*": AlignmentConstraint.ONE_TO_MANY,
            "*:1": AlignmentConstraint.MANY_TO_ONE,
            "*:*": AlignmentConstraint.MANY_TO_MANY,
        }[constraint]
        metric = derive_metric(cls, constraint=alignment_constraint)
        if isinstance(normalizer, Normalizer):
            normalized_metric = NormalizedMetric(metric, normalizer=normalizer)
        else:
            normalized_metric = NormalizedMetric(metric, normalizer=Normalizer.from_str(normalizer))
        if dataclass_has_variable(cls):
            setattr(cls, "latent_metric", normalized_metric)  # type: ignore
        else:
            setattr(cls, "metric", normalized_metric)  # type: ignore
        return cls

    return class_decorator

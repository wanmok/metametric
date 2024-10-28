"""Decorator for deriving metrics from dataclasses."""

from dataclasses import fields, is_dataclass
from typing import Annotated, Any, Callable, Literal, TypeVar, Union, get_args, get_origin, Optional
from collections.abc import Collection

from metametric.core.matching_metrics import MatchingConstraint, LatentSetMatchingMetric, SetMatchingMetric
from metametric.core.metric import (
    DiscreteMetric,
    HasLatentMetric,
    HasMetric,
    Metric,
    ProductMetric,
    UnionMetric,
    Variable,
)
from metametric.core.normalizers import NormalizedMetric, Normalizer

NormalizerLiteral = Literal["none", "jaccard", "dice", "f1"]
ConstraintLiteral = Literal["<->", "<-", "->", "~", "1:1", "1:*", "*:1", "*:*"]

T = TypeVar("T", contravariant=True)


def may_be_variable(cls: Any) -> bool:
    """Check if a type may be a `Variable`."""
    if cls is Variable:
        return True
    if get_origin(cls) is not None and get_origin(cls) is Union:
        if any(t is Variable for t in get_args(cls)):
            return True
    return False


def dataclass_has_variable(cls: type) -> bool:
    """Check if a dataclass has a field with `Variable` in its type signature."""
    if cls is Variable:
        return True
    if is_dataclass(cls):
        if any(may_be_variable(f.type) for f in fields(cls)):
            return True
    return False


def derive_metric(cls: type, constraint: MatchingConstraint) -> Metric:  # dependent type, can't enforce
    """Derive a unified metric from any type.

    Args:
        cls (`Type`): The type to derive the metric from.
        constraint (`MatchingConstraint`): The matching constraint to use.

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
            cls=cls,
            field_metrics={
                fld.name: derive_metric(fld.type, constraint=constraint)  # pyright: ignore
                for fld in fields(cls)
            },
        )

    # derive union metric from unions
    elif cls_origin is Union:
        return UnionMetric(
            cls=cls, case_metrics={case: derive_metric(case, constraint=constraint) for case in get_args(cls)}
        )
    # TODO: graph, dag, tree, sequence

    # derive set matching metric from collections
    elif cls_origin is not None and isinstance(cls_origin, type) and issubclass(cls_origin, Collection):
        elem_type = get_args(cls)[0]
        inner_metric = derive_metric(elem_type, constraint=constraint)
        if dataclass_has_variable(elem_type):
            return LatentSetMatchingMetric(
                cls=elem_type,
                inner=inner_metric,
                constraint=constraint,
            )
        else:
            return SetMatchingMetric(
                inner=inner_metric,
                constraint=constraint,
            )

    # derive discrete metric from equality
    elif getattr(cls, "__eq__", None) is not None:
        return DiscreteMetric(cls=cls)

    else:
        raise ValueError(f"Could not derive metric from type {cls}.")


def metametric(
    cls: Optional[type] = None,
    /,
    normalizer: Union[NormalizerLiteral, Normalizer] = "none",
    constraint: Union[ConstraintLiteral, MatchingConstraint] = "<->",
) -> Callable[[type], type]:
    """Decorate a dataclass to have corresponding metric derived.

    Args:
        cls (`Type`, optional): The class to decorate.
        normalizer (`Union[NormalizerLiteral, Normalizer]`, defaults to "none"):
            The normalizer to use.
        constraint (`ConstraintLiteral`, defaults to "<->"):
            The matching constraint to use.

    Returns:
        `Callable[[Type], Type]`: The class decorator.
    """

    def class_decorator(cls: type) -> type:
        nonlocal normalizer, constraint
        if isinstance(constraint, MatchingConstraint):
            metric = derive_metric(cls, constraint=constraint)
        else:
            metric = derive_metric(cls, constraint=MatchingConstraint.from_str(constraint))
        if isinstance(normalizer, Normalizer):
            normalized_metric = NormalizedMetric(metric, normalizer=normalizer)
        else:
            if normalizer == "none":
                normalized_metric = metric
            else:
                normalizer_obj = Normalizer.from_str(normalizer)
                assert normalizer_obj is not None
                normalized_metric = NormalizedMetric(metric, normalizer=normalizer_obj)
        if dataclass_has_variable(cls):
            setattr(cls, "latent_metric", normalized_metric)  # type: ignore
        else:
            setattr(cls, "metric", normalized_metric)  # type: ignore
        return cls

    if cls is None:  # called with parentheses
        return class_decorator
    else:  # called without parentheses
        return class_decorator(cls)

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
)

from unimetric.alignment import AlignmentConstraint, AlignmentMetric
from unimetric.latent_alignment import dataclass_has_variable, LatentAlignmentMetric
from unimetric.metric import Metric, ProductMetric, DiscreteMetric, FScore, Jaccard, Precision, Recall, UnionMetric

T = TypeVar("T", covariant=True)

NormalizerLiteral = Literal["none", "jaccard", "dice", "f1"]
ConstraintLiteral = Literal["<->", "<-", "->", "~"]


@runtime_checkable
class HasMetric(Protocol):
    """Protocol for classes that have a metric."""

    metric: Metric


@runtime_checkable
class HasLatentMetric(Protocol):
    """Protocol for classes that have a latent metric."""

    latent_metric: Metric


def derive_metric(cls: object, constraint: AlignmentConstraint) -> Metric:
    """Derive a unified metric from any type.

    Parameters
    ----------
    cls : object
        The dataclass-like class to derive the metric from.
    constraint : AlignmentConstraint
        The alignment constraint to use.

    Returns
    -------
    Metric
        The derived metric.
    """
    # if the type is annotated with a metric instance, use the metric annotation
    if get_origin(cls) is Annotated:
        metric = get_args(cls)[1]
        if isinstance(metric, Metric):
            return metric

    # if an explicit metric is defined, use it
    # if getattr(cls, "metric", None) is not None:
    if isinstance(cls, HasMetric):
        return cls.metric

    cls_origin = get_origin(cls)
    # if getattr(cls, "latent_metric", None) is not None:
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


def unimetric(
    normalizer: NormalizerLiteral = "none",
    constraint: ConstraintLiteral = "<->",
) -> Callable[[T], T]:
    """Decorate a dataclass to have corresponding metric derived.

    Parameters
    ----------
    normalizer : NormalizerLiteral
        The normalizer to use, by default "none"
    constraint : ConstraintLiteral
        The alignment constraint to use, by default "<->"

    Returns
    -------
    Callable[[T], T]
        The decorated new class.
    """

    def class_decorator(cls: T) -> T:
        alignment_constraint = {
            "<->": AlignmentConstraint.OneToOne,
            "<-": AlignmentConstraint.OneToMany,
            "->": AlignmentConstraint.ManyToOne,
            "~": AlignmentConstraint.ManyToMany,
            "1:1": AlignmentConstraint.OneToOne,
            "1:*": AlignmentConstraint.OneToMany,
            "*:1": AlignmentConstraint.ManyToOne,
            "*:*": AlignmentConstraint.ManyToMany,
        }[constraint]
        metric = derive_metric(cls, constraint=alignment_constraint)
        normalized_metric = {
            "none": lambda x: x,
            "jaccard": Jaccard,
            "dice": FScore,
            "f1": FScore,
            "precision": Precision,
            "recall": Recall,
        }[normalizer](metric)

        if dataclass_has_variable(cls):
            setattr(cls, "latent_metric", normalized_metric)  # type: ignore
        else:
            setattr(cls, "metric", normalized_metric)  # type: ignore
        return cls

    return class_decorator

from dataclasses import fields, is_dataclass
from typing import Literal, get_args, get_origin, Collection, Annotated, Union

from unimetric.alignment import AlignmentConstraint, AlignmentMetric
from unimetric.latent_alignment import dataclass_has_variable, LatentAlignmentMetric
from unimetric.metric import Metric, ProductMetric, DiscreteMetric, FScore, Jaccard, Precision, Recall, UnionMetric


def derive_metric(cls: type, constraint: AlignmentConstraint) -> Metric:
    """
    Derive a unified metric from any type.
    :param cls: The type to derive a metric from.
    :return: A metric object for the given type.
    """

    # if the type is annotated with a metric instance, use the metric annotation
    if get_origin(cls) is Annotated:
        metric = get_args(cls)[1]
        if isinstance(metric, Metric):
            return metric

    # if an explicit metric is defined, use it
    if getattr(cls, "metric", None) is not None:
        return cls.metric

    if getattr(cls, "latent_metric", None) is not None:
        return cls.latent_metric

    # derive product metric from dataclass
    elif is_dataclass(cls):
        return ProductMetric(
            cls=cls,
            field_metrics={
                fld.name: derive_metric(fld.type, constraint=constraint)
                for fld in fields(cls)
            }
        )

    # derive union metric from unions
    elif get_origin(cls) is Union:
        return UnionMetric(
            cls=cls,
            case_metrics={
                case: derive_metric(case, constraint=constraint)
                for case in get_args(cls)
            }
        )

    # derive alignment metric from collections
    elif get_origin(cls) is not None and isinstance(get_origin(cls), type) and issubclass(get_origin(cls), Collection):
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
        normalizer: Literal['none', 'jaccard', 'dice', 'f1'] = 'none',
        constraint: Literal['<->', '<-', '->', '~'] = '<->',
):
    """
    Derive a unified metric from a class.
    :param normalizer:
    :return:
    """
    def class_decorator(cls):
        alignment_constraint = {
            '<->': AlignmentConstraint.OneToOne,
            '<-': AlignmentConstraint.OneToMany,
            '->': AlignmentConstraint.ManyToOne,
            '~': AlignmentConstraint.ManyToMany,
            '1:1': AlignmentConstraint.OneToOne,
            '1:*': AlignmentConstraint.OneToMany,
            '*:1': AlignmentConstraint.ManyToOne,
            '*:*': AlignmentConstraint.ManyToMany,
        }[constraint]
        metric = derive_metric(cls, constraint=alignment_constraint)
        normalized_metric = {
            'none': lambda x: x,
            'jaccard': Jaccard,
            'dice': FScore,
            'f1': FScore,
            'precision': Precision,
            'recall': Recall,
        }[normalizer](metric)

        if dataclass_has_variable(cls):
            setattr(cls, 'latent_metric', normalized_metric)
        else:
            setattr(cls, 'metric', normalized_metric)
        return cls
    return class_decorator

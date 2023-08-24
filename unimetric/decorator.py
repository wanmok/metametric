from typing import Literal, get_args, get_origin, Collection, Annotated

from unimetric.metric import Metric, ProductMetric, AlignmentMetric, DiscreteMetric, FScore, Jaccard, Precision, \
    Recall


def derive_metric(cls: type) -> Metric:
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

    # derive product metric from dataclass
    elif getattr(cls, "__dataclass_fields__", None) is not None:
        return ProductMetric(
            cls=cls,
            field_metrics={
                fld: derive_metric(cls.__dataclass_fields__[fld].type)
                for fld in cls.__dataclass_fields__.keys()
            }
        )

    # derive alignment metric from collections
    elif get_origin(cls) is not None and issubclass(get_origin(cls), Collection):
        elem_type = get_args(cls)[0]
        return AlignmentMetric(
            inner=derive_metric(elem_type)
        )

    # derive discrete metric from equality
    elif getattr(cls, "__eq__", None) is not None:
        return DiscreteMetric(cls=cls)

    else:
        raise ValueError(f"Could not derive metric from type {cls}.")


def unimetric(
        normalizer: Literal['none', 'jaccard', 'dice', 'f1'] = 'none',
):
    """
    Derive a unified metric from a class.
    :param normalizer:
    :return:
    """
    def class_decorator(cls):
        metric = derive_metric(cls)
        normalized_metric = {
            'none': lambda x: x,
            'jaccard': Jaccard,
            'dice': FScore,
            'f1': FScore,
            'precision': Precision,
            'recall': Recall,
        }[normalizer](metric)
        setattr(cls, 'metric', normalized_metric)
        return cls
    return class_decorator

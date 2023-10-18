"""Abstract Meaning Representation (AMR) structure."""
from dataclasses import dataclass
from typing import Collection, Union

from metametric.core.decorator import metametric
from metametric.core.metric import Variable


@metametric()
@dataclass(eq=True, frozen=True)
class Prop:
    """A Proposition in an AMR."""

    subj: Variable
    pred: str
    obj: Union[Variable, str]


@metametric()
@dataclass
class AMR:
    """Abstract Meaning Representation (AMR) structure."""

    props: Collection[Prop]

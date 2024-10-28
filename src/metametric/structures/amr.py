"""Abstract Meaning Representation (AMR) structure."""

from dataclasses import dataclass
from typing import Union
from collections.abc import Collection

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

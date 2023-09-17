"""Abstract Meaning Representation (AMR) structure."""
from dataclasses import dataclass
from typing import Collection, Union

from unimetric.core.decorator import unimetric
from unimetric.core.latent_alignment import Variable


@unimetric()
@dataclass(eq=True, frozen=True)
class Prop:
    """A Proposition in an AMR."""

    subj: Variable
    pred: str
    obj: Union[Variable, str]


@unimetric(normalizer="f1")
@dataclass
class AMR:
    """Abstract Meaning Representation (AMR) structure."""

    props: Collection[Prop]

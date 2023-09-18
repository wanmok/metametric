"""Abstract Meaning Representation (AMR) structure."""
from dataclasses import dataclass
from typing import Collection, Union

from autometric.core.decorator import autometric
from autometric.core.latent_alignment import Variable


@autometric()
@dataclass(eq=True, frozen=True)
class Prop:
    """A Proposition in an AMR."""

    subj: Variable
    pred: str
    obj: Union[Variable, str]


@autometric(normalizer="f1")
@dataclass
class AMR:
    """Abstract Meaning Representation (AMR) structure."""

    props: Collection[Prop]

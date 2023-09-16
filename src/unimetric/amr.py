"""Abstract Meaning Representation (AMR) structure."""
from dataclasses import dataclass
from typing import Collection, Set, Union

from unimetric.decorator import unimetric
from unimetric.latent_alignment import Variable


@unimetric()
@dataclass
class Prop:
    """A property in an AMR."""

    subj: Variable
    pred: str
    obj: Union[Variable, str]

    def __hash__(self):
        return hash((self.subj, self.pred, self.obj))


@unimetric(normalizer="f1")
@dataclass
class AMR:
    """Abstract Meaning Representation (AMR) structure."""

    props: Collection[Prop]

    def variables(self) -> Set[Variable]:
        """Return the set of variables in the AMR."""
        vars = set()
        for p in self.props:
            if isinstance(p.subj, Variable):
                vars.add(p.subj)
            if isinstance(p.obj, Variable):
                vars.add(p.obj)
        return vars

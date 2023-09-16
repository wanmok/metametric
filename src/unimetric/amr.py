from dataclasses import dataclass
from typing import Collection, Set, Union

from unimetric.decorator import unimetric
from unimetric.latent_alignment import Variable


@unimetric()
@dataclass
class Prop:
    subj: Variable
    pred: str
    obj: Union[Variable, str]

    def __hash__(self):
        return hash((self.subj, self.pred, self.obj))


@unimetric(normalizer="f1")
@dataclass
class AMR:
    props: Collection[Prop]

    def variables(self) -> Set[Variable]:
        vars = set()
        for p in self.props:
            if isinstance(p.subj, Variable):
                vars.add(p.subj)
            if isinstance(p.obj, Variable):
                vars.add(p.obj)
        return vars

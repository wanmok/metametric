from dataclasses import dataclass
from typing import Collection, Set, Union


@dataclass
class Variable:
    name: str

    def __hash__(self):
        return hash(self.name)


@dataclass
class Prop:
    subj: Union[Variable, str]
    pred: str
    obj: Union[Variable, str]

    def __hash__(self):
        return hash((self.subj, self.pred, self.obj))


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


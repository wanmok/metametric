from typing import Tuple, Generic, TypeVar, Collection, Mapping, Iterator

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from unimetric.amr import AMR, Variable
from unimetric.metric import Metric

T = TypeVar('T')


class PairIndexer(Generic[T], Mapping[Tuple[T, T], int]):

    def __init__(self, x: Collection[T], y: Collection[T], offset: int = 0):
        self.x = x
        self.y = y
        self.x2i = {t: i for i, t in enumerate(x)}
        self.y2i = {t: i for i, t in enumerate(y)}
        self.offset = offset

    def __getitem__(self, item: Tuple[T, T]) -> int:
        x, y = item
        return self.offset + self.x2i[x] * len(self.y) + self.y2i[y]

    def __len__(self):
        return len(self.x) * len(self.y)

    def __iter__(self) -> Iterator[Tuple[T, T]]:
        return ((x, y) for x in self.x for y in self.y)


class SMatch(Metric[AMR]):

    def score(self, x: AMR, y: AMR) -> float:

        var_indexer = PairIndexer(x.variables(), y.variables())
        prop_indexer = PairIndexer(x.props, y.props, offset=len(var_indexer))

        # build the constraint matrix
        n = len(var_indexer) + len(prop_indexer)
        constraint_vectors = []

        num_x_var = len(x.variables())
        num_y_var = len(y.variables())
        num_var_constraints = num_x_var + num_y_var
        num_x_prop = len(x.props)
        num_y_prop = len(y.props)
        num_prop_constraints = num_x_prop + num_y_prop
        num_all_constraints = num_var_constraints + num_prop_constraints

        var_constraint_matrix = np.zeros((num_var_constraints, num_all_constraints))
        # each variable must be mapped to exactly one variable
        for x_var in x.variables():
            for y_var in y.variables():
                constraint_vectors[]

        prop_constraint_matrix = np.zeros((num_prop_constraints, num_all_constraints))

        # each property may be mapped to at most one property if the variable is matched

        x_props = list(x.props)
        y_props = list(y.props)
        for k in range(num_x_prop):
            for l in range(num_y_prop):
                index = prop_indexer[x_props[k], y_props[l]]
                if x_props[k].pred == y_props[l].pred:
                    if isinstance(x_props[k].subj, Variable) and isinstance(y_props[k].subj, Variable):
                        prop_constraint_matrix[index, num_var_constraints + num_x_prop * k + l] = 1
                        prop_constraint_matrix[index, ]
                        constraint_vectors.append(row)
                    if isinstance(x_prop.obj, Variable) and isinstance(y_prop.obj, Variable):
                        row = np.zeros(n)
                        row[prop_indexer[x_prop, y_prop]] = 1
                        row[var_indexer[x_prop.obj, y_prop.obj]] = -1
                        constraint_vectors.append(row)

        num_constraints = len(constraint_vectors)
        num_prop_constraints = num_constraints - num_var_constraints
        constraint_vectors = np.array(constraint_vectors)
        c = -np.concatenate([np.zeros(num_var_constraints), np.ones(num_prop_constraints)])

        result = milp(
            c=c,
            constraints=LinearConstraint(
                np.array(constraint_vectors),
                np.zeros(len(constraint_vectors)),
                np.ones(len(constraint_vectors)),
            ),
            bounds=Bounds(0, 1),
            integrality=np.ones(n),
        )

        return -result.fun


from unimetric.amr import AMR, Variable, Prop

amr1 = AMR(
    props=[
        Prop(Variable('a'), 'instance', 'want-01'),
        Prop(Variable('b'), 'instance', 'boy'),
        Prop(Variable('c'), 'instance', 'go-01'),
        Prop(Variable('a'), 'ARG0', Variable('b')),
        Prop(Variable('a'), 'ARG1', Variable('c')),
        Prop(Variable('c'), 'ARG0', Variable('b')),
    ]
)

amr2 = AMR(
    props=[
        Prop(Variable('x'), 'instance', 'want-01'),
        Prop(Variable('y'), 'instance', 'boy'),
        Prop(Variable('z'), 'instance', 'football'),
        Prop(Variable('x'), 'ARG0', Variable('y')),
        Prop(Variable('x'), 'ARG1', Variable('z')),
    ]
)

print(SMatch().score(amr1, amr2))

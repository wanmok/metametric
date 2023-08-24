from typing import Tuple, Generic, TypeVar, Collection, Mapping, Iterator

import numpy as np
import scipy.optimize as spo
from unimetric.amr import AMR, Variable
from unimetric.metric import Metric

T = TypeVar('T')

__all__ = ['SMatchCount']


class _PairIndexer(Generic[T], Mapping[Tuple[T, T], int]):
    """
    Creates a mapping of a Cartesian product of objects to a single index.
    """

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


class SMatchCount(Metric[AMR]):
    """
    Implements the SMatch metric of AMRs.
    This function just counts the number of matches between two AMRs.
    For the common Smatch metric which is F1-normalized, use `Dice(SMatchCount())`.
    References:
        - S Cai, K Knight (2013): Smatch: An Evaluation Metric for Semantic Feature Structures.
        ACL. https://aclanthology.org/P13-2131.pdf.
    """

    def score(self, x: AMR, y: AMR) -> float:

        x_vars = list(x.variables())
        y_vars = list(y.variables())
        n_x = len(x_vars)
        n_y = len(y_vars)
        var_indexer = _PairIndexer(x_vars, y_vars)
        prop_indexer = _PairIndexer(x.props, y.props, offset=n_x * n_y)

        # coefficient vector for the objective function
        coef = np.zeros(len(var_indexer) + len(prop_indexer))

        # build the constraint matrix
        n = len(var_indexer) + len(prop_indexer)

        # each variable must be mapped to exactly one variable
        mask_x = np.zeros([n_x, n_x, n_y])
        mask_x[np.arange(n_x), np.arange(n_x), :] = 1
        mask_x = mask_x.reshape([n_x, n_x * n_y])

        mask_y = np.zeros([n_y, n_x, n_y])
        mask_y[np.arange(n_y), :, np.arange(n_y)] = 1
        mask_y = mask_y.reshape([n_y, n_x * n_y])

        var_constraint_matrix = np.concatenate(
            [
                np.concatenate([mask_x, mask_y], axis=0),
                np.zeros([n_x + n_y, len(prop_indexer)])
            ],
            axis=1
        )
        var_constraints = spo.LinearConstraint(
            A=var_constraint_matrix,
            ub=np.ones(var_constraint_matrix.shape[0]),
        )

        prop_constraint_vectors = []
        # each property may be mapped to at most one property if the variable is matched
        for px in x.props:
            for py in y.props:
                if px.pred == py.pred:
                    v = np.zeros(n)
                    v[prop_indexer[px, py]] = 1
                    v[var_indexer[px.subj, py.subj]] = -1
                    prop_constraint_vectors.append(v)
                    if isinstance(px.obj, Variable) and isinstance(py.obj, Variable):
                        v = np.zeros(n)
                        v[prop_indexer[px, py]] = 1
                        v[var_indexer[px.obj, py.obj]] = -1
                        prop_constraint_vectors.append(v)
                        coef[prop_indexer[px, py]] = 1

                    if isinstance(px.obj, str) and isinstance(py.obj, str) and px.obj == py.obj:
                        coef[prop_indexer[px, py]] = 1

        prop_constraint_matrix = np.stack(prop_constraint_vectors, axis=0)  # [PC, N]
        prop_constraints = spo.LinearConstraint(
            A=prop_constraint_matrix,
            ub=np.zeros(prop_constraint_matrix.shape[0]),
        )

        result = spo.milp(
            c=-coef,
            constraints=[var_constraints, prop_constraints],
            bounds=spo.Bounds(lb=0, ub=1),
            integrality=np.ones_like(coef),
        )

        return -result.fun

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Generic, Optional, TypeVar
from collections.abc import Collection, Iterator, Sequence

import numpy as np
import scipy as sp

from metametric.core.constraint import MatchingConstraint
from metametric.core._problem import MatchingProblem
from metametric.core.metric import Variable

T = TypeVar("T")


@dataclass
class ConstraintBuilder(ABC):
    n_x: int
    n_y: int
    n_x_vars: int
    n_y_vars: int

    def index_pair(self, i: int, j: int) -> int:
        return i * self.n_y + j

    def index_var_pair(self, i: int, j: int) -> int:
        return self.n_x * self.n_y + i * self.n_y_vars + j

    @abstractmethod
    def build(self) -> Optional[sp.optimize.LinearConstraint]:
        pass


@dataclass
class MatchingConstraintBuilder(ConstraintBuilder):
    constraint: MatchingConstraint = MatchingConstraint.ONE_TO_ONE

    def build(self) -> Optional[sp.optimize.LinearConstraint]:
        constraint_matrix_ctor: Callable[[int, int], Optional[np.ndarray]] = {
            MatchingConstraint.ONE_TO_ONE: _get_one_to_one_constraint_matrix,
            MatchingConstraint.ONE_TO_MANY: _get_one_to_many_constraint_matrix,
            MatchingConstraint.MANY_TO_ONE: _get_many_to_one_constraint_matrix,
            MatchingConstraint.MANY_TO_MANY: lambda _0, _1: None,
        }[self.constraint]
        m = constraint_matrix_ctor(self.n_x, self.n_y)
        if m is not None:
            m = np.concatenate(
                [
                    m,
                    np.zeros([m.shape[0], self.n_x_vars * self.n_y_vars]),
                ],  # pad with zeros for the latent variables
                axis=1,
            )
            return sp.optimize.LinearConstraint(
                A=m,
                ub=np.ones([m.shape[0]]),
            )
        else:
            return None  # no constraint when the constraint is MANY_TO_MANY


@dataclass
class VariableMatchingConstraintBuilder(ConstraintBuilder):
    def build(self) -> Optional[sp.optimize.LinearConstraint]:
        if self.n_x_vars == 0 or self.n_y_vars == 0:
            return None
        # only one-to-one matching between variables
        m = _get_one_to_one_constraint_matrix(self.n_x_vars, self.n_y_vars)
        m = np.concatenate(
            [
                np.zeros([m.shape[0], self.n_x * self.n_y]),
                m,
            ],  # pad with zeros for the matching items
            axis=1,
        )
        return sp.optimize.LinearConstraint(
            A=m,
            ub=np.ones([m.shape[0]]),
        )


@dataclass
class MonotonicityConstraintBuilder(ConstraintBuilder):
    gram_matrix: np.ndarray  # R[n_x, n_y]
    x_reachability: np.ndarray  # R[n_x, n_x]
    y_reachability: np.ndarray  # R[n_y, n_y]

    def build(self) -> Optional[sp.optimize.LinearConstraint]:
        vectors = []
        possible_matching_pairs = [
            (u, v) for u in range(self.n_x) for v in range(self.n_y) if self.gram_matrix[u, v] > 0
        ]
        for u0, v0 in possible_matching_pairs:
            for u1, v1 in possible_matching_pairs:
                if self.x_reachability[u0, u1] != self.y_reachability[v0, v1]:
                    vec = np.zeros(self.n_x * self.n_y)
                    vec[self.index_pair(u0, v0)] = 1
                    vec[self.index_pair(u1, v1)] = 1
                    vectors.append(vec)
                    # Enforce monotonicity of the matching
                    #    [u0 ~ v0] & [u1 ~ v1] -> [u0 <= u1] ≡ [v0 <= v1]
                    # => 1 - ((1 - t[u0~v0]) + (1 - t[u1~v1])) <= 1[[u0 <= u1] ≡ [v0 <= v1]]
                    # => t[u0~v0] + t[u1~v1] <= 1[[u0 <= u1] ≡ [v0 <= v1]] + 1
                    # => t[u0~v0] + t[u1~v1] <= 1  (if [u0 <= u1] ≡ [v0 <= v1], constraint redundant)

        if len(vectors) == 0:
            return None
        constraint_matrix = np.stack(vectors, axis=0)  # R[n_constraints, n_x * n_y]
        return sp.optimize.LinearConstraint(
            A=constraint_matrix,
            ub=np.ones([constraint_matrix.shape[0]]),
        )


@dataclass
class LatentVariableConstraintBuilder(ConstraintBuilder, Generic[T]):
    x: Collection[T]
    y: Collection[T]
    cls: type[T]
    gram_matrix: np.ndarray  # R[n_x, n_y]

    def __post_init__(self):
        assert is_dataclass(self.cls)

    def build(self) -> Optional[sp.optimize.LinearConstraint]:
        x_vars = list(_all_variables(self.x))
        y_vars = list(_all_variables(self.y))
        x_var_to_id = {t: i for i, t in enumerate(x_vars)}
        y_var_to_id = {t: j for j, t in enumerate(y_vars)}
        vectors = []
        for i, a in enumerate(self.x):
            for j, b in enumerate(self.y):
                if self.gram_matrix[i, j] > 0:
                    for fld in fields(self.cls):  # pyright: ignore
                        a_fld = getattr(a, fld.name, None)
                        b_fld = getattr(b, fld.name, None)
                        if isinstance(a_fld, Variable) and isinstance(b_fld, Variable):
                            vec = np.zeros(self.n_x * self.n_y + self.n_x_vars * self.n_y_vars)
                            vec[self.index_pair(i, j)] = 1
                            vec[self.index_var_pair(x_var_to_id[a_fld], y_var_to_id[b_fld])] = -1
                            vectors.append(vec)
                            # Item matches implies variable matches
                            #    [a ~ b] -> [a_fld ~ b_fld]
                            # => t[a~b] <= t[a_fld~b_fld]
                            # => t[a~b] - t[a_fld~b_fld] <= 0
        if len(vectors) == 0:
            return None
        constraint_matrix = np.stack(vectors, axis=0)  # R[n_constraints, n_x * n_y + n_x_vars * n_y_vars]
        return sp.optimize.LinearConstraint(
            A=constraint_matrix,
            ub=np.zeros([constraint_matrix.shape[0]]),
        )


class ILPMatchingProblem(MatchingProblem[T]):
    """Creates a matching problem that is solved by ILP.

    The constrained ILP problem has variables
    - for each pair of elements in X and Y, and
    - for each pair of potential latent variables in X and Y.
    """

    def __init__(
        self,
        x: Sequence[T],
        y: Sequence[T],
        gram_matrix: np.ndarray,
        has_vars: bool = False,
    ):
        super().__init__(x, y, gram_matrix)
        self.n_x = len(x)
        self.n_y = len(y)
        if has_vars:
            self.x_vars = list(_all_variables(x))
            self.y_vars = list(_all_variables(y))
            self.n_x_vars = len(self.x_vars)
            self.n_y_vars = len(self.y_vars)
        else:
            self.x_vars = []
            self.y_vars = []
            self.n_x_vars = 0
            self.n_y_vars = 0
        self.constraints: list[sp.optimize.LinearConstraint] = []

    def add_matching_constraint(self, constraint_type: MatchingConstraint):
        constraint = MatchingConstraintBuilder(
            n_x=self.n_x,
            n_y=self.n_y,
            n_x_vars=self.n_x_vars,
            n_y_vars=self.n_y_vars,
            constraint=constraint_type,
        ).build()
        if constraint is not None:
            self.constraints.append(constraint)

    def add_variable_matching_constraint(self):
        constraint = VariableMatchingConstraintBuilder(
            n_x=self.n_x,
            n_y=self.n_y,
            n_x_vars=self.n_x_vars,
            n_y_vars=self.n_y_vars,
        ).build()
        if constraint is not None:
            self.constraints.append(constraint)

    def add_monotonicity_constraint(self, x_reachability: np.ndarray, y_reachability: np.ndarray):
        constraint = MonotonicityConstraintBuilder(
            n_x=self.n_x,
            n_y=self.n_y,
            n_x_vars=self.n_x_vars,
            n_y_vars=self.n_y_vars,
            gram_matrix=self.gram_matrix,
            x_reachability=x_reachability,
            y_reachability=y_reachability,
        ).build()
        if constraint is not None:
            self.constraints.append(constraint)

    def add_latent_variable_constraint(self, cls: type[T]):
        constraint = LatentVariableConstraintBuilder(
            n_x=self.n_x,
            n_y=self.n_y,
            n_x_vars=self.n_x_vars,
            n_y_vars=self.n_y_vars,
            x=self.x,
            y=self.y,
            cls=cls,
            gram_matrix=self.gram_matrix,
        ).build()
        if constraint is not None:
            self.constraints.append(constraint)

    def solve(self):
        # Layout of the constraint matrix:
        # ╒═════════════════════════╤═══════════════════════════╕
        # │        n_x * n_y        │    n_x_vars * n_y_vars    │
        # ┝━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
        # │   MATCHING_CONSTRAINT   │                           │
        # ├─────────────────────────┼───────────────────────────┤
        # │                         │  VAR_MATCHING_CONSTRAINT  │
        # ├─────────────────────────┼───────────────────────────┤
        # │ MONOTONICITY_CONSTRAINT │                           │
        # ├─────────────────────────┴───────────────────────────┤
        # │            LATENT_VARIABLE_CONSTRAINT               │
        # ╘═════════════════════════════════════════════════════┙
        coef = np.concatenate(
            [
                self.gram_matrix.reshape([self.n_x * self.n_y]),
                np.zeros([self.n_x_vars * self.n_y_vars]),
            ],  # pad with zeros for the latent variables
            axis=0,
        )
        result = sp.optimize.milp(
            c=-coef,
            constraints=self.constraints,
            bounds=sp.optimize.Bounds(lb=0, ub=1),
            integrality=np.ones_like(coef),
        )
        solution = result.x[: self.n_x * self.n_y].reshape([self.n_x, self.n_y])
        matching = [
            (i, j, self.gram_matrix[i, j].item())
            for i in range(self.n_x)
            for j in range(self.n_y)
            if solution[i, j] > 0
        ]
        return -result.fun, matching


def _get_one_to_many_constraint_matrix(n_x: int, n_y: int) -> np.ndarray:  # [Y, X * Y]
    mask_y = np.zeros([n_y, n_x, n_y])
    mask_y[np.arange(n_y), :, np.arange(n_y)] = 1
    mask_y = mask_y.reshape([n_y, n_x * n_y])
    return mask_y


def _get_many_to_one_constraint_matrix(n_x: int, n_y: int) -> np.ndarray:  # [X, X * Y]
    mask_x = np.zeros([n_x, n_x, n_y])
    mask_x[np.arange(n_x), np.arange(n_x), :] = 1
    mask_x = mask_x.reshape([n_x, n_x * n_y])
    return mask_x


def _get_one_to_one_constraint_matrix(n_x: int, n_y: int) -> np.ndarray:  # [X + Y, X * Y]
    mask_x = _get_one_to_many_constraint_matrix(n_x, n_y)
    mask_y = _get_many_to_one_constraint_matrix(n_x, n_y)
    return np.concatenate([mask_x, mask_y], axis=0)


def _all_variables(obj: Any) -> Iterator[Variable]:
    if isinstance(obj, Variable):
        yield obj
    elif isinstance(obj, Collection) and not isinstance(obj, str):
        for item in obj:
            yield from _all_variables(item)
    elif getattr(obj, "__dict__", None) is not None:
        for fld in vars(obj).values():
            yield from _all_variables(fld)

"""Metric derivation with latent alignments."""
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    ClassVar,
    Generic,
    Collection,
    Mapping,
    Tuple,
    TypeVar,
    Union,
    Set,
    Iterator,
    get_origin,
    get_args,
    Any, Type,
)

import numpy as np
import scipy.optimize as spo

from autometric.core.alignment import AlignmentConstraint, solve_alignment
from autometric.core.metric import Metric

T = TypeVar("T")


@dataclass(eq=True, frozen=True)
class Variable:
    """A variable in latent alignments."""

    name: str

    latent_metric: ClassVar[Metric["Variable"]] = Metric.from_function(lambda x, y: 1.0)


class LatentAlignmentMetric(Metric[Collection[T]]):
    """A metric derived to support aligning latent variables defined in structures."""

    def __init__(self, cls: Type[T], inner: Metric[T], constraint: AlignmentConstraint = AlignmentConstraint.ONE_TO_ONE):
        if is_dataclass(cls):
            self.fields = fields(cls)
        else:
            raise ValueError(f"{cls} has to be a dataclass.")
        self.inner = inner
        self.constraint = constraint

    def score(self, x: Collection[T], y: Collection[T]) -> float:
        """Score two collections of objects."""
        x = list(x)
        y = list(y)
        x_vars = _all_variables(x)
        y_vars = _all_variables(y)
        n_x_vars = len(x_vars)
        n_y_vars = len(y_vars)
        n_x = len(x)
        n_y = len(y)
        n_pairs = n_x * n_y
        var_indexer = _PairIndexer(x_vars, y_vars)

        gram_matrix = self.inner.gram_matrix(x, y)

        # coefficient vector for the objective function
        coef = np.concatenate(
            [np.zeros(n_x_vars * n_y_vars), gram_matrix.reshape([n_pairs])], axis=0  # score that x_i matches x_j
        )

        # build the constraint matrix
        n = n_x_vars * n_y_vars + n_pairs

        # each variable must be mapped to exactly one variable
        var_constraint_matrix = _get_one_to_one_constraint_matrix(n_x_vars, n_y_vars)
        var_constraint_matrix = np.concatenate(
            [var_constraint_matrix, np.zeros([var_constraint_matrix.shape[0], n_pairs])], axis=1
        )
        var_constraints = spo.LinearConstraint(
            A=var_constraint_matrix,
            ub=np.ones(var_constraint_matrix.shape[0]),  # type: ignore
        )

        # each item may be mapped to some other items given the constraint
        item_constraint_matrix_ctor = {
            AlignmentConstraint.ONE_TO_ONE: _get_one_to_one_constraint_matrix,
            AlignmentConstraint.ONE_TO_MANY: _get_one_to_many_constraint_matrix,
            AlignmentConstraint.MANY_TO_ONE: _get_many_to_one_constraint_matrix,
            AlignmentConstraint.MANY_TO_MANY: lambda _0, _1: None,
        }[self.constraint]
        item_constraint_matrix = item_constraint_matrix_ctor(n_x, n_y)
        if item_constraint_matrix is not None:
            item_constraint_matrix = np.concatenate(
                [
                    np.zeros([item_constraint_matrix.shape[0], n_x_vars * n_y_vars]),
                    item_constraint_matrix,
                ],
                axis=1,
            )
        item_constraints = (
            None
            if item_constraint_matrix is None
            else spo.LinearConstraint(
                A=item_constraint_matrix,
                ub=np.ones(item_constraint_matrix.shape[0]),  # type: ignore
            )
        )

        # constrain the item and their variables
        item_var_constraint_vectors = []
        for i, a in enumerate(x):
            for j, b in enumerate(y):
                m = gram_matrix[i, j]
                if m > 0:  # only consider items that can possibly match
                    for fld in self.fields:
                        a_fld = getattr(a, fld.name, None)
                        b_fld = getattr(b, fld.name, None)
                        if isinstance(a_fld, Variable) and isinstance(b_fld, Variable):
                            v = np.zeros(n)
                            v[n_x_vars * n_y_vars + i * len(y) + j] = 1
                            v[var_indexer[a_fld, b_fld]] = -1
                            item_var_constraint_vectors.append(v)

        item_var_constraint_matrix = np.stack(item_var_constraint_vectors, axis=0)  # [PC, N]
        item_var_constraints = spo.LinearConstraint(
            A=item_var_constraint_matrix,
            ub=np.zeros(item_var_constraint_matrix.shape[0]),  # type: ignore
        )

        constraints = [var_constraints, item_var_constraints]
        if item_constraints is not None:
            constraints.append(item_constraints)

        result = spo.milp(
            c=-coef,
            constraints=constraints,
            bounds=spo.Bounds(lb=0, ub=1),
            integrality=np.ones_like(coef),
        )
        return -result.fun

    def score_self(self, x: Collection[T]) -> float:
        """Score a collection of objects with itself."""
        return solve_alignment(self.inner.gram_matrix(x, x), self.constraint)


class _PairIndexer(Generic[T], Mapping[Tuple[T, T], int]):
    """Creates a mapping of a Cartesian product of objects to a single index."""

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


def _all_variables(x: Any) -> Set[Variable]:
    def _all_variables_iterator(obj: Any) -> Iterator[Variable]:
        if isinstance(obj, Variable):
            yield obj
        elif isinstance(obj, Collection) and not isinstance(obj, str):
            for item in obj:
                yield from _all_variables(item)
        elif getattr(obj, "__dict__", None) is not None:
            for fld in vars(x).values():
                yield from _all_variables(fld)

    return set(_all_variables_iterator(x))


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


def may_be_variable(cls: Any) -> bool:
    """Check if a type may be a `Variable`."""
    if cls is Variable:
        return True
    if get_origin(cls) is not None and get_origin(cls) is Union:
        if any(t is Variable for t in get_args(cls)):
            return True
    return False


def dataclass_has_variable(cls: Type) -> bool:
    """Check if a dataclass has a field is in `Variable` type."""
    if cls is Variable:
        return True
    if is_dataclass(cls):
        if any(may_be_variable(t.type) for t in cls.__dataclass_fields__.values()):
            return True
    return False

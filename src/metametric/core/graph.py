"""Define a `Graph` protocol that `networkx.DiGraph` satisfies."""

from typing import Protocol, TypeVar, runtime_checkable
from collections.abc import Collection, Iterator

import numpy as np

T = TypeVar("T")


@runtime_checkable
class Graph(Protocol[T]):
    """A minimum graph definition of which `networkx.DiGraph` is an instance."""

    def nodes(self) -> Collection[T]:
        raise NotImplementedError()

    def successors(self, x: T) -> Iterator[T]:
        raise NotImplementedError()

    def predecessors(self, x: T) -> Iterator[T]:
        raise NotImplementedError()


def _adjacency_matrix(graph: Graph) -> np.ndarray:
    """Get the adjacency matrix of a graph."""
    nodes = list(graph.nodes())
    node_to_id = {x: i for i, x in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros([n, n], dtype=bool)
    for i, x in enumerate(nodes):
        for y in graph.successors(x):
            j = node_to_id[y]
            adj[i, j] = 1
    return adj


def _reachability_matrix(graph: Graph) -> np.ndarray:
    """Get the reachability matrix of a graph."""
    a = _adjacency_matrix(graph)
    b = np.eye(a.shape[0], dtype=bool) + a
    c = b @ b
    while not np.all(c == b):
        b = c
        c = b @ b
    return c

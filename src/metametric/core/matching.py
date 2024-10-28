"""Defines utilities for obtaining the inner matching after a metric is computed."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Any
from collections.abc import Iterable
from dataclasses import dataclass

from metametric.core.path import Path

T = TypeVar("T", covariant=True)
Tc = TypeVar("Tc", contravariant=True)


@dataclass
class Match(Generic[T]):
    """Represents a match between a pair of inner objects in a prediction and a reference."""

    pred_path: Path
    pred: T
    ref_path: Path
    ref: T
    score: float

    def __str__(self):
        """Returns a string representation of the match."""
        return f"{self.pred_path} -> {self.ref_path} ({self.score})"


class Hook(ABC, Generic[Tc]):
    """A hook that is called when a match is found."""

    @abstractmethod
    def on_match(self, data_id: int, pred_path: str, pred: Tc, ref_path: str, ref: Tc, score: float):
        """Called when a match is found."""
        raise NotImplementedError

    @staticmethod
    def from_callable(func: Callable[[int, str, Tc, str, Tc, float], None]) -> "Hook[Tc]":
        """Creates a hook from a callback."""
        return _HookFromCallable(func)


class _HookFromCallable(Hook[Tc]):
    def __init__(self, func: Callable[[int, str, Tc, str, Tc, float], None]):
        self.func = func

    def on_match(self, data_id: int, pred_path: str, pred: Tc, ref_path: str, ref: Tc, score: float):
        self.func(data_id, pred_path, pred, ref_path, ref, score)


class Matching(Iterable[Match[object]]):
    """An object that can be used to iterate over matches and run hooks on them."""

    def __init__(self, matches: Iterable[Match[object]]):
        self.matches = matches

    def __iter__(self):
        """Traverses all matching pairs of inner objects."""
        return iter(self.matches)

    def run_with_hooks(self, hooks: dict[str, Hook[Any]], data_id: int = 0):
        """Runs hooks on the matches."""
        parsed_hooks = {Path.parse(selector): hook for selector, hook in hooks.items()}
        for match in self.matches:
            for selector, hook in parsed_hooks.items():
                if selector.selects(match.pred_path):
                    hook.on_match(
                        data_id, str(match.pred_path), match.pred, str(match.ref_path), match.ref, match.score
                    )

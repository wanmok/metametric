"""Defines utilities for obtaining the inner matching after a metric is computed."""
from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar, Iterable, Union, Callable, Dict, Any
from dataclasses import dataclass


T = TypeVar("T", covariant=True)


def _index_to_int_str(i: int) -> str:
    return str(i) if i != -1 else "*"


def _int_str_to_index(s: str) -> int:
    return int(s) if s != "*" else -1


def _component_covers(selector: Union[str, int], component: Union[str, int]) -> bool:
    if isinstance(selector, int) and isinstance(component, int):
        return selector == component or selector == -1
    elif isinstance(selector, str) and isinstance(component, str):
        return selector == component or selector == "*"
    else:
        return False


@dataclass
class Path:
    """Represents a path selector. Its default string representation is in JMESPath format."""
    components: Tuple[Union[str, int], ...] = ()  # [-1] means [*]

    def is_root(self) -> bool:
        """Returns True if the path is the root path."""
        return len(self.components) == 0

    def __hash__(self):
        """Returns a hash of the path."""
        return hash(self.components)

    def __str__(self):
        """Returns a string representation of the path in JMESPath format."""
        if len(self.components) == 0:
            return "@"
        else:
            components = [
                f"[{_index_to_int_str(item)}]" if isinstance(item, int) else f".{item}" if i != 0 else item
                for i, item in enumerate(self.components)
            ]
            return "".join(components)

    def prepend(self, item: Union[str, int]) -> 'Path':
        """Prepends an item to the path."""
        return Path((item,) + self.components)

    def selects(self, other: 'Path') -> bool:
        """Returns True if the other path is selected by this path selector."""
        return len(self.components) == len(other.components) and all(
            _component_covers(sel, comp) for sel, comp in zip(self.components, other.components)
        )

    @classmethod
    def parse(cls, s: str):
        """Parses a string representation of the path in JMESPath format."""
        tokens = []
        t = ""
        for c in s:
            if c in ["@", ".", "[", "]"]:
                if t:
                    tokens.append(t)
                    t = ""
                tokens.append(c)
            else:
                t += c
        if t:
            tokens.append(t)
        components = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "@":
                i += 1
                continue
            if tokens[i] == ".":
                i += 1
                continue
            elif tokens[i] == "[":
                components.append(_int_str_to_index(tokens[i + 1]))
                i += 3
            else:
                components.append(tokens[i])
                i += 1
        return Path(tuple(components))


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


class Hook(ABC, Generic[T]):
    """A hook that is called when a match is found."""

    @abstractmethod
    def on_match(self, data_id: int, pred_path: str, pred: T, ref_path: str, ref: T, score: float):
        """Called when a match is found."""
        raise NotImplementedError

    @staticmethod
    def from_callable(func: Callable[[int, str, T, str, T, float], None]) -> 'Hook[T]':
        """Creates a hook from a callback."""
        return _HookFromCallable(func)


class _HookFromCallable(Hook[T]):

    def __init__(self, func: Callable[[int, str, T, str, T, float], None]):
        self.func = func

    def on_match(self, data_id: int, pred_path: str, pred: T, ref_path: str, ref: T, score: float):
        self.func(data_id, pred_path, pred, ref_path, ref, score)


class Matching(Iterable[Match[object]]):
    """An object that can be used to iterate over matches and run hooks on them."""

    def __init__(self, matches: Iterable[Match[object]]):
        self.matches = matches

    def __iter__(self):
        """Traverses all matching pairs of inner objects."""
        return iter(self.matches)

    def run_with_hooks(self, hooks: Dict[str, Hook[Any]], data_id: int = 0):
        """Runs hooks on the matches."""
        hooks = {Path.parse(selector): hook for selector, hook in hooks.items()}
        for match in self.matches:
            for selector, hook in hooks.items():
                if selector.selects(match.pred_path):
                    hook.on_match(data_id, str(match.pred_path), match.pred, str(match.ref_path), match.ref, match.score)

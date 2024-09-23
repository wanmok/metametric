from typing import Generic, Tuple, TypeVar, Iterable, Union
from dataclasses import dataclass


T = TypeVar("T", covariant=True)


@dataclass
class Path:
    components: Tuple[Union[str, int], ...] = ()

    def is_root(self) -> bool:
        return len(self.components) == 0

    def __str__(self):  # returns a string representation of the path in JMESPath format
        if len(self.components) == 0:
            return "@"
        else:
            components = [
                f"[{item}]" if isinstance(item, int) else f".{item}" if i != 0 else item
                for i, item in enumerate(self.components)
            ]
            return "".join(components)

    def prepend(self, item: Union[str, int]) -> 'Path':
        return Path((item,) + self.components)


@dataclass
class Match(Generic[T]):
    pred_path: Path
    pred: T
    ref_path: Path
    ref: T
    score: float

    def __str__(self):
        return f"{self.pred_path} -> {self.ref_path} ({self.score})"


class Matching(Iterable[Match[object]]):

    def __init__(self, matches: Iterable[Match[object]]):
        self.matches = matches

    def __iter__(self):
        return iter(self.matches)

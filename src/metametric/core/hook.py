from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar, List, Iterable, Callable

T = TypeVar("T")


class _Path(Iterable[str]):
    def __init__(self, components: List[str]):
        self.components = components

    def to_jmespath(self) -> str:
        if len(self.components) == 0:
            return "@"
        return ".".join(self.components)

    def __str__(self):
        return self.to_jmespath()

    def __hash__(self):
        return hash(tuple(self.components))

    def __len__(self):
        return len(self.components)

    def __getitem__(self, i):
        return self.components[i]

    def __iter__(self):
        return iter(self.components)

    def startswith(self, prefix: "_Path") -> bool:
        return len(self) >= len(prefix) and self.components[:len(prefix.components)] == prefix.components

    @classmethod
    def from_str(cls, s: str):
        if s == "@":
            return cls([])
        return cls(s.split("."))


class Hook(ABC, Generic[T]):

    def __init__(self, callback: Callable[[T, T, str, float], None]):
        self.callback = callback

    def on_match(self, pred: T, ref: T, path: str, score: float):
        self.callback(pred, ref, path, score)


class Hooks:
    def __init__(self, hooks: Dict[_Path, Hook],  prefix: _Path):
        self.prefix = prefix
        self.hooks = hooks

    @classmethod
    def from_dict(cls, hooks: Dict[str, Hook]):
        return cls({_Path.from_str(k): v for k, v in hooks.items()}, _Path([]))

    def advance(self, name: str):
        new_prefix = _Path(self.prefix.components + [name])
        subhooks = {
            _Path(k.components[1:]): v
            for k, v in self.hooks.items()
            if k.startswith(_Path([name]))
        }
        return Hooks(subhooks, new_prefix)

    def on_match(self, pred: T, ref: T, score: float):
        for path, hook in self.hooks.items():
            if len(path) == 0:
                hook.on_match(pred, ref, self.prefix.to_jmespath(), score)

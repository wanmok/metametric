"""`Path` object that has a string representation in JMESPath format.

This module provides a `Path` object that represents a path and corresponding methods to build and select paths.
The `Path` object has a string representation in JMESPath format. However, it does not support all JMESPath features.
"""

from dataclasses import dataclass
from typing import Union, SupportsInt, overload


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

    components: tuple[Union[str, int], ...] = ()  # [-1] means [*]

    @overload
    def __getitem__(self, item: SupportsInt) -> Union[str, int]: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[Union[str, int], ...]: ...

    def __getitem__(self, item):
        """Returns a component of the path. This does not consider the root path."""
        if isinstance(item, slice):
            return self.components[item]
        elif isinstance(item, SupportsInt):
            return self.components[int(item)]
        else:
            raise NotImplementedError(f"Unsupported item type: {type(item)}")

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

    def prepend(self, item: Union[str, int]) -> "Path":
        """Prepends an item to the path."""
        return Path((item,) + self.components)

    def append(self, item: Union[str, int]) -> "Path":
        """Appends an item to the path."""
        return Path(self.components + (item,))

    def selects(self, other: "Path") -> bool:
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

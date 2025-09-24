"""Matching constraints for the matching metric."""

from enum import Enum, auto


class MatchingConstraint(Enum):
    """Matching constraints for the matching metric."""

    ONE_TO_ONE = auto()
    ONE_TO_MANY = auto()
    MANY_TO_ONE = auto()
    MANY_TO_MANY = auto()

    @staticmethod
    def from_str(s: str) -> "MatchingConstraint":
        return {
            "<->": MatchingConstraint.ONE_TO_ONE,
            "<-": MatchingConstraint.ONE_TO_MANY,
            "->": MatchingConstraint.MANY_TO_ONE,
            "~": MatchingConstraint.MANY_TO_MANY,
            "1:1": MatchingConstraint.ONE_TO_ONE,
            "1:*": MatchingConstraint.ONE_TO_MANY,
            "*:1": MatchingConstraint.MANY_TO_ONE,
            "*:*": MatchingConstraint.MANY_TO_MANY,
        }[s]

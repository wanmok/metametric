"""Alignment constraints for the alignment metric."""
from enum import Enum, auto


class AlignmentConstraint(Enum):
    """Alignment constraints for the alignment metric."""

    ONE_TO_ONE = auto()
    ONE_TO_MANY = auto()
    MANY_TO_ONE = auto()
    MANY_TO_MANY = auto()

    @staticmethod
    def from_str(s: str) -> "AlignmentConstraint":
        return {
            "<->": AlignmentConstraint.ONE_TO_ONE,
            "<-": AlignmentConstraint.ONE_TO_MANY,
            "->": AlignmentConstraint.MANY_TO_ONE,
            "~": AlignmentConstraint.MANY_TO_MANY,
            "1:1": AlignmentConstraint.ONE_TO_ONE,
            "1:*": AlignmentConstraint.ONE_TO_MANY,
            "*:1": AlignmentConstraint.MANY_TO_ONE,
            "*:*": AlignmentConstraint.MANY_TO_MANY,
        }[s]

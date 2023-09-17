"""Data structures commonly used in information extraction.

The data structures defined here can automatically derive commonly used metrics in IE.
"""
from dataclasses import dataclass
from typing import List

from unimetric.core.decorator import unimetric


@unimetric()
@dataclass
class Mention:
    """A mention span commonly used for .

    The mention span here does not enforce whether it is right exclusive or not.
    However, it is a convention that the mention span is right exclusive.

    Attributes
    ----------
        left: The left index of the span.
        right: The right index of the span.
    """

    left: int
    right: int


@unimetric()
@dataclass
class Relation:
    """A relation between two mentions commonly used in relation extraction.

    Attributes
    ----------
        type: The type of the relation.
        subj: The subject mention.
        obj: The object mention.

    """

    type: str
    subj: Mention
    obj: Mention


@unimetric()
class Trigger:
    """A trigger mention commonly used in event extraction.

    Attributes
    ----------
        type: The type of the trigger, commonly used to indicate event type.
    """

    mention: Mention
    type: str


@unimetric()
class Argument:
    """An argument mention commonly used in event extraction.

    Attributes
    ----------
        mention: The mention of the argument.
        role: The role of the argument.
    """

    mention: Mention
    role: str


@unimetric()
@dataclass
class Event:
    """An event commonly used in event extraction.

    Attributes
    ----------
        trigger: The trigger of the event.
        args: The arguments of the event.
    """

    trigger: Trigger
    args: List[Argument]


@unimetric()
@dataclass
class EventSet:
    """A set of events to present predicted or referenced events."""

    events: List[Event]


@unimetric()
@dataclass
class RelationSet:
    """A set of relations to present predicted or referenced relations."""

    relations: List[Relation]


@unimetric(normalizer="f1")
@dataclass
class Entity:
    """An entity comprises multiple mentions, commonly used in coreference resolution."""

    mentions: List[Mention]


@unimetric(normalizer="f1")
@dataclass
class EntitySet:
    """A set of entities to present predicted or referenced entities."""

    entities: List[Entity]

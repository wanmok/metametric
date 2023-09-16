from dataclasses import dataclass
from typing import List

from unimetric.decorator import unimetric


@unimetric()
@dataclass
class Mention:
    left: int
    right: int


@unimetric()
@dataclass
class Relation:
    type: str
    subj: Mention
    obj: Mention


@unimetric()
class Trigger:
    mention: Mention
    type: str


@unimetric()
class Argument:
    mention: Mention
    role: str


@unimetric()
@dataclass
class Event:
    trig: Trigger
    args: List[Argument]


@unimetric()
@dataclass
class EventSet:
    events: List[Event]


@unimetric()
@dataclass
class RelationSet:
    relations: List[Relation]


@unimetric(normalizer='f1')
@dataclass
class Entity:
    mentions: List[Mention]


@unimetric(normalizer='f1')
@dataclass
class EntitySet:
    entities: List[Entity]

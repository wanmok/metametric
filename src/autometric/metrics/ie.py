"""Data structures commonly used in information extraction.

The data structures defined here can automatically derive commonly used metrics in IE.
"""
from dataclasses import dataclass
from typing import List

from autometric.core.decorator import autometric
from autometric.core.metric import Metric
import autometric.core.dsl as am


@autometric()
@dataclass(eq=True, frozen=True)
class Mention:
    """A mention span commonly used for .

    The mention span here does not enforce whether it is right exclusive or not.
    However, it is a convention that the mention span is right exclusive.

    Attributes:
    ----------
        left: The left index of the span.
        right: The right index of the span.
    """

    left: int
    right: int


@autometric()
@dataclass
class Relation:
    """A relation between two mentions commonly used in relation extraction.

    Attributes:
    ----------
        type: The type of the relation.
        subj: The subject mention.
        obj: The object mention.

    """

    type: str
    subj: Mention
    obj: Mention


@autometric()
class Trigger:
    """A trigger mention commonly used in event extraction.

    Attributes:
    ----------
        type: The type of the trigger, commonly used to indicate event type.
    """

    mention: Mention
    type: str


@autometric()
class Argument:
    """An argument mention commonly used in event extraction.

    Attributes:
    ----------
        mention: The mention of the argument.
        role: The role of the argument.
    """

    mention: Mention
    role: str


@autometric()
@dataclass
class Event:
    """An event commonly used in event extraction.

    Attributes:
    ----------
        trigger: The trigger of the event.
        args: The arguments of the event.
    """

    trigger: Trigger
    args: List[Argument]


@autometric()
@dataclass
class EventSet:
    """A set of events to present predicted or referenced events."""

    events: List[Event]


@autometric()
@dataclass
class RelationSet:
    """A set of relations to present predicted or referenced relations."""

    relations: List[Relation]


@dataclass
class Entity:
    """An entity comprises multiple mentions, commonly used in coreference resolution."""

    mentions: List[Mention]


@autometric()
@dataclass
class EntitySet:
    """A set of entities to present predicted or referenced entities."""

    entities: List[Entity]


@autometric()
@dataclass
class Membership:
    """A membership relation between an entity and a mention."""
    mention: Mention
    entity: Entity


muc_link: Metric[Entity] = am.from_func(lambda x, y: max(0, len(set(x.mentions) & set(y.mentions)) - 1))

muc = am.dataclass[EntitySet]({
    "entities": am.alignment[Entity, "~"](muc_link)
})

muc_family = am.family(muc, am.macro_average(["precision", "recall", "f1"]))


def entity_set_to_membership_set(es: EntitySet) -> List[Membership]:
    return [Membership(mention=m, entity=e) for e in es.entities for m in e.mentions]


b_cubed_precision = am.preprocess(
    entity_set_to_membership_set,
    am.alignment[Membership, "<->", "precision"](
        am.dataclass[Membership]({
            "mention": ...,
            "entity": am.normalize["precision"](am.auto[Entity])
        })
    )
)

b_cubed_recall = am.preprocess(
    entity_set_to_membership_set,
    am.alignment[Membership, "<->", "recall"](
        am.dataclass[Membership]({
            "mention": ...,
            "entity": am.normalize["recall"](am.auto[Entity])
        })
    )
)

b_cubed_family = am.multiple_families({
    "precision": am.family(b_cubed_precision, am.macro_average(["none"])),
    "recall": am.family(b_cubed_recall, am.macro_average(["none"])),
}).with_extra(lambda m: {
    "f1": (
        2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
        if (m["precision"] + m["recall"]) > 0 else 0.0
    )
})


ceaf_phi4 = am.dataclass[EntitySet]({
    "entities": am.alignment[Entity, "<->"](
        am.dataclass[Entity]({
            "mentions": am.alignment[Mention, "<->", "f1"](...)
        })
    )
})


ceaf_phi4_family = am.family(ceaf_phi4, am.macro_average(["precision", "recall", "f1"]))


coref_family = am.multiple_families({
    "muc": muc_family,
    "b_cubed": b_cubed_family,
    "ceaf_phi4": ceaf_phi4_family,
}).with_extra(lambda m: {
    "avg-f1": (m["muc-f1"] + m["b_cubed-f1"] + m["ceaf_phi4-f1"]) / 3
})

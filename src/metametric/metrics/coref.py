"""Metric suite definitions for coreference resolution."""

from collections.abc import Collection

import metametric.dsl as mm
from metametric.core.metric import Metric
from metametric.structures.ie import Entity, EntitySet, Membership, Mention


def _muc_common_links(x: Entity, y: Entity) -> int:
    return max(0, len(set(x.mentions) & set(y.mentions)) - 1)


muc_link: Metric[Entity] = mm.from_func(_muc_common_links)

muc = mm.dataclass[EntitySet]({"entities": mm.set_matching[Entity, "~"](muc_link)})


def _entity_set_to_membership_set(es: EntitySet) -> Collection[Membership]:
    return [Membership(mention=m, entity=e) for e in es.entities for m in e.mentions]


b_cubed_precision = mm.preprocess(
    _entity_set_to_membership_set,
    mm.set_matching[Membership, "<->", "precision"](
        mm.dataclass[Membership]({"mention": ..., "entity": mm.normalize["precision"](mm.auto[Entity])})
    ),
)

b_cubed_recall = mm.preprocess(
    _entity_set_to_membership_set,
    mm.set_matching[Membership, "<->", "recall"](
        mm.dataclass[Membership]({"mention": ..., "entity": mm.normalize["recall"](mm.auto[Entity])})
    ),
)


ceaf_phi4 = mm.dataclass[EntitySet](
    {
        "entities": mm.set_matching[Entity, "<->"](
            mm.dataclass[Entity]({"mentions": mm.set_matching[Mention, "<->", "f1"](...)})
        )
    }
)


coref_suite = mm.suite(
    {
        "muc": mm.family(muc, mm.macro_average(["precision", "recall", "f1"])),
        "b_cubed": mm.suite(
            {
                "precision": mm.family(b_cubed_precision, mm.macro_average(["none"])),
                "recall": mm.family(b_cubed_recall, mm.macro_average(["none"])),
            }
        ).with_extra(
            lambda m: {
                "f1": (
                    2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
                    if (m["precision"] + m["recall"]) > 0
                    else 0.0
                )
            }
        ),
        "ceaf_phi4": mm.family(ceaf_phi4, mm.macro_average(["precision", "recall", "f1"])),
    }
).with_extra(lambda m: {"avg-f1": (m["muc-f1"] + m["b_cubed-f1"] + m["ceaf_phi4-f1"]) / 3})

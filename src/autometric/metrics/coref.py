"""Metric suite definitions for coreference resolution."""
from typing import Collection

import autometric.core.dsl as am
from autometric.core.metric import Metric
from autometric.structures.ie import Entity, EntitySet, Membership, Mention


def _muc_common_links(x: Entity, y: Entity) -> int:
    return max(0, len(set(x.mentions) & set(y.mentions)) - 1)


muc_link: Metric[Entity] = am.from_func(_muc_common_links)

muc = am.dataclass[EntitySet]({
    "entities": am.set_matching[Entity, "~"](muc_link)
})


def _entity_set_to_membership_set(es: EntitySet) -> Collection[Membership]:
    return [Membership(mention=m, entity=e) for e in es.entities for m in e.mentions]


b_cubed_precision = am.preprocess(
    _entity_set_to_membership_set,
    am.set_matching[Membership, "<->", "precision"](
        am.dataclass[Membership]({
            "mention": ...,
            "entity": am.normalize["precision"](am.auto[Entity])
        })
    )
)

b_cubed_recall = am.preprocess(
    _entity_set_to_membership_set,
    am.set_matching[Membership, "<->", "recall"](
        am.dataclass[Membership]({
            "mention": ...,
            "entity": am.normalize["recall"](am.auto[Entity])
        })
    )
)


ceaf_phi4 = am.dataclass[EntitySet]({
    "entities": am.set_matching[Entity, "<->"](
        am.dataclass[Entity]({
            "mentions": am.set_matching[Mention, "<->", "f1"](...)
        })
    )
})


coref_suite = am.suite({
    "muc": am.family(muc, am.macro_average(["precision", "recall", "f1"])),
    "b_cubed": am.suite({
            "precision": am.family(b_cubed_precision, am.macro_average(["none"])),
            "recall": am.family(b_cubed_recall, am.macro_average(["none"])),
        }).with_extra(lambda m: {
            "f1": (
                2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
                if (m["precision"] + m["recall"]) > 0 else 0.0
            )
        }),
    "ceaf_phi4": am.family(ceaf_phi4, am.macro_average(["precision", "recall", "f1"])),
}).with_extra(lambda m: {
    "avg-f1": (m["muc-f1"] + m["b_cubed-f1"] + m["ceaf_phi4-f1"]) / 3
})

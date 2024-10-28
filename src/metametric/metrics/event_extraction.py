"""Contains metrics for event extraction."""

import metametric.dsl as mm

from metametric.structures.ie import Event, EventSet, Trigger, Argument

trigger_identification = mm.dataclass[EventSet](
    {
        "events": mm.set_matching[Event, "<->"](
            mm.dataclass[Event](
                {
                    "trigger": mm.dataclass[Trigger](
                        {
                            "mention": ...,
                        }
                    ),
                }
            )
        )
    }
)

trigger_classification = mm.dataclass[EventSet](
    {
        "events": mm.set_matching[Event, "<->"](
            mm.dataclass[Event](
                {
                    "trigger": ...,
                }
            )
        )
    }
)

argument_identification = mm.dataclass[EventSet](
    {
        "events": mm.set_matching[Event, "<->"](
            mm.dataclass[Event](
                {
                    "trigger": ...,
                    "args": mm.set_matching[Argument, "<->"](
                        mm.dataclass[Argument](
                            {
                                "mention": ...,
                            }
                        )
                    ),
                }
            )
        )
    }
)

argument_classification = mm.dataclass[EventSet](
    {
        "events": mm.set_matching[Event, "<->"](
            mm.dataclass[Event]({"trigger": ..., "args": mm.set_matching[Argument, "<->"](...)})
        )
    }
)


event_extraction_suite = mm.suite(
    {
        "trigger_identification": mm.family(trigger_identification, mm.macro_average(["precision", "recall", "f1"])),
        "trigger_classification": mm.family(trigger_classification, mm.macro_average(["precision", "recall", "f1"])),
        "argument_identification": mm.family(argument_identification, mm.macro_average(["precision", "recall", "f1"])),
        "argument_classification": mm.family(argument_classification, mm.macro_average(["precision", "recall", "f1"])),
    }
)

# Information extraction


Information Extraction (IE) tasks employ a wide array of data structures. We support the following, all of which can be imported from `metametric.structures.ie`:

- `Mention`: A span of text, defined by its starting and ending indices (inclusive or exclusive) in a passage of text.
- `Relation`: A typed binary relation between two `Mention`s.
- `RelationSet`: A collection of `Relation`s.
- `Trigger`: A typed, event-denoting `Mention`.
- `Argument`: A typed, entity-denoting `Mention` that satisfies a certain `role` in an event.
- `Entity`: An entity, represented as a collection of `Mention`s that refer to it.
- `EntitySet`: A collection of `Entity`s.
- `Membership`: A membership relation between a particular `Mention` and the `Entity` it refers to.
- `Event`: A complete event, represented by a particular `Trigger` together with all of its `Argument`s.
- `EventSet`: A collection of `Event`s.

One can define a wide array of metrics based on these data structures.


### Coreference Resolution

We support three of the most widely used metrics for coreference resolution, including $\text{MUC}$ (`muc`; [paper](https://aclanthology.org/M95-1005/)), $B^3$ [`b_cubed_precision`, `b_cubed_recall`; [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ccdacc60d9d68dfc1f94e7c68bd56646c000e4ab)) and $\text{CEAF}_{\phi_4}$ (`ceaf_phi4`; [paper](https://aclanthology.org/H05-1004/)), as well as a metric suite (`coref_suite`) that includes all of these, plus the commonly reported average of all three. These metrics can be imported from `metametric.metrics.coref`.

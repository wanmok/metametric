## DSL for creating a metric

```python
import metametric.core.dsl as mm
```

## Data Structures

The `metametric` package includes a number of pre-defined data structures commonly used for various structured prediction tasks. These come equipped with their own identity-based metrics and can be used to quickly define new, more complex metrics. All such data structures be found in `metametric.structures`. Those currently supported are listed below:

### Information Extraction

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

### Semantic Parsing

We support two essential data structures for [Abstract Meaning Representation (AMR)](https://aclanthology.org/W13-2322/) parsing, which can be imported from `metametric.structures.amr`:

- `Prop`: A proposition, expressing some relation (`pred`) between a subject (`subj`) and an object (`obj`)
- `AMR`: A (rooted, directed, acyclic) AMR graph, represented as a collection of `Prop`s.


## Metric Implementations

Lastly, `metametric` provides several off-the-shelf implementations of common structured prediction metrics. The currently supported implementations are listed below, and this list may be expanded in the future. All implementations can be found in `metametric.metrics`.

### Coreference Resolution

We support three of the most widely used metrics for coreference resolution, including [$\text{MUC}$](https://aclanthology.org/M95-1005/) (`muc`), [$B^3$](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ccdacc60d9d68dfc1f94e7c68bd56646c000e4ab) (`b_cubed_precision`, `b_cubed_recall`), [$\text{CEAF}_{\phi_4}$](https://aclanthology.org/H05-1004/) (`ceaf_phi4`), as well as a metric suite (`coref_suite`) that includes all of these, plus the commonly reported average of all three. These metrics can be imported from `metametric.metrics.coref`.

### Semantic Parsing

We support the standard [Smatch score](https://aclanthology.org/P13-2131/) (`s_match`) for semantic parsing &mdash; most commonly used for AMR parsing. `s_match` can be imported from `metametrics.metrics.semantic_parsing`.
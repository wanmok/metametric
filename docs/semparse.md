### Semantic Parsing

We support two essential data structures for [Abstract Meaning Representation (AMR)](https://aclanthology.org/W13-2322/)
parsing, which can be imported from `metametric.structures.amr`:

- `Prop`: A proposition, expressing some relation (`pred`) between a subject (`subj`) and an object (`obj`)
- `AMR`: A (rooted, directed, acyclic) AMR graph, represented as a collection of `Prop`s.

We support the standard Smatch score (`s_match`; [paper](https://aclanthology.org/P13-2131/)) for semantic parsing
&mdash; most commonly used for AMR parsing. `s_match` can be imported from `metametrics.metrics.semantic_parsing`.

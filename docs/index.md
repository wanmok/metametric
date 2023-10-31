# Introduction

The `metametric` Python package offers a set of tools for quickly and easily defining and implementing evaluation
metrics for a variety of structured prediction tasks in natural language processing (NLP) based on the framework
presented in the following paper:

> [A Unified View of Evaluation Metrics for Structured Prediction](https://arxiv.org/abs/2310.13793). Yunmo Chen,
> William Gantt, Tongfei Chen, Aaron Steven White, and Benjamin Van Durme. *EMNLP 2023*.

The key features of the package include:

- A decorator for automatically defining and implementing a custom metric for an arbitrary `dataclass`.
- A collection of generic components for defining arbitrary new metrics based on the framework in the paper.
- Implementations of a number of metrics for common structured prediction tasks.

To install, run:

```bash
pip install metametric
```

# Quickstart

## Scoring a Pair of Objects

`metametric` comes with a set of prebuilt metrics for common structured prediction tasks. For example, to compute the standard suite of evaluation metrics for coreference resolution &mdash;
$\text{MUC}$ (`muc`; [paper](https://aclanthology.org/M95-1005/)),
$B^3$ [`b_cubed_precision`, `b_cubed_recall`; [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ccdacc60d9d68dfc1f94e7c68bd56646c000e4ab))
and $\text{CEAF}_{\phi_4}$ (`ceaf_phi4`; [paper](https://aclanthology.org/H05-1004/))
&mdash; on a pair of predicted and reference coreference clusters, one can simply do:

<div class="code-typing">

```python
from metametric.structures.ie import Mention, Entity, EntitySet
from metametric.metrics.coref import coref_suite

scorer = coref_suite.new()
for p, r in zip(predicted_entities, reference_entities):
    scorer.update_single(p, r)

metrics = scorer.compute()

>> {"muc-f1": 0.4, "b_cubed-f1": 0.45, "ceaf_phi4-f1": 0.52, "avg-f1": 0.46, ...}

```

</div>

## Defining a Metric

If you want to implement a *new* metric based on structure (and substructure) matching, you can leverage the
built-in matching algorithms and **just focus on the structure of interests**. For example, if you want to implement
`s_match` for [Abstract Meaning Representation (AMR)](https://aclanthology.org/W13-2322/) parsing, you need only define dataclasses for the relevant structures (here, `Prop` for AMR propositions and `AMR` for AMR graphs), and then define the matching between them:

<div class="code-typing">

```python
# Define structures
@metametric()
@dataclass(eq=True, frozen=True)
class Prop:
    """A Proposition in an AMR."""

    subj: Variable
    pred: str
    obj: Union[Variable, str]


@metametric()
@dataclass
class AMR:
    """Abstract Meaning Representation (AMR) structure."""

    props: Collection[Prop]
    

# Let metametric derive Smatch for you!
s_match = mm.normalize["f1"](mm.auto[AMR])

# `s_match` is now ready to use on any pair of `AMR` graphs!

```

</div>

# Citation

If you use this codebase (package) in your work, please cite the following paper:

```tex
@inproceedings{metametric,
    title={A Unified View of Evaluation Metrics for Structured Prediction},
    author={Yunmo Chen and William Gantt and Tongfei Chen and Aaron Steven White and Benjamin {Van Durme}},
    booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    year={2023},
    address={Singapore},
}
```

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

If you use this codebase in your work, please cite the following paper:

```tex
@inproceedings{metametric,
    title={A Unified View of Evaluation Metrics for Structured Prediction},
    author={Yunmo Chen and William Gantt and Tongfei Chen and Aaron Steven White and Benjamin {Van Durme}},
    booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    year={2023},
    address={Singapore},
}
```

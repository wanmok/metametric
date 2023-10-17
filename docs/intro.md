# autometric

The `autometric` Python package offers a set of tools for quickly and easily defining and implementing evaluation metrics for a variety of structured prediction tasks in natural language processing (NLP) based on the framework presented in the following paper:

> [A Unified View of Evaluation Metrics for Structured Prediction](/insert/link/here). Yunmo Chen, William Gantt, Tongfei Chen, Aaron Steven White, and Benjamin Van Durme. *EMNLP 2023*.

The key features of the package include:

- A decorator for automatically defining and implementing a custom metric for an arbitrary `dataclass`.
- A collection of generic components for defining arbitrary new metrics based on the framework in the paper.
- Implementations of a number of metrics for common structured prediction tasks.

See [decorator.md] for a more detailed discussion of the decorator and see [dsl.md] for an overview of the metric components and metric implementations.
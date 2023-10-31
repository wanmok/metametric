```python
import metametric.dsl as mm
```

#### Auto

`mm.auto[X]` derives an automatic metric for type `X`. This is the default behavior of the `@metametric` decorator.

#### Discrete similarity

`mm.discrete[X]` constructs a discrete similarity metric for type `X`. That is, given that type `X` has method `__eq__`,
the metric returns 1.0 if two objects of type `X` are equal, and 0.0 otherwise.

$$ \text{sim}(x, y) = \begin{cases} 1 & \text{if } x = y \\ 0 & \text{otherwise} \end{cases} $$

#### Custom similarity

`mm.from_func(f)` constructs a metric from a function `f: Callable[[X, X], float]` that takes two arguments of the same
type and returns a `float`.

$$ \text{sim}(x, y) = f(x, y) $$

#### With preprocessing

`mm.preprocess(g, M)` is a metric that first applies a preprocessing function `g: Callable[[X], Y]` to both arguments,
and then applies a metric `f: Metric[Y]` to the results.
This is the contramap operation of the metric type.

$$ \text{sim}(x, y) = f(g(x), g(y)) $$

#### Product (dataclass) similarity

`mm.dataclass[X](M)` constructs a metric for a dataclass `X` by taking the product of the metrics for each of its fields
defined in `M: Dict[str, Metric[Any]]`.

$$ \text{sim}(x, y) = \prod_{(f, m_f) \in M} m_f(x.\!f, y.\!f) $$

#### Sum (union) similarity

`mm.union[X](M)` constructs a metric for a union type `X` by a dictionary of each case of the union defined
in `M: Dict[type, Metric[Any]]`.

$$ \text{sim}(x, y) = \sum_{(t, m_t) \in M} \mathbb{1}_{x \in t} \mathbb{1}_{y \in t} m_t(x, y) $$

#### Set matching similarity

`mm.set_matching[X, ◇, N](f)` constructs a set matching metric between two objects of type `Set[X]`,
with $\diamond \in \{\leftrightarrow, \to, \leftarrow, \sim\}$ as the matching constraint, and `N` as the normalizer.

$$ \Sigma^{\diamond}[f](x, y) = \max_{M^\diamond} \sum_{(u, v) \in M^\diamond} f(u, v) $$

$$ \textrm{sim}(x, y) = \mathsf{N}(\Sigma^{\diamond}[f](x, y)) $$

#### Latent set matching similarity
`mm.latet_set_matching[X, ◇, N](f)` constructs a latent set matching metric between two objects of type `Set[X]` where `X` has `Variable`s.

\[ \Sigma(X, Y) = \max_{M^{\leftrightarrow}_V, M^\diamond} \sum_{(u, v) \in M^\diamond} \phi_T(u, v) \]

where $M^\diamond$ is a matching between $X$ and $Y$ according to the specified matching constraint, and
$M^\leftrightarrow_V$ is a one-to-one matching between the variables in $X$ and $Y$.

`mm.latet_set_matching[X, ◇, N](f)` constructs a latent set matching metric between two objects of type `Set[X]`
where `X` has `Variable`s.

\[ \Sigma(X, Y) = \max_{M^{\leftrightarrow}_V, M^\diamond} \sum_{(u, v) \in M^\diamond} \phi_T(u, v) \]

where $M^\diamond$ is a matching between $X$ and $Y$ according to the specified matching constraint, and
$M^\leftrightarrow_V$ is a one-to-one matching between the variables in $X$ and $Y$.

#### Sequence matching similarity

#### Graph matching similarity

# Decorator

`metametric` provides a Python decorator (`@metametric`) for automatically deriving a metric given an arbitrary dataclass `D`. Practically speaking, this instantiates a new `Metric` object based on the dataclass definition and on the arguments passed to the decorator, and assigns this object to a new `metric` class attribute (or `latent_metric` if the class has latent variables). To compute the derived metric for a pair of objects `p` and `r` of type `D`, one then need only call `D.metric.score(p,r)`.

The decorator takes two parameters, `normalizer` and `constraint`, which we detail below. We also provide an example of its use.

## Normalizer
The `normalizer` parameter specifies how the raw score computed by the metric should be normalized. As illustrated in the [paper](link) associated with this package, these normalizers can be understood in terms of the **overlap** ($\Sigma$) between a predicted set $P$ and a reference set $R$, where $P, R \subseteq X$ for some set of discrete elements $X$:
$$\Sigma_\delta(P,R) = \lvert P \cap R \rvert$$
where $\delta$ is the similarity function used for elements of $X$.

Currently, the following choices are supported:
- `none` (*default*): No normalization. This is the default and will not apply any normalization to the raw metric score.
- `precision`: Standard precision, i.e., the overlap normalized by the size of the *predicted* set, $P$. Formally, $\mathrm{P}(P,R) = \frac{\lvert P \cap R \rvert}{\lvert P \rvert} = \frac{\Sigma_\delta(P,R)}{\Sigma_\delta(P,P)}$.
- `recall`: Standard recall, i.e., the overlap normalized by the size of the *reference* set, $R$. Formally, $\mathrm{R}(P,R) = \frac{\lvert P \cap R \rvert}{\lvert R \rvert} = \frac{\Sigma_\delta(P,R)}{\Sigma_\delta(R,R)}$.
- `jaccard`: The Jaccard similarity or *intersection-over-union*, i.e., the overlap of $P$ and $R$ normalized by their union. Formally, $\mathrm{J}(P,R) = \frac{\lvert P \cap R \rvert}{\lvert R \rvert} = \frac{\Sigma_\delta(P,R)}{\Sigma_\delta(R,R) + \Sigma_\delta(P,P) - \Sigma_\delta(P,R)}$.
- `dice`: The *dice score* more commonly known as *$\rm F_1$ score*, i.e., $\frac{2 * \text{precision} * \text{recall}}{\text{precision} + \text{recall}}$.
- `f{beta}`: $\rm F_\beta$ score, or generalized $\rm F$ score, where $\beta$ is a positive real number that indicates the relative weighting of precision vs. recall: $(1 + \beta^2) * \frac{\text{precision} * \text{recall}}{(\beta^2 * \text{precision}) + \text{recall}}$. Note that $\beta = 1$ recovers the dice score. Any positive float may be used for `{beta}`, e.g., `f0.5`, `f2`, etc.

## Constraint
The `constraint` parameter specifies restrictions on the *matching* (i.e. the alignment) between predicted and reference objects of the dataclass's type. The following choices are supported; each choice can be written one of two ways:
- **One-to-One** (`<->` or `1:1`; *default*): this specifies a *partial bijection* constraint: each predicted object can be aligned to *at most one* reference object, and vice-versa. The overwhelming majority of metrics impose this constraint, and so it is the default option.
- **One-to-Many** (`->` or `1:*`): this specifies a (non-bijective) *partial function* from *predicted* objects to *reference* objects: each predicted object can be aligned to *at most one* reference object, but the same reference object can potentially be aligned to *multiple* predicted ones.
- **Many-to-One** (`<-` or `*:1`): this specifies a (non-bijective) *partial function* from *reference* objects to *predicted* objects: each reference object can be aligned to *at most one* predicted object, but the same predicted object can potentially be aligned to *multiple* reference ones. (**N.B.**: while we provide support for this constraint, we aren't aware of actual metrics that impose it.)
- **No Constraints** (`~` or `*:*`): this specifies a generic *relation*: each predicted object can be aligned to multiple reference objects, and vice-versa.

## Example: Event Trigger F1

Here, we show an example of how to use the decorator to automatically derive a metric for a dataclass &mdash; specifically, $\rm F_1$ (dice score), commonly used for event extraction.

An event trigger is just a word or phrase (i.e. a *mention*) in a passage of text that evokes an event, like "kick" or "bombing", and that's associated with some event type. First, we'll define a dataclass for mentions:

```python
@metametric(normalizer="none", constraint="<->")
@dataclass(eq=True, frozen=True)
class Mention:
	left: int  # left character offset of the mention (inclusive)
	right: int # right character offset of the mention (inclusive)
```

The dataclass just has two attributes &mdash; a left index and a right index &mdash; indicating the character offsets of the start and end of mention within the passage of text. (We assume here they are both inclusive, though they need not be.) Note that above the `dataclass` decorator, we have added the `metametric` decorator as well, using the default values for the `normalizer` and `constraint` parameters (we could just as well have written `@metametric()`, but have written out the defaults explicitly for clarity). As discussed [above](#decorator), this sets a new `metric` attribute on the `Mention` dataclass. In this case, it's just about the simplest metric you could have &mdash; an indicator function (or [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta)) that returns 1 iff two mentions have the same `left` and `right` offsets, and zero otherwise. Let's try it out:

```python
m1 = Mention(1,2)
m2 = Mention(1,2)
m3 = Mention(1,3)

> Mention.metric.score(m1, m2) # returns 1.0, since m1 == m2
> Mention.metric.score(m1, m3) # returns 0.0, since m1 != m3
```
You might wonder why one would go to all this trouble for such simple functionality. The value of the `@metametric` decorator becomes more apparent when working with more complex dataclasses, where some fields may *themselves* be dataclasses. The `Trigger` dataclass, which is just an event-denoting `Mention` paired with its type, is an example of this:

```python
@metametric(normalizer="none", constraint="<->")
@dataclass
class Trigger:
	mention: Mention
	type: str
```
The decorator is the same as above, but the automatically derived metric for `Trigger`s will recursively evaluate the `mention` field using the automatically derived metric for *that* dataclass:
```python
# m1, m2, m3 are as defined above
t1 = Trigger(m1, "foo")
t2 = Trigger(m2, "foo")
t3 = Trigger(m3, "foo")

Trigger.metric.score(t1, t2) # returns 1.0, since m1 == m2 and t1.type == t2.type
Trigger.metric.score(t1, t3) # returns 0.0, since m1 != m2 (though t1.type == t2.type)
```
Setting aside the problem of *argument extraction*, let's imagine that the output for our trigger extraction task is just a collection of `Trigger`s. We can define a final `dataclass` for storing these outputs:
```python
@metametric(normalizer="f1", constraint="<->")
@dataclass
class TriggerExtractionOutput:
	triggers: Collection[Trigger]
```
*This* gives us our trigger $\rm F_1$ score. We can now compute it as follows:
```python
# the predicted event triggers (supposing our system predicts t1 and t2 only)
# triggers t1, t2, and t3 are as defined above
predicted_triggers = [t1, t2]
predictions = TriggerExtractionOutput(predicted_triggers)

# the reference event triggers (t1, t2, and t3)
reference_triggers = [t1, t2, t3]
references = TriggerExtractionOutput(reference_triggers)

# compute F1 score for predictions against the references
TriggerExtractionOutput.metric.score(predictions, references) # returns 0.8
```

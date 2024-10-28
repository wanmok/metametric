"""Tests for SMatch derived from the AMR structure."""

from pytest import approx

from metametric import Variable
from metametric.core.matching import Hook
from metametric.structures.amr import AMR, Prop
from metametric.metrics.semantic_parsing import s_match


def test_smatch():
    """Example from https://aclanthology.org/P13-2131.pdf."""
    amr1 = AMR(
        props=[
            a11 := Prop(Variable("a"), "instance", "want-01"),
            a12 := Prop(Variable("b"), "instance", "boy"),
            a13 := Prop(Variable("c"), "instance", "go-01"),
            a14 := Prop(Variable("a"), "ARG0", Variable("b")),
            a15 := Prop(Variable("a"), "ARG1", Variable("c")),
            a16 := Prop(Variable("c"), "ARG0", Variable("b")),
        ]
    )

    amr2 = AMR(
        props=[
            a21 := Prop(Variable("x"), "instance", "want-01"),
            a22 := Prop(Variable("y"), "instance", "boy"),
            a23 := Prop(Variable("z"), "instance", "football"),
            a24 := Prop(Variable("x"), "ARG0", Variable("y")),
            a25 := Prop(Variable("x"), "ARG1", Variable("z")),
        ]
    )

    score, matching = s_match.compute(amr1, amr2)
    assert score == approx(0.73, abs=0.01)

    matches = []
    hooks = {"props[*]": Hook.from_callable(lambda i, pp, p, rp, r, s: matches.append((p, r)))}

    matching.run_with_hooks(hooks)
    assert matches == [
        (a11, a21),
        (a12, a22),
        (a14, a24),
        (a15, a25),
    ]

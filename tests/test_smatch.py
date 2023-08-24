from pytest import approx

from unimetric.metric import FScore
from unimetric.amr import AMR, Variable, Prop
from unimetric.metrics.smatch import SMatchCount


def test_smatch():
    """
    Example from https://aclanthology.org/P13-2131.pdf.
    """

    amr1 = AMR(
        props=[
            Prop(Variable('a'), 'instance', 'want-01'),
            Prop(Variable('b'), 'instance', 'boy'),
            Prop(Variable('c'), 'instance', 'go-01'),
            Prop(Variable('a'), 'ARG0', Variable('b')),
            Prop(Variable('a'), 'ARG1', Variable('c')),
            Prop(Variable('c'), 'ARG0', Variable('b')),
        ]
    )

    amr2 = AMR(
        props=[
            Prop(Variable('x'), 'instance', 'want-01'),
            Prop(Variable('y'), 'instance', 'boy'),
            Prop(Variable('z'), 'instance', 'football'),
            Prop(Variable('x'), 'ARG0', Variable('y')),
            Prop(Variable('x'), 'ARG1', Variable('z')),
        ]
    )

    smatch = FScore(SMatchCount())
    assert smatch.score(amr1, amr2) == approx(0.73, abs=0.01)

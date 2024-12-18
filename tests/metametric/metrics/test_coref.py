"""Tests for coreference metrics."""

from pytest import fixture, approx

import metametric.dsl as mm
from metametric.structures.ie import Mention, Entity, EntitySet
from metametric.metrics.coref import muc, b_cubed_precision, b_cubed_recall, ceaf_phi4, coref_suite


@fixture
def data():
    """From https://aclanthology.org/P14-2006.pdf."""
    a = Mention(left=0, right=1)
    b = Mention(left=2, right=3)
    c = Mention(left=4, right=5)
    d = Mention(left=6, right=7)
    e = Mention(left=8, right=9)
    f = Mention(left=10, right=11)
    g = Mention(left=12, right=13)
    h = Mention(left=14, right=15)
    i = Mention(left=16, right=17)

    pred = EntitySet(
        entities=[
            Entity(mentions=[a, b]),
            Entity(mentions=[c, d]),
            Entity(mentions=[f, g, h, i]),
        ]
    )
    ref = EntitySet(
        entities=[
            Entity(mentions=[a, b, c]),
            Entity(mentions=[d, e, f, g]),
        ]
    )
    return pred, ref


def test_muc(data):
    """MUC metric."""
    pred, ref = data
    muc_precision = mm.normalize["precision"](muc)
    muc_recall = mm.normalize["recall"](muc)
    assert muc_precision.score(pred, ref) == approx(0.40, abs=0.01)
    assert muc_recall.score(pred, ref) == approx(0.40, abs=0.01)


def test_b_cubed(data):
    """B-cubed metric."""
    pred, ref = data
    assert b_cubed_precision.score(pred, ref) == approx(0.50, abs=0.01)
    assert b_cubed_recall.score(pred, ref) == approx(0.42, abs=0.01)


def test_ceaf_phi4(data):
    """CEAF phi4 metric."""
    pred, ref = data
    ceaf_phi4_precision = mm.normalize["precision"](ceaf_phi4)
    ceaf_phi4_recall = mm.normalize["recall"](ceaf_phi4)
    assert ceaf_phi4_precision.score(pred, ref) == approx(0.43, abs=0.01)
    assert ceaf_phi4_recall.score(pred, ref) == approx(0.65, abs=0.01)


def test_coref_family(data):
    """Coreference suite."""
    pred, ref = data
    agg = coref_suite.new()
    agg.update_single(pred, ref)
    metrics = agg.compute()
    assert metrics["muc-precision"] == approx(0.40, abs=0.01)
    assert metrics["muc-recall"] == approx(0.40, abs=0.01)
    assert metrics["muc-f1"] == approx(0.40, abs=0.01)
    assert metrics["b_cubed-precision"] == approx(0.50, abs=0.01)
    assert metrics["b_cubed-recall"] == approx(0.42, abs=0.01)
    assert metrics["b_cubed-f1"] == approx(0.46, abs=0.01)
    assert metrics["ceaf_phi4-precision"] == approx(0.43, abs=0.01)
    assert metrics["ceaf_phi4-recall"] == approx(0.65, abs=0.01)
    assert metrics["ceaf_phi4-f1"] == approx(0.52, abs=0.01)
    assert metrics["avg-f1"] == approx(0.46, abs=0.01)

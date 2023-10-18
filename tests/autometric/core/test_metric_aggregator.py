"""Tests for metric aggregators."""
from pytest import approx

import metametric.core.dsl as am


def test_metric_aggregator():
    """Basic test for metric aggregator that computes precision, recall, and F-score."""
    a = [[0, 1], [2], [1, 2]]
    b = [[0, 1, 2, 3], [2, 3], [1, 2, 3]]

    mf = am.family(
        am.set_matching[int, '<->', 'none'](...),
        {
            "macro": am.macro_average(["precision", "recall", "f1", "f0.5", "f2"]),
            "micro": am.micro_average(["precision", "recall", "f1", "f0.5", "f2"]),
        }
    )
    agg = mf.new()
    agg.update_batch(a, b)
    metrics = agg.compute()

    assert metrics["micro-precision"] == approx(1.0, abs=0.01)
    assert metrics["micro-recall"] == approx(0.55, abs=0.01)
    assert metrics["micro-f1"] == approx(0.71, abs=0.01)
    assert metrics["micro-f0.5"] == approx(0.86, abs=0.01)
    assert metrics["micro-f2"] == approx(0.61, abs=0.01)
    assert metrics["macro-precision"] == approx(1.0, abs=0.01)
    assert metrics["macro-recall"] == approx(0.55, abs=0.01)
    assert metrics["macro-f1"] == approx(0.71, abs=0.01)
    assert metrics["macro-f0.5"] == approx(0.86, abs=0.01)
    assert metrics["macro-f2"] == approx(0.61, abs=0.01)

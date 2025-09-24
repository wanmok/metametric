"""Tests for retrieval metrics."""

from pytest import approx
from metametric.metrics.retrieval import ranking_ap


def test_retrieval():
    """Basic test for retrieval metrics."""
    predicted = [
        ("a", 0.4),
        ("b", 0.3),
        ("c", 0.2),
        ("d", 0.1),
    ]

    reference = [
        ("c", 1.0),
        ("d", 1.0),
        ("e", 1.0),
    ]
    assert ranking_ap.score(predicted, reference) == approx(0.2778, abs=0.01)

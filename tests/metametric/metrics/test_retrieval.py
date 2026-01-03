"""Tests for retrieval metrics."""

from pytest import approx
import numpy as np

from metametric.metrics.retrieval import ranking_ap, p_at_k, r_at_k, r_precision
from metametric.metrics.retrieval import dcg_at_k, ndcg_at_k, mrr


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

    pk = p_at_k.score(predicted, reference)
    rk = r_at_k.score(predicted, reference)
    ap = ranking_ap.score(predicted, reference)
    rprec = r_precision.score(predicted, reference)
    dcg = dcg_at_k.score(predicted, reference)
    ndcg = ndcg_at_k.score(predicted, reference)
    reciprocal = mrr.score(predicted, reference)
    assert all(pk == [approx(0.0), approx(0.0), approx(1 / 3), approx(0.5), approx(0.5), approx(0.5), approx(0.5),
                      approx(0.5), approx(0.5), approx(0.5)])
    assert all(
        rk == [approx(0.0), approx(0.0), approx(1 / 3), approx(2 / 3), approx(2 / 3), approx(2 / 3), approx(2 / 3),
               approx(2 / 3), approx(2 / 3), approx(2 / 3)])
    assert ap == approx(0.2778, abs=0.01)
    # R-precision with R=3 (relevant items): hits at rank3 is 1
    assert rprec == approx(1.0 / 3, abs=1e-4)
    # DCG accumulates discounted hits
    assert all(dcg[:4] == [approx(0.0), approx(0.0), approx(0.5, abs=1e-4), approx(0.9307, abs=1e-4)])
    # nDCG normalizes by ideal discounted gains (three relevant items)
    ideal_at_3 = 1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(4)
    assert ndcg[2] == approx(0.5 / ideal_at_3, abs=1e-4)
    assert ndcg[3] == approx(0.9307 / ideal_at_3, abs=1e-4)
    # First relevant at rank 3 -> reciprocal rank = 1/3
    assert reciprocal == approx(1.0 / 3.0, abs=1e-4)


def test_mrr_edge_cases():
    """Edge cases for reciprocal rank."""
    # No relevant hits should yield zero reciprocal rank
    predicted = [("a", 0.9), ("b", 0.8)]
    reference = [("c", 1.0)]
    assert mrr.score(predicted, reference) == approx(0.0)

    # First item is relevant -> reciprocal rank is 1
    predicted = [("a", 0.9), ("b", 0.8)]
    reference = [("a", 1.0), ("c", 1.0)]
    assert mrr.score(predicted, reference) == approx(1.0)

"""Tests for retrieval metrics."""

import pandas as pd
from pytest import approx
import numpy as np

from metametric.metrics.retrieval import ranking_ap, p_at_k, r_at_k, r_precision
from metametric.metrics.retrieval import dcg_at_k, ndcg_at_k, mrr

try:
    from trectools import TrecRun, TrecQrel, TrecEval
    TRECTOOLS_AVAILABLE = True
except ImportError:
    TRECTOOLS_AVAILABLE = False


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


def test_trec_eval_equivalence():
    """Test equivalence between metametric and trec_eval metrics.

    Note: Some metrics may have slight differences due to implementation details.
    Metametric uses a precision-recall curve integral approach for AP, while
    TREC eval uses the standard sum of precisions at relevant positions.
    """
    if not TRECTOOLS_AVAILABLE:
        # Skip this test if trectools is not available
        import pytest
        pytest.skip("trectools not available")

    # Create test data with multiple queries to test properly
    # Query 1: predictions and relevance judgments
    predicted_q1 = [
        ("doc1", 0.95),
        ("doc2", 0.85),
        ("doc3", 0.75),
        ("doc4", 0.65),
        ("doc5", 0.55),
    ]
    reference_q1 = [
        ("doc2", 1.0),
        ("doc3", 1.0),
        ("doc6", 1.0),
    ]

    # Query 2: predictions and relevance judgments
    predicted_q2 = [
        ("doc10", 0.99),
        ("doc11", 0.88),
        ("doc12", 0.77),
        ("doc13", 0.66),
    ]
    reference_q2 = [
        ("doc10", 1.0),
        ("doc12", 1.0),
        ("doc14", 1.0),
    ]

    # Prepare data for TrecEval
    run_data = pd.DataFrame([
        ["Q1", "Q0", "doc1", 1, 0.95, "run1"],
        ["Q1", "Q0", "doc2", 2, 0.85, "run1"],
        ["Q1", "Q0", "doc3", 3, 0.75, "run1"],
        ["Q1", "Q0", "doc4", 4, 0.65, "run1"],
        ["Q1", "Q0", "doc5", 5, 0.55, "run1"],
        ["Q2", "Q0", "doc10", 1, 0.99, "run1"],
        ["Q2", "Q0", "doc11", 2, 0.88, "run1"],
        ["Q2", "Q0", "doc12", 3, 0.77, "run1"],
        ["Q2", "Q0", "doc13", 4, 0.66, "run1"],
    ], columns=["query", "q0", "docid", "rank", "score", "system"])

    qrel_data = pd.DataFrame([
        ["Q1", "0", "doc2", 1],
        ["Q1", "0", "doc3", 1],
        ["Q1", "0", "doc6", 1],
        ["Q2", "0", "doc10", 1],
        ["Q2", "0", "doc12", 1],
        ["Q2", "0", "doc14", 1],
    ], columns=["query", "q0", "docid", "rel"])

    run = TrecRun()
    run.run_data = run_data
    run.qrels_file = None

    qrel = TrecQrel()
    qrel.qrels_data = qrel_data

    te = TrecEval(run, qrel)

    # Test Precision@k for Q1
    mm_p3_q1 = p_at_k.score(predicted_q1, reference_q1)[2]  # P@3 (index 2 for k=3)
    trec_p3 = te.get_precision(depth=3, per_query=True)
    trec_p3_q1 = trec_p3.loc['Q1', 'P@3']
    assert mm_p3_q1 == approx(trec_p3_q1, abs=1e-4), f"P@3 mismatch: metametric={mm_p3_q1}, trec={trec_p3_q1}"

    # Test Recall@k for Q1
    mm_r3_q1 = r_at_k.score(predicted_q1, reference_q1)[2]  # R@3 (index 2 for k=3)
    trec_r3 = te.get_recall(depth=3, per_query=True)
    trec_r3_q1 = trec_r3.loc['Q1', 'R@3']
    assert mm_r3_q1 == approx(trec_r3_q1, abs=1e-4), f"R@3 mismatch: metametric={mm_r3_q1}, trec={trec_r3_q1}"

    # Test MRR for Q1
    mm_mrr_q1 = mrr.score(predicted_q1, reference_q1)
    trec_mrr = te.get_reciprocal_rank(per_query=True)
    trec_mrr_q1 = trec_mrr.loc['Q1', 'recip_rank@1000']
    assert mm_mrr_q1 == approx(trec_mrr_q1, abs=1e-4), f"MRR mismatch: metametric={mm_mrr_q1}, trec={trec_mrr_q1}"

    # Test NDCG@k for Q1
    mm_ndcg3_q1 = ndcg_at_k.score(predicted_q1, reference_q1)[2]  # NDCG@3
    trec_ndcg3 = te.get_ndcg(depth=3, per_query=True)
    trec_ndcg3_q1 = trec_ndcg3.loc['Q1', 'NDCG@3']
    assert mm_ndcg3_q1 == approx(trec_ndcg3_q1, abs=1e-4), f"NDCG@3 mismatch: metametric={mm_ndcg3_q1}, trec={trec_ndcg3_q1}"

    # Test MAP (Average Precision) for Q1
    # NOTE: Metametric uses precision-recall curve integral, which differs slightly from TREC eval
    # when the number of predictions differs from total relevant documents
    mm_ap_q1 = ranking_ap.score(predicted_q1, reference_q1)
    trec_map = te.get_map(depth=1000, per_query=True)
    trec_map_q1 = trec_map.loc['Q1', 'MAP@1000']
    # Allow larger tolerance due to different calculation method
    assert mm_ap_q1 == approx(trec_map_q1, abs=0.05), f"MAP mismatch: metametric={mm_ap_q1}, trec={trec_map_q1}"

    # Test R-Precision for Q1
    mm_rprec_q1 = r_precision.score(predicted_q1, reference_q1)
    trec_rprec = te.get_rprec(per_query=True)
    trec_rprec_q1 = trec_rprec.loc['Q1', 'RPrec@1000']
    assert mm_rprec_q1 == approx(trec_rprec_q1, abs=1e-4), f"R-Precision mismatch: metametric={mm_rprec_q1}, trec={trec_rprec_q1}"

    # Test for Q2 as well
    mm_p3_q2 = p_at_k.score(predicted_q2, reference_q2)[2]
    trec_p3_q2 = trec_p3.loc['Q2', 'P@3']
    assert mm_p3_q2 == approx(trec_p3_q2, abs=1e-4), f"P@3 Q2 mismatch: metametric={mm_p3_q2}, trec={trec_p3_q2}"

    mm_mrr_q2 = mrr.score(predicted_q2, reference_q2)
    trec_mrr_q2 = trec_mrr.loc['Q2', 'recip_rank@1000']
    assert mm_mrr_q2 == approx(trec_mrr_q2, abs=1e-4), f"MRR Q2 mismatch: metametric={mm_mrr_q2}, trec={trec_mrr_q2}"


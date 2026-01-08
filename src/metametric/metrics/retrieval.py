"""Metric definitions for retrieval tasks."""

import metametric.dsl as mm


def sort_by_score(x: list[tuple[str, float]]) -> list[str]:
    """Sort a list of (str, float) tuples by the float value."""
    return [u for u, _ in sorted(x, key=lambda t: t[1], reverse=True)]


def create_retrieval_metrics(max_k: int = 10, extend_score_self: bool = False):
    """Create retrieval metrics with configurable max_k.

    Args:
        max_k: Maximum number of positions to consider in ranking (default: 10).
               For equivalence with TREC eval, use a large value like 1000.
        extend_score_self: If True, extends score_self beyond input length up to max_k.
                          This is needed for TREC eval equivalence (default: False).

    Returns:
        A dictionary containing all retrieval metrics configured with the specified max_k.
    """
    ranking_match = mm.preprocess_param[sort_by_score](mm.ranking[(max_k, extend_score_self)](mm.auto[str]))

    return {
        "p_at_k": mm.normalize_param["precision@k"](ranking_match),
        "r_at_k": mm.normalize_param["recall@k"](ranking_match),
        "ranking_ap": mm.normalize_param["ranking_ap"](ranking_match),
        "r_precision": mm.normalize_param["r_precision"](ranking_match),
        "dcg_at_k": mm.normalize_param["dcg@k"](ranking_match),
        "ndcg_at_k": mm.normalize_param["ndcg@k"](ranking_match),
        "mrr": mm.normalize_param["reciprocal_rank"](ranking_match),
    }


# Default metrics with max_k=10 for backward compatibility
_default_metrics = create_retrieval_metrics(max_k=10)

ranking_match = mm.preprocess_param[sort_by_score](mm.ranking[10](mm.auto[str]))
p_at_k = _default_metrics["p_at_k"]
r_at_k = _default_metrics["r_at_k"]
ranking_ap = _default_metrics["ranking_ap"]
r_precision = _default_metrics["r_precision"]
dcg_at_k = _default_metrics["dcg_at_k"]
ndcg_at_k = _default_metrics["ndcg_at_k"]
# Reciprocal rank / MRR
mrr = _default_metrics["mrr"]

"""Metric definitions for retrieval tasks."""

import metametric.dsl as mm


def sort_by_score(x: list[tuple[str, float]]) -> list[str]:
    """Sort a list of (str, float) tuples by the float value."""
    return [u for u, _ in sorted(x, key=lambda t: t[1], reverse=True)]


ranking_match = mm.preprocess_param[sort_by_score](mm.ranking[10](mm.auto[str]))

p_at_k = mm.normalize_param["precision@k"](ranking_match)
r_at_k = mm.normalize_param["recall@k"](ranking_match)
ranking_ap = mm.normalize_param["ranking_ap"](ranking_match)

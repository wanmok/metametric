"""Tests for metrics derived with alignments."""
import metametric.core.dsl as am


def test_solve_alignment():
    """Test the alignment solver."""
    a = [1, 2, 2]
    b = [1, 1, 1, 2]

    assert am.set_matching[int, '<->', 'none'](...).score(a, b) == 2
    assert am.set_matching[int, '->', 'none'](...).score(a, b) == 3
    assert am.set_matching[int, '<-', 'none'](...).score(a, b) == 4
    assert am.set_matching[int, '~', 'none'](...).score(a, b) == 5

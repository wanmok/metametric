"""Tests for metrics derived with alignments."""
import metametric.dsl as mm


def test_solve_alignment():
    """Test the alignment solver."""
    a = [1, 2, 2]
    b = [1, 1, 1, 2]
    c = []

    # non-empty prediction and non-empty reference
    assert mm.set_matching[int, '<->', 'none'](...).score(a, b) == 2
    assert mm.set_matching[int, '->', 'none'](...).score(a, b) == 3
    assert mm.set_matching[int, '<-', 'none'](...).score(a, b) == 4
    assert mm.set_matching[int, '~', 'none'](...).score(a, b) == 5

    # empty reference
    assert mm.set_matching[int, '<->', 'none'](...).score(a,c) == 0
    assert mm.set_matching[int, '->', 'none'](...).score(a,c) == 0
    assert mm.set_matching[int, '<-', 'none'](...).score(a,c) == 0
    assert mm.set_matching[int, '~', 'none'](...).score(a,c) == 0

    # empty prediction
    assert mm.set_matching[int, '<->', 'none'](...).score(c,a) == 0
    assert mm.set_matching[int, '->', 'none'](...).score(c,a) == 0
    assert mm.set_matching[int, '<-', 'none'](...).score(c,a) == 0
    assert mm.set_matching[int, '~', 'none'](...).score(c,a) == 0

    # self-scoring
    assert mm.set_matching[int, '<->', 'none'](...).score(a,a) == 2
    assert mm.set_matching[int, '->', 'none'](...).score(a,a) == 3
    assert mm.set_matching[int, '<-', 'none'](...).score(a,a) == 3
    assert mm.set_matching[int, '~', 'none'](...).score(a,a) == 5

    assert mm.set_matching[int, '<->', 'none'](...).score(b,b) == 2
    assert mm.set_matching[int, '->', 'none'](...).score(b,b) == 4
    assert mm.set_matching[int, '<-', 'none'](...).score(b,b) == 4
    assert mm.set_matching[int, '~', 'none'](...).score(b,b) == 10

    assert mm.set_matching[int, '<->', 'none'](...).score(c,c) == 1
    assert mm.set_matching[int, '->', 'none'](...).score(c,c) == 1
    assert mm.set_matching[int, '<-', 'none'](...).score(c,c) == 1
    assert mm.set_matching[int, '~', 'none'](...).score(c,c) == 1

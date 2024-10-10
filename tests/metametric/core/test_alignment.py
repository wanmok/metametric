"""Tests for metrics derived with alignments."""
import metametric.dsl as mm


def test_solve_alignment():
    """Test the alignment solver."""
    a = [1, 2, 2]
    b = [1, 1, 1, 2]
    c = []
    
    m0 = mm.set_matching[int, '<->', 'none'](...)
    m1 = mm.set_matching[int, '->', 'none'](...)
    m2 = mm.set_matching[int, '<-', 'none'](...)
    m3 = mm.set_matching[int, '~', 'none'](...)

    _, matching0 = m0.compute(a, b)
    matches = []
    hooks = {
        "[*]": mm.Hook.from_callable(lambda i, pp, p, rp, r, s: matches.append((p, r)))
    }
    matching0.run_with_hooks(hooks)
    assert matches == [(1, 1), (2, 2)]

    # non-empty prediction and non-empty reference
    assert m0.score(a, b) == 2
    assert m1.score(a, b) == 3
    assert m2.score(a, b) == 4
    assert m3.score(a, b) == 5
    # empty reference
    assert m0.score(a, c) == 0
    assert m1.score(a, c) == 0
    assert m2.score(a, c) == 0
    assert m3.score(a, c) == 0

    # empty prediction
    assert m0.score(c, a) == 0
    assert m1.score(c, a) == 0
    assert m2.score(c, a) == 0
    assert m3.score(c, a) == 0

    # self-scoring
    assert m0.score(a, a) == 2
    assert m1.score(a, a) == 3
    assert m2.score(a, a) == 3
    assert m3.score(a, a) == 5

    assert m0.score(b, b) == 2
    assert m1.score(b, b) == 4
    assert m2.score(b, b) == 4
    assert m3.score(b, b) == 10

    assert m0.score(c, c) == 1
    assert m1.score(c, c) == 1
    assert mm.set_matching[int, '<-', 'none'](...).score(c, c) == 1
    assert mm.set_matching[int, '~', 'none'](...).score(c, c) == 1

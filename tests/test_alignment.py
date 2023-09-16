from unimetric.alignment import AlignmentConstraint, solve_alignment
from unimetric.metric import DiscreteMetric


def test_alignment():
    a = [1, 2, 2]
    b = [1, 1, 1, 2]

    g = DiscreteMetric(int).gram_matrix(a, b)

    assert solve_alignment(g, AlignmentConstraint.ONE_TO_ONE) == 2
    assert solve_alignment(g, AlignmentConstraint.MANY_TO_ONE) == 3
    assert solve_alignment(g, AlignmentConstraint.ONE_TO_MANY) == 4
    assert solve_alignment(g, AlignmentConstraint.MANY_TO_MANY) == 5

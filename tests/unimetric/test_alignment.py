from unimetric.alignment import solve_alignment, AlignmentConstraint
from unimetric.metric import DiscreteMetric


def test_solve_alignment():
    a = [1, 2, 2]
    b = [1, 1, 1, 2]

    g = DiscreteMetric(int).gram_matrix(a, b)

    assert solve_alignment(g, AlignmentConstraint.OneToOne) == 2
    assert solve_alignment(g, AlignmentConstraint.ManyToOne) == 3
    assert solve_alignment(g, AlignmentConstraint.OneToMany) == 4
    assert solve_alignment(g, AlignmentConstraint.ManyToMany) == 5

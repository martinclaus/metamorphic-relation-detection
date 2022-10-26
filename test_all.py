# Unit test of mereldet
import numpy as np
from mereldet import calculate_cost, MRCandidate


def test_calculate_cost():
    # training data
    input = np.random.rand(1000, 2)
    # function under test
    fun = np.prod
    # known metamorphic relations
    g_f = [MRCandidate.from_identity()]

    # this is a metamorphic relation
    g_guess = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.5]]))

    assert calculate_cost(fun, list(input), g_guess, g_f) == 0.0

    # this is not a metamorphic relation
    g_guess_2 = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.51]]))

    # this is even more not a metamorphic relation
    g_guess_3 = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.6]]))

    assert calculate_cost(fun, list(input), g_guess_2, g_f) < calculate_cost(
        fun, list(input), g_guess_3, g_f
    )

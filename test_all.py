# Unit test of mereldet

import pytest
import numpy as np
from mereldet import calculate_cost


def test_calculate_cost():
    # training data
    input = np.random.rand(1000, 2)
    # function under test
    fun = np.prod
    # known metamorphic relations
    g_f = [lambda x: x]

    # this is a metamorphic relation
    g_guess = lambda x: x * np.array([2.0, 0.5])

    assert calculate_cost(fun, list(input), g_guess, g_f) == 0.0

    # this is not a metamorphic relation
    g_guess_2 = lambda x: x * np.array([2.0, 0.4])

    # this is even more not a metamorphic relation
    g_guess_3 = lambda x: x * np.array([2.0, 0.3])

    assert calculate_cost(fun, list(input), g_guess_2, g_f) < calculate_cost(
        fun, list(input), g_guess_3, g_f
    )

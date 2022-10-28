# Unit test of mereldet
import numpy as np
from mereldet import function_under_test, calculate_cost, MRCandidate, FuncUnderTest


def test_eval_mr_for_single_input():
    input = np.random.rand(2)
    assert (MRCandidate.from_identity(size=2)(input) == input).all()


def test_mr_eval_broadcasting_correctly():
    input = np.random.rand(10, 2)
    assert (MRCandidate.from_identity(size=2)(input) == input).all()


def test_cost_of_mr_evals_to_zero():
    # training data
    input = np.random.rand(10, 2)
    # function under test
    fun: FuncUnderTest = function_under_test["prod"]  # type: ignore
    # known metamorphic relations
    g_f = [MRCandidate.from_identity()]

    # this is a metamorphic relation
    g_guess = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.5]]))

    assert calculate_cost(fun, input, g_guess, g_f) == 0.0


def test_calculate_cost():
    # training data
    input = np.random.rand(10, 2)
    # function under test
    fun = function_under_test["prod"]
    # known metamorphic relations
    g_f = [MRCandidate.from_identity()]

    # this is a metamorphic relation
    g_guess = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.5]]))

    assert calculate_cost(fun, input, g_guess, g_f) == 0.0

    # this is not a metamorphic relation
    g_guess_2 = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.51]]))

    # this is even more not a metamorphic relation
    g_guess_3 = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.6]]))

    assert calculate_cost(fun, input, g_guess_2, g_f) < calculate_cost(
        fun, input, g_guess_3, g_f
    )

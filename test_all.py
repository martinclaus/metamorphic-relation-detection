# Unit test of mereldet
import numpy as np
from mereldet import (
    Optimizer,
    function_under_test,
    MRCandidate,
    CostFunction,
)


def test_eval_mr_for_single_input():
    input = np.random.rand(2)
    assert (MRCandidate.from_identity(size=2)(input) == input).all()


def test_mr_eval_broadcasting_correctly():
    input = np.random.rand(10, 2)
    assert (MRCandidate.from_identity(size=2)(input) == input).all()


def test_cost_of_mr_evals_to_zero():
    # training data
    input = np.random.rand(10, 2)
    cost_function = CostFunction(
        function_under_test["prod"], [MRCandidate.from_identity(size=2)]  # type: ignore
    )

    # this is a metamorphic relation
    g_guess = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.5]]))

    assert cost_function(input, g_guess) == 0.0


def test_calculate_cost():
    # training data
    input = np.random.rand(10, 2)
    cost_function = CostFunction(
        function_under_test["prod"], [MRCandidate.from_identity(size=2)]  # type: ignore
    )

    # this is a metamorphic relation
    g_guess = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.5]]))

    assert cost_function(input, g_guess) == 0.0

    # this is not a metamorphic relation
    g_guess_2 = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.51]]))

    # this is even more not a metamorphic relation
    g_guess_3 = MRCandidate.from_identity().set_scale(np.array([[2.0, 0], [0, 0.6]]))

    assert cost_function(input, g_guess_2) < cost_function(input, g_guess_3)


def test_optimizer_mutates_guess():
    # training data
    input = np.random.rand(10, 2)
    # cost function
    cost_function = CostFunction(
        function_under_test["prod"], [MRCandidate.from_identity(size=2)]  # type: ignore
    )

    optimizer = Optimizer(cost_function, input)
    mr = optimizer.create_new_candidate()
    mod_mr = optimizer.mutate(mr)
    assert (mr.scale != mod_mr.scale).any()
    assert (mr.bias != mod_mr.bias).any()

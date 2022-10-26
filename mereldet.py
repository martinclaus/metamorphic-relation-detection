"""MEtamorphic RELation DETection.

This software is a POC to detect metamorphic relations of the form

f(g(x)) = f(x)

where f is a function under test accepting numerical input only and g
is a affine transformation.

See https://doi.org/10.1109/MET52542.2021.00014
"""

import numpy as np
import math
from typing import Any, Callable

# Type aliases
FuncUnderTest = Callable[[np.ndarray], float]
MRType = Callable[[np.ndarray], np.ndarray]

# Small number to prevent devision by zero in cost function
EPS = 1e-10

# Test functions for experimentation
# Note that `sum` has one non-trivial metamorphic relation with significant
# entries only for the bias and `prod` has one non-trivial metamorphic relation
# with significant entries only for the scaling part of the affine transformation.
function_under_test = dict(
    sum=np.sum,
    prod=np.prod,
)


def calculate_cost(
    fun: FuncUnderTest,
    input: list[np.ndarray],
    morph_relation_guess: MRType,
    morph_relations: list[MRType],
) -> float:
    """Evaluate cost function.

    The cost function given by equation (7) of https://doi.org/10.1109/MET52542.2021.00014

    Parameters:
    -----------
    fun: FuncUnderTest
        Funtion under test. Here, we are limiting ourself to functions accepting a single numpy
        array of floats as argument which return a float.
    input: list[np.ndarray]
        Set of input data to estimate the cost from.
    morph_relation_guess: MRType
        Metamorphic relation candidate
    morph_relations: list[MRType]
        List of already identified metmorphic relations.

    Returns:
    --------
    float: Value of the cost function
    """
    cost = sum(
        map(
            lambda x: (
                _nominator(x, fun, morph_relation_guess)
                / _denominator(x, morph_relation_guess, morph_relations)
            ),
            input,
        )
    )
    return cost


def _denominator(
    x: np.ndarray, morph_relation_guess: MRType, morph_relations: list[MRType]
) -> float:
    """Calculate the denominator of the cost function for a single input."""
    return EPS + math.prod(
        map(lambda g: ((morph_relation_guess(x) - g(x)) ** 2).sum(), morph_relations)
    )


def _nominator(
    x: np.ndarray, fun: FuncUnderTest, morph_relation_guess: MRType
) -> float:
    """Calculate the nominator of the cost function for a single input."""
    return np.sqrt(((fun(morph_relation_guess(x)) - fun(x)) ** 2))

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
    fun: Callable[[np.ndarray], float],
    input: list[np.ndarray],
    morph_relation_guess: Any,
    morph_relations: list[Any],
) -> float:
    """Evaluate cost function.

    The cost function given by equation (7) of https://doi.org/10.1109/MET52542.2021.00014

    Parameters:
    -----------
    fun: Callable[[np.ndarray], float]
        Funtion under test. Here, we are limiting ourself to functions accepting a single numpy
        array of floats as argument which return a float.
    input: list[np.ndarray]
        Set of input data to estimate the cost from.
    morph_relations: list[Any]
        List of already identified metmorphic relations.

    Returns:
    --------
    float: Value of the cost function
    """
    nominator = map(
        lambda x: np.sqrt(((fun(morph_relation_guess(x)) - fun(x)) ** 2)), input
    )
    denominator = map(
        lambda x: EPS
        + math.prod(
            map(
                lambda g: ((morph_relation_guess(x) - g(x)) ** 2).sum(), morph_relations
            )
        ),
        input,
    )
    cost = sum(map(lambda x, y: x / y, nominator, denominator))
    return cost

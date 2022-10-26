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

# Small number to prevent devision by zero in cost function
EPS = 1e-10

# mutation scale
MUT_SCALE = 1e-1

# Test functions for experimentation
# Note that `sum` has one non-trivial metamorphic relation with significant
# entries only for the bias and `prod` has one non-trivial metamorphic relation
# with significant entries only for the scaling part of the affine transformation.
function_under_test = dict(
    sum=np.sum,
    prod=np.prod,
)


class MRCandidate:
    """Candidate for a metamorphic relation."""

    __slots__ = ["scale", "bias"]

    def __init__(self, scale: np.ndarray, bias: np.ndarray):
        """Create a metamorphic relation candidate."""
        assert scale.ndim == 2
        assert bias.ndim == 1
        assert scale.shape == (len(bias), len(bias))
        self.scale = scale
        self.bias = bias

    @classmethod
    def from_identity(cls, size: int = 2) -> "MRCandidate":
        """Create a metamorphic relation candidate from the identity map."""
        scale = np.eye(size)
        bias = np.zeros(size)
        return cls(scale, bias)

    def new_guess(self) -> "MRCandidate":
        """Return a mutated version of this candidate."""
        scale, bais = self.scale, self.bias
        scale = self.scale + MUT_SCALE * (np.random.rand(*self.scale.shape) - 1)
        bias = self.bias + MUT_SCALE * (np.random.rand(*self.bias.shape) - 1)
        return MRCandidate(scale, bias)

    def set_bias(self, bias: np.ndarray) -> "MRCandidate":
        self.bias = bias
        return self

    def set_scale(self, scale: np.ndarray) -> "MRCandidate":
        self.scale = scale
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale.dot(x) + self.bias


def calculate_cost(
    fun: FuncUnderTest,
    input: list[np.ndarray],
    morph_relation_guess: MRCandidate,
    morph_relations: list[MRCandidate],
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
    x: np.ndarray, morph_relation_guess: MRCandidate, morph_relations: list[MRCandidate]
) -> float:
    """Calculate the denominator of the cost function for a single input."""
    return EPS + math.prod(
        map(lambda g: ((morph_relation_guess(x) - g(x)) ** 2).sum(), morph_relations)
    )


def _nominator(
    x: np.ndarray, fun: FuncUnderTest, morph_relation_guess: MRCandidate
) -> float:
    """Calculate the nominator of the cost function for a single input."""
    return np.sqrt(((fun(morph_relation_guess(x)) - fun(x)) ** 2))

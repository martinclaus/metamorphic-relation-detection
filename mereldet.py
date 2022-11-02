"""MEtamorphic RELation DETection.

This software is a POC to detect metamorphic relations of the form

f(g(x)) = f(x)

where f is a function under test accepting numerical input only and g
is a affine transformation.

See https://doi.org/10.1109/MET52542.2021.00014
"""

import numpy as np
from typing import Callable, Optional
from functools import partial, reduce

# Type aliases
FuncUnderTest = Callable[[np.ndarray], float]

# Small number to prevent devision by zero in cost function
EPS = 1e-10

# mutation scale
MUT_SCALE = 1e-2

# Optimizer tolerance
OPT_TOL = 1e-6

# Test functions for experimentation
# Note that `sum` has one non-trivial metamorphic relation with significant
# entries only for the bias and `prod` has one non-trivial metamorphic relation
# with significant entries only for the scaling part of the affine transformation.
function_under_test = dict(sum=partial(np.sum, axis=-1), prod=partial(np.prod, axis=-1))


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

    def set_bias(self, bias: np.ndarray) -> "MRCandidate":
        self.bias = bias
        return self

    def set_scale(self, scale: np.ndarray) -> "MRCandidate":
        self.scale = scale
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.scale.T) + self.bias


class CostFunction:
    """Base class of cost function.

    Also provides implementation as in the short paper https://doi.org/10.1109/MET52542.2021.00014
    """

    def __init__(self, f: FuncUnderTest, morph_relations: list[MRCandidate]):
        self.f = f
        self.morph_relations = morph_relations

    def __call__(self, input: np.ndarray, mr_candidate: MRCandidate) -> float:
        self.eval.__doc__
        return self.eval(input, mr_candidate)

    def eval(self, input: np.ndarray, candidate: MRCandidate) -> float:
        """Evaluate cost function.

        The cost function given by equation (7) of https://doi.org/10.1109/MET52542.2021.00014

        Parameters:
        -----------
        input: np.ndarray
            Set of input data to estimate the cost from. Each row is a record of input data.
        mr_candidate: MRCandidate
            Metamorphic relation candidate

        Returns:
        --------
        float: Value of the cost function
        """
        morph_in = candidate(input)
        # TODO: Needs to be changed if function under test returns an array
        nom = np.abs(self.f(morph_in) - self.f(input))
        denom = EPS + reduce(
            np.multiply,
            map(
                lambda g: ((morph_in - g(input)) ** 2).sum(axis=-1),
                self.morph_relations,
            ),
        )
        cost = (nom / denom).sum()
        return cost

    def distance_in_codomain(
        self, input: np.ndarray, candidate: MRCandidate
    ) -> np.ndarray:
        """Compute absolute distance of morphed input to original input in codomain."""
        return np.array(np.abs(self.f(candidate(input)) - self.f(input)))


class Optimizer:
    """Find the minimum of a given cost function for a funciton under test."""

    def __init__(
        self,
        cost_function: CostFunction,
        training_data: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cost_function = cost_function
        self.training_data = training_data
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.known_mrs = [MRCandidate.from_identity(size=self._input_size())]
        self.iteration: int = 0
        self.trace: dict[str, list] = dict(iteration=[], cost=[], mean_dist=[])

    def create_new_candidate(self) -> MRCandidate:
        """Create a new initial MR Candidate by mutating the identity map.

        The initial mutation is 10 times larger than all subsequent mutations.
        """
        new_guess = MRCandidate.from_identity(size=self._input_size())
        new_guess = self.mutate(new_guess, std=10 * MUT_SCALE)
        return new_guess

    def mutate(self, guess: MRCandidate, std: float = MUT_SCALE):
        """Create a new MR candidate by randomly mutating an existing one.

        The exiting MR guess will be randomly mutated where the mutations are
        drawn from a normal distribution located at 0 and scaled by `std`.
        """
        scale = guess.scale
        bias = guess.bias
        return MRCandidate(
            scale=scale + self.rng.normal(loc=0, scale=std, size=scale.shape),
            bias=bias + self.rng.normal(loc=0, scale=std, size=bias.shape),
        )

    def optimize(self, mut_scale=MUT_SCALE, tol=OPT_TOL, timeout=1_000) -> MRCandidate:
        mr = self.create_new_candidate()
        cost = self.cost_function(self.training_data, mr)
        for _ in range(timeout):
            if self._is_close(tol):
                break
            self.iteration += 1
            new_mr = self.mutate(mr, std=mut_scale)
            new_cost = self.cost_function(self.training_data, new_mr)
            if new_cost < cost:
                mr = new_mr
                cost = new_cost
            self.trace["iteration"].append(self.iteration)
            self.trace["cost"].append(cost)
            self.trace["mean_dist"].append(
                self.cost_function.distance_in_codomain(self.training_data, mr).mean()
            )

        return mr

    def _input_size(self) -> int:
        return self.training_data.shape[-1]

    def _is_close(self, tol) -> bool:
        return False

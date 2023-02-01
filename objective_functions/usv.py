"""Module with unconstrained scalar valued objective functions
"""
import numpy as np
import math


def check_dim(dim, minimum=1):
    assert (type(
        dim) == int and dim >= minimum), f"Dimension should be an integer and not less than " \
                                         f"{minimum} for this function (got {dim})."


class Griewank:

    def __init__(self, dim):
        check_dim(dim)

    def __call__(self, x, *args, **kwargs):
        return 1 + np.sum(np.square(x)) / 4000. - np.prod(np.cos(x / np.arange(x.shape[-1] + 1)))

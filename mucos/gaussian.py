import mrich
import numpy as np
from itertools import combinations


def alpha(standard_deviation: float) -> float:
    """Calculate alpha parameter based on standard deviation:
    alpha = standard_deviation^2 / 2"""
    return 0.5 * standard_deviation**2.0


def spherical_gaussian_volume(standard_deviation) -> float:
    """Calculate the volume of a 3D gaussian function"""
    return (0.5 * np.pi / alpha(standard_deviation)) ** 1.5


def spherical_gaussian_overlap(
    standard_deviation_1: float, standard_deviation_2: float, distance: float
) -> float:
    """Calculate the intersectional volume between two 3D gaussian functions"""
    alpha_1 = alpha(standard_deviation_1)
    alpha_2 = alpha(standard_deviation_2)
    c = 0.5 * np.pi / (alpha_1 + alpha_2)
    return c * np.exp(-0.5 * (alpha_1 * alpha_2) ** 0.5 * distance**2)


def spherical_gaussian_union_volume(
    centres: list[list[float]],
    standard_deviations: list[float],
    debug: bool = True,
) -> float:

    union_volume = sum(spherical_gaussian_volume(sd) for sd in standard_deviations)

    mrich.debug("union_volume", union_volume)

    # subtract 2nd-order intersections
    volume = union_volume
    for (c1, s1), (c2, s2) in combinations(zip(centres, standard_deviations), 2):
        dist = np.linalg.norm(np.array(c1) - np.array(c2))
        volume -= spherical_gaussian_overlap(s1, s2, dist)

    return volume

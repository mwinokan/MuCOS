import numpy as np


def alpha(standard_deviation: float) -> float:
    """Calculate alpha parameter based on standard deviation:
    alpha = standard_deviation^2 / 2"""
    return 0.5 * standard_deviation**2.0


def spherical_gaussian_volume(standard_deviation) -> float:
    """Calculate the volume of a 3D gaussian function"""
    return (0.5 * np.pi / alpha(standard_deviation)) ** 1.5


def spherical_gaussian_overlap(
    standard_deviation_1, standard_deviation_2, distance
) -> float:
    """Calculate the intersectional volume between two 3D gaussian functions"""
    alpha_1 = alpha(standard_deviation_1)
    alpha_2 = alpha(standard_deviation_2)
    c = 0.5 * np.pi / (alpha_1 + alpha_2)
    return c * np.exp(-0.5 * (alpha_1 * alpha_2) ** 0.5 * distance**2)

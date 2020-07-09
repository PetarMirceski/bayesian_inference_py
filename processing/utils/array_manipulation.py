from typing import Tuple

import numpy as np


def _norm(vector_parameter: np.ndarray) -> float:
    """Calculate the norm of the vector (euclid distance)"""
    return float(np.sum(np.square(vector_parameter)) ** (1 / 2))


def _has_matrix_nan(matrix: np.ndarray) -> bool:
    """Check if any element in the matrix is nan"""
    return np.isnan(matrix).any()


def _has_matrix_inf(matrix: np.ndarray) -> bool:
    """Check if any element in the matrix is inf"""
    return not np.isfinite(matrix).any()


def _midpoint(series: np.ndarray) -> np.ndarray:
    """Calculate the midpoin approximation of a vector"""
    return (series[0, 1:] + series[0, :-1]) / 2


def _get_quadrants(XIpt: np.ndarray) -> np.ndarray:
    """Split the matrix into four equal quadrants.
       The resulting shape is [4, n, n]
        1|2
        ___   --> [1, 2, 3, 4]
        3|4                      """
    rows, cols = XIpt.shape
    return (
        XIpt.reshape(2, rows // 2, -1, cols // 2)
        .swapaxes(1, 2)
        .reshape(-1, rows // 2, cols // 2)
    )


def _reconstruct_from_quadrants(quadrants: np.ndarray) -> np.ndarray:
    """Reconstruct the four-way split function into a matrix again.
       The transformation is equvalent to.
                         1|2
       [1,2,3,4]    -->  ___
                         3|4"""
    return np.vstack(
        (
            np.hstack((quadrants[0], quadrants[1])),
            np.hstack((quadrants[2], quadrants[3])),
        )
    )


def _parameter_sin_cos(i_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the sin and cos from a given vector"""
    return np.sin(i_phi), np.cos(i_phi)

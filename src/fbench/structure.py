from typing import NamedTuple

import numpy as np

__all__ = (
    "CoordinateMatrices",
    "CoordinatePairs",
)


class CoordinateMatrices(NamedTuple):
    """An immutable data structure for X, Y, Z coordinate matrices."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


class CoordinatePairs(NamedTuple):
    """An immutable data structure for (x, y) pairs."""

    x: np.ndarray
    y: np.ndarray

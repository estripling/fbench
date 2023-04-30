from typing import NamedTuple

import numpy as np

__all__ = ("CoordinateMatrices",)


class CoordinateMatrices(NamedTuple):
    """An immutable data structure for X, Y, Z coordinate matrices."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

from typing import NamedTuple

import numpy as np

__all__ = (
    "CoordinateMatrices",
    "CoordinatePairs",
    "Optimum",
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


class Optimum(NamedTuple):
    """Define optimum for :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`."""

    x: np.ndarray
    fx: float

    @property
    def n(self):
        """Dimensionality of :math:`x`."""
        return len(self.x)

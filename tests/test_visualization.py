import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import toolz

import fbench
from fbench import structure


def test_create_contour_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.create_coordinates3d(fbench.sphere),
        fbench.create_contour_plot(),
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_create_coordinates3d():
    actual = fbench.create_coordinates3d(fbench.sphere, [-1, 0, 1])
    expected = structure.CoordinateMatrices(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        z=np.array([[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]),
    )
    npt.assert_almost_equal(actual, expected)

    actual = fbench.create_coordinates3d(fbench.sphere, [-1, 0, 1], [2, 3, 4])
    expected = structure.CoordinateMatrices(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        z=np.array([[5.0, 4.0, 5.0], [10.0, 9.0, 10.0], [17.0, 16.0, 17.0]]),
    )
    npt.assert_almost_equal(actual, expected)

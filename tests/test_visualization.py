import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import numpy.testing as npt
import pytest
import toolz

import fbench
from fbench import structure


@pytest.mark.parametrize(
    "method_name",
    [
        "get_kws_contour__base",
        "get_kws_contourf__base",
        "get_kws_contourf__YlOrBr",
        "get_kws_contourf__YlOrBr_r",
        "get_kws_surface__base",
        "get_kws_surface__YlOrBr",
        "get_kws_surface__YlOrBr_r",
    ],
)
def test_PlotConfig(method_name):
    actual = getattr(fbench.PlotConfig, method_name)()
    assert isinstance(actual, dict)


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


def test_create_surface_plot():
    ax = toolz.pipe(
        [-1, 0, 1],
        fbench.create_coordinates3d(fbench.sphere),
        fbench.create_surface_plot(),
    )
    plt.close()
    assert isinstance(ax, mpl_toolkits.mplot3d.Axes3D)

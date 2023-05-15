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
        "get_kws_line__base",
        "get_kws_surface__base",
        "get_kws_surface__YlOrBr",
        "get_kws_surface__YlOrBr_r",
    ],
)
def test_viz_config_enum(method_name):
    actual = getattr(fbench.visualization.VizConfig, method_name)()
    assert isinstance(actual, dict)


def test_create_contour_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.visualization.create_coordinates3d(fbench.sphere),
        fbench.visualization.create_contour_plot(),
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_create_coordinates2d():
    actual = fbench.visualization.create_coordinates2d(fbench.sphere, [-2, -1, 0, 1, 2])
    expected = fbench.structure.CoordinatePairs(
        x=np.array([-2, -1, 0, 1, 2]),
        y=np.array([4.0, 1.0, 0.0, 1.0, 4.0]),
    )
    npt.assert_almost_equal(actual, expected)


def test_create_coordinates3d():
    actual = fbench.visualization.create_coordinates3d(fbench.sphere, [-1, 0, 1])
    expected = fbench.structure.CoordinateMatrices(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        z=np.array([[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]),
    )
    npt.assert_almost_equal(actual, expected)

    actual = fbench.visualization.create_coordinates3d(
        fbench.sphere, [-1, 0, 1], [2, 3, 4]
    )
    expected = fbench.structure.CoordinateMatrices(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        z=np.array([[5.0, 4.0, 5.0], [10.0, 9.0, 10.0], [17.0, 16.0, 17.0]]),
    )
    npt.assert_almost_equal(actual, expected)


def test_create_line_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.visualization.create_coordinates2d(fbench.sphere),
        fbench.visualization.create_line_plot(),
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_create_surface_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.visualization.create_coordinates3d(fbench.sphere),
        fbench.visualization.create_surface_plot(),
    )
    plt.close()
    assert isinstance(actual, mpl_toolkits.mplot3d.Axes3D)


def test_create_discrete_cmap():
    n = 5
    color_list = fbench.visualization.create_discrete_cmap(n)
    assert len(color_list) == n
    assert all(len(color_tuple) == 4 for color_tuple in color_list)
    assert all(
        isinstance(value, float) for color_tuple in color_list for value in color_tuple
    )

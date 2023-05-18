from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import numpy.testing as npt
import pytest
import toolz

import fbench


class TestFunctionPlotter:
    @pytest.fixture
    def func(self):
        return lambda x: (x**2).sum()

    def test_init(self, func):
        bounds = [(-5, 5)]
        actual = fbench.viz.FunctionPlotter(func, bounds)
        assert isinstance(actual.func, Callable)
        assert actual.bounds == bounds

    def test_init_with_invalid_bounds(self, func):
        with pytest.raises(TypeError):
            fbench.viz.FunctionPlotter(func=func, bounds=[])

        with pytest.raises(TypeError):
            fbench.viz.FunctionPlotter(func=func, bounds=[(-5, 5)] * 3)

    def test_init_with_invalid_values_for_with_surface_and_with_contour(self, func):
        with pytest.raises(ValueError):
            fbench.viz.FunctionPlotter(
                func=func,
                bounds=[(-5, 5)] * 2,
                with_surface=False,
                with_contour=False,
            )

    def test_default_plot__1d(self, func):
        plotter = fbench.viz.FunctionPlotter(func=func, bounds=[(-5, 5)] * 1)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax3d is None

    def test_default_plot__2d(self, func):
        plotter = fbench.viz.FunctionPlotter(func=func, bounds=[(-5, 5)] * 2)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(ax3d, mpl_toolkits.mplot3d.Axes3D)

    def test_surface_plot(self, func):
        plotter = fbench.viz.FunctionPlotter(
            func=func,
            bounds=[(-5, 5)] * 2,
            with_surface=True,
            with_contour=False,
        )
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is None
        assert isinstance(ax3d, mpl_toolkits.mplot3d.Axes3D)

    def test_contour_plot(self, func):
        plotter = fbench.viz.FunctionPlotter(
            func=func,
            bounds=[(-5, 5)] * 2,
            with_surface=False,
            with_contour=True,
        )
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax3d is None

    def test_default_plot__ackley(self):
        plotter = fbench.viz.FunctionPlotter(func=fbench.ackley, bounds=[(-5, 5)] * 2)
        fig, ax, ax3d = plotter.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(ax3d, mpl_toolkits.mplot3d.Axes3D)


@pytest.mark.parametrize(
    "method_name",
    [
        "get_kws_contour__base",
        "get_kws_contourf__base",
        "get_kws_contourf__YlOrBr",
        "get_kws_contourf__YlOrBr_r",
        "get_kws_plot__base",
        "get_kws_scatter__base",
        "get_kws_surface__base",
        "get_kws_surface__YlOrBr",
        "get_kws_surface__YlOrBr_r",
    ],
)
def test_viz_config_enum(method_name):
    actual = getattr(fbench.viz.VizConfig, method_name)()
    assert isinstance(actual, dict)


def test_create_contour_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.viz.create_coordinates3d(fbench.sphere),
        fbench.viz.create_contour_plot(),
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_create_coordinates2d():
    actual = fbench.viz.create_coordinates2d(fbench.sphere, [-2, -1, 0, 1, 2])
    expected = fbench.structure.CoordinatePairs(
        x=np.array([-2, -1, 0, 1, 2]),
        y=np.array([4.0, 1.0, 0.0, 1.0, 4.0]),
    )
    npt.assert_almost_equal(actual, expected)


def test_create_coordinates3d():
    actual = fbench.viz.create_coordinates3d(fbench.sphere, [-1, 0, 1])
    expected = fbench.structure.CoordinateMatrices(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        z=np.array([[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]),
    )
    npt.assert_almost_equal(actual, expected)

    actual = fbench.viz.create_coordinates3d(fbench.sphere, [-1, 0, 1], [2, 3, 4])
    expected = fbench.structure.CoordinateMatrices(
        x=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        y=np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        z=np.array([[5.0, 4.0, 5.0], [10.0, 9.0, 10.0], [17.0, 16.0, 17.0]]),
    )
    npt.assert_almost_equal(actual, expected)


def test_create_line_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.viz.create_coordinates2d(fbench.sphere),
        fbench.viz.create_line_plot(),
    )
    plt.close()
    assert isinstance(actual, matplotlib.axes.Axes)


def test_create_surface_plot():
    actual = toolz.pipe(
        [-1, 0, 1],
        fbench.viz.create_coordinates3d(fbench.sphere),
        fbench.viz.create_surface_plot(),
    )
    plt.close()
    assert isinstance(actual, mpl_toolkits.mplot3d.Axes3D)


def test_create_discrete_cmap():
    n = 5
    color_list = fbench.viz.create_discrete_cmap(n)
    assert len(color_list) == n
    assert all(len(color_tuple) == 4 for color_tuple in color_list)
    assert all(
        isinstance(value, float) for color_tuple in color_list for value in color_tuple
    )


def test_get_1d_plotter():
    function_plotters = fbench.viz.get_1d_plotter()
    for name, plotter in function_plotters.items():
        assert isinstance(name, str)
        assert isinstance(plotter, fbench.viz.FunctionPlotter)
        assert isinstance(plotter.func, Callable)


def test_get_2d_plotter():
    function_plotters = fbench.viz.get_2d_plotter()
    for name, plotter in function_plotters.items():
        assert isinstance(name, str)
        assert isinstance(plotter, fbench.viz.FunctionPlotter)
        assert isinstance(plotter.func, Callable)


def test_plot_optima():
    func = fbench.sphere
    ax = toolz.pipe(
        [-1, 0, 1],
        fbench.viz.create_coordinates2d(func),
        fbench.viz.create_line_plot(),
    )

    optima = fbench.get_optima(1, func)
    ax, _ = fbench.viz.plot_optima(optima, ax=ax)

    plt.close()
    assert isinstance(ax, matplotlib.axes.Axes)

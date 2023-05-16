from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import toolz
from mpl_toolkits.axes_grid1 import make_axes_locatable

import fbench

__all__ = (
    "VizConfig",
    "FunctionPlotter",
    "create_contour_plot",
    "create_coordinates2d",
    "create_coordinates3d",
    "create_line_plot",
    "create_surface_plot",
    "create_discrete_cmap",
    "get_1d_plotter",
    "get_2d_plotter",
)


class VizConfig(Enum):
    """Visualization configurations."""

    @classmethod
    def get_kws_contour__base(cls):
        """Returns kwargs for ``contour``: base configuration."""
        return dict(
            levels=12,
            colors="dimgray",
            antialiased=True,
            linewidths=0.25,
            alpha=1.0,
            zorder=1,
        )

    @classmethod
    def get_kws_contourf__base(cls):
        """Returns kwargs for ``contourf``: base configuration."""
        return dict(
            levels=100,
            antialiased=True,
            alpha=0.61803,
            zorder=0,
        )

    @classmethod
    def get_kws_contourf__YlOrBr(cls):
        """Returns kwargs for ``contourf``: YlOrBr configuration for dark max."""
        output = dict(
            cmap=plt.get_cmap("YlOrBr"),
        )
        output.update(cls.get_kws_contourf__base())
        return output

    @classmethod
    def get_kws_contourf__YlOrBr_r(cls):
        """Returns kwargs for ``contourf``: YlOrBr_r configuration for dark min."""
        output = dict(
            cmap=plt.get_cmap("YlOrBr_r"),
        )
        output.update(cls.get_kws_contourf__base())
        return output

    @classmethod
    def get_kws_line__base(cls):
        """Returns kwargs for ``line``: base configuration."""
        return dict(
            linewidth=2,
            zorder=0,
        )

    @classmethod
    def get_kws_surface__base(cls):
        """Returns kwargs for ``plot_surface``: base configuration."""
        return dict(
            rstride=1,
            cstride=1,
            edgecolors="dimgray",
            antialiased=True,
            linewidth=0.1,
            alpha=0.61803,
            zorder=0,
        )

    @classmethod
    def get_kws_surface__YlOrBr(cls):
        """Returns kwargs for ``plot_surface``: YlOrBr configuration for dark max."""
        output = dict(
            cmap=plt.get_cmap("YlOrBr"),
        )
        output.update(cls.get_kws_surface__base())
        return output

    @classmethod
    def get_kws_surface__YlOrBr_r(cls):
        """Returns kwargs for ``plot_surface``: YlOrBr_r configuration for dark min."""
        output = dict(
            cmap=plt.get_cmap("YlOrBr_r"),
        )
        output.update(cls.get_kws_surface__base())
        return output


class FunctionPlotter:
    """Plot a scalar-valued function with an 1-vector or 2-vector input.

    Parameters
    ----------
    func : callable
        The function to plot.
    bounds : sequence
        A sequence of ``(min, max)`` pairs for each element of the vector.
    with_surface : bool, default=True
        Specify if the function surface plot should be generated.
    with_contour : bool, default=True
        Specify if the contour plot should be generated.
    n_grid_points : int, default=101
        Specify the number of grid points on one axis.
        Ignored if ``x_coord`` or ``y_coord`` is specified.
    x_coord : sequence, default=None
        Specify coordinates on the x-axis.
    x_coord : sequence, default=None
        Specify coordinates on the y-axis.
    kws_surface : dict of keyword arguments, default=None
        The kwargs are passed to ``mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface``.
        By default, using configuration: ``VizConfig.get_kws_surface__YlOrBr_r()``.
        Optionally specify a dict of keyword arguments to update configurations.
    kws_contourf : dict of keyword arguments, default=None
        The kwargs are passed to ``mpl_toolkits.mplot3d.axes3d.Axes3D.contourf``.
        By default, using configuration: ``VizConfig.get_kws_contourf__YlOrBr_r()``.
        Optionally specify a dict of keyword arguments to update configurations.
    kws_contour : dict of keyword arguments, default=None
        The kwargs are passed to ``matplotlib.axes.Axes.contour``.
        By default, using configuration: ``VizConfig.get_kws_contour__base()``.
        Optionally specify a dict of keyword arguments to update configurations.
    kws_line : dict of keyword arguments, default=None
        The kwargs are passed to ``matplotlib.axes.Axes.plot``.
        By default, using configuration: ``VizConfig.get_kws_line__base()``.
        Optionally specify a dict of keyword arguments to update configurations.

    Notes
    -----
    - Function is curried.
    - Examples are shown in the
      `Overview of fBench functions <https://fbench.readthedocs.io/en/stable/fBench-functions.html>`_.

    Examples
    --------
    >>> import fbench
    >>> fbench.viz.FunctionPlotter(func=fbench.sphere, bounds=[(-5, 5)])
    FunctionPlotter(func=sphere, bounds=[(-5, 5)])
    """  # noqa: E501

    def __init__(
        self,
        func,
        bounds,
        with_surface=True,
        with_contour=True,
        n_grid_points=101,
        x_coord=None,
        y_coord=None,
        kws_surface=None,
        kws_contourf=None,
        kws_contour=None,
        kws_line=None,
    ):
        self._func = func
        self._bounds = bounds
        self._with_surface = with_surface
        self._with_contour = with_contour
        self._n_grid_points = n_grid_points
        self._x_coord = x_coord
        self._y_coord = y_coord
        self._kws_contourf = kws_contourf
        self._kws_contour = kws_contour
        self._kws_surface = kws_surface
        self._kws_line = kws_line

        self._size = len(bounds)
        self._coord = None

        if self._size not in (1, 2):
            raise TypeError("the total number of bounds must be either 1 or 2")

        if self._size == 2 and with_surface is False and with_contour is False:
            raise ValueError("`with_surface` and `with_contour` cannot both be False")

    def __repr__(self):
        return f"{type(self).__name__}(func={self.func.__name__}, bounds={self.bounds})"

    @property
    def func(self):
        """The function to plot."""
        return self._func

    @property
    def bounds(self):
        """Bounds to use for the plot."""
        return self._bounds

    def plot(self, fig=None, ax=None, ax3d=None):
        """Generate the plot.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, default=None
            Optionally supply a ``Figure`` object.
            If None, the current ``Figure`` object is retrieved.
        ax : matplotlib.axes.Axes, default=None
            Optionally supply an ``Axes`` object.
            If None, the current ``Axes`` object is retrieved.
        ax3d : mpl_toolkits.mplot3d.axes3d.Axes3D, default=None
            Optionally supply an ``Axes3D`` object.
            If None, the current ``Axes3D`` object is retrieved.

        Returns
        -------
        tuple[fig, ax, ax3d]
            The created plot objects.

        Notes
        -----
        When creating both a surface and contour plot and either
        ``ax`` or ``ax3d`` is specified, it is best to also supply ``fig``.
        To this end, it might be easier to only supply a ``fig`` object.
        """
        self._set_coord_attr()

        if self._size == 1:
            fig, ax, ax3d = self._plot_line(fig, ax, ax3d)

        else:
            if self._with_surface and self._with_contour:
                fig, ax, ax3d = self._plot_surface_and_contour(fig, ax, ax3d)

            if self._with_surface and not self._with_contour:
                fig, ax, ax3d = self._plot_surface(fig, ax, ax3d)

            if not self._with_surface and self._with_contour:
                fig, ax, ax3d = self._plot_contour(fig, ax, ax3d)

        return fig, ax, ax3d

    def _plot_surface_and_contour(self, fig, ax, ax3d):
        fig = fig or plt.figure(figsize=plt.figaspect(0.4))

        ax3d = ax3d or fig.add_subplot(1, 2, 1, projection="3d")
        ax3d = create_surface_plot(
            self._coord,
            kws_surface=self._kws_surface,
            kws_contourf=self._kws_contourf,
            ax=ax3d,
        )

        ax = ax or fig.add_subplot(1, 2, 2)
        ax = create_contour_plot(
            self._coord,
            kws_contourf=self._kws_contourf,
            kws_contour=self._kws_contour,
            ax=ax,
        )
        ax.axis("scaled")

        return fig, ax, ax3d

    def _plot_surface(self, fig, ax, ax3d):
        fig = fig or plt.gcf()

        ax3d = ax3d or fig.add_subplot(1, 1, 1, projection="3d")
        ax3d = create_surface_plot(self._coord, ax=ax3d)

        ax = None

        return fig, ax, ax3d

    def _plot_contour(self, fig, ax, ax3d):
        fig = fig or plt.gcf()

        ax3d = None

        ax = ax or fig.add_subplot(1, 1, 1)
        ax = create_contour_plot(
            self._coord,
            kws_contourf=self._kws_contourf,
            kws_contour=self._kws_contour,
            ax=ax,
        )
        ax.axis("scaled")

        return fig, ax, ax3d

    def _plot_line(self, fig, ax, ax3d):
        fig = fig or plt.gcf()

        ax3d = None

        ax = ax or fig.add_subplot(1, 1, 1)
        ax = create_line_plot(
            self._coord,
            kws_line=self._kws_line,
            ax=ax,
        )

        return fig, ax, ax3d

    def _set_coord_attr(self):
        """Private setter for coordinate attribute."""
        if self._coord is None:
            if self._size == 1:
                (x_bounds,) = self._bounds
                x_coord = self._x_coord or np.linspace(
                    min(x_bounds), max(x_bounds), self._n_grid_points
                )
                self._coord = create_coordinates2d(self._func, x_coord)

            else:
                x_bounds, y_bounds = self._bounds
                x_coord = self._x_coord or np.linspace(
                    min(x_bounds), max(x_bounds), self._n_grid_points
                )
                y_coord = self._y_coord or np.linspace(
                    min(y_bounds), max(y_bounds), self._n_grid_points
                )
                self._coord = create_coordinates3d(self._func, x_coord, y_coord)


@toolz.curry
def create_contour_plot(coord, /, *, kws_contourf=None, kws_contour=None, ax=None):
    """Create a contour plot from X, Y, Z coordinate matrices.

    Parameters
    ----------
    coord : CoordinateMatrices
        The X, Y, Z coordinate matrices to plot.
    kws_contourf : dict of keyword arguments, default=None
        The kwargs are passed to ``matplotlib.axes.Axes.contourf``.
        By default, using configuration: ``VizConfig.get_kws_contourf__YlOrBr_r()``.
        Optionally specify a dict of keyword arguments to update configurations.
    kws_contour : dict of keyword arguments, default=None
        The kwargs are passed to ``matplotlib.axes.Axes.contour``.
        By default, using configuration: ``VizConfig.get_kws_contour__base()``.
        Optionally specify a dict of keyword arguments to update configurations.
    ax : matplotlib.axes.Axes, default=None
        Optionally supply an ``Axes`` object.
        If None, the current ``Axes`` object is retrieved.

    Returns
    -------
    matplotlib.axes.Axes
        The ``Axes`` object with filled contours and superimposed contour lines.

    Notes
    -----
    - Function is curried.
    - Examples are shown in the
      `Overview of fBench functions <https://fbench.readthedocs.io/en/stable/fBench-functions.html>`_.
    """  # noqa: E501
    ax = ax or plt.gca()

    settings_contourf = VizConfig.get_kws_contourf__YlOrBr_r()
    settings_contourf.update(kws_contourf or dict())
    contour_plot = ax.contourf(coord.x, coord.y, coord.z, **settings_contourf)

    settings_contour = VizConfig.get_kws_contour__base()
    settings_contour.update(kws_contour or dict())
    ax.contour(coord.x, coord.y, coord.z, **settings_contour)

    plt.colorbar(
        contour_plot,
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15),
    )

    return ax


@toolz.curry
def create_coordinates2d(func, x_coord, /):
    """Create (x, y) pairs from coordinate vector and function.

    For each value of :math:`x`, compute function value :math:`y = f(x)`.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        A scalar-valued function that takes an 1-vector as input.
    x_coord : array_like
        An one-dimensional array for the x-coordinates of the grid.

    Returns
    -------
    CoordinatePairs
        The (x, y) coordinate pairs.

    Notes
    -----
    Function is curried.

    Examples
    --------
    >>> import fbench
    >>> fbench.viz.create_coordinates2d(fbench.sphere, [-2, -1, 0, 1, 2])
    CoordinatePairs(x=array([-2, -1,  0,  1,  2]), y=array([4., 1., 0., 1., 4.]))
    """
    x = fbench.check_vector(x_coord, n_min=2)
    y = np.apply_along_axis(func1d=func, axis=1, arr=np.c_[x.ravel()])
    return fbench.structure.CoordinatePairs(x, y)


@toolz.curry
def create_coordinates3d(func, x_coord, y_coord=None, /):
    """Create X, Y, Z coordinate matrices from coordinate vectors and function.

    First, a meshgrid of (x, y)-coordinates is constructed from the coordinate vectors.
    Then, the z-coordinate for each (x, y)-point is computed using the function.

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        A scalar-valued function that takes a two-dimensional, real vector as input.
    x_coord : array_like
        An one-dimensional array for the x-coordinates of the grid.
    y_coord : array_like, default=None
        An one-dimensional array for the y-coordinates of the grid.
        If None, ``y_coord`` equals ``x_coord``.

    Returns
    -------
    CoordinateMatrices
        The coordinate matrices.

    Notes
    -----
    Function is curried.

    Examples
    --------
    >>> import fbench
    >>> fbench.viz.create_coordinates3d(fbench.sphere, [-1, 0, 1])
    CoordinateMatrices(x=array([[-1,  0,  1],
           [-1,  0,  1],
           [-1,  0,  1]]), y=array([[-1, -1, -1],
           [ 0,  0,  0],
           [ 1,  1,  1]]), z=array([[2., 1., 2.],
           [1., 0., 1.],
           [2., 1., 2.]]))
    """
    x_coord = fbench.check_vector(x_coord, n_min=2)
    y_coord = x_coord if y_coord is None else fbench.check_vector(y_coord, n_min=2)
    x, y = np.meshgrid(x_coord, y_coord)
    z = np.apply_along_axis(func1d=func, axis=1, arr=np.c_[x.ravel(), y.ravel()])
    return fbench.structure.CoordinateMatrices(x, y, z.reshape(x.shape))


@toolz.curry
def create_discrete_cmap(n, /, *, name="viridis_r", lower_bound=0.05, upper_bound=0.9):
    """Create discrete values from colormap.

    Parameters
    ----------
    n : int
        Specify the number of discrete values.
    name : str, default="viridis_r"
        Specify the name of the colormap.
    lower_bound : float, default=0.05,
        Specify the lower bound of the colormap.
    upper_bound : float, default=0.9,
        Specify the upper bound of the colormap.

    Returns
    -------
    list of tuple[float, float, float, float]
        Discrete values from colormap.

    Notes
    -----
    Function is curried.

    Examples
    --------
    >>> import fbench
    >>> fbench.viz.create_discrete_cmap(2)
    [(0.876168, 0.891125, 0.09525, 1.0), (0.282623, 0.140926, 0.457517, 1.0)]
    """
    cmap = plt.get_cmap(name)
    return [cmap(i) for i in np.linspace(lower_bound, upper_bound, num=n)]


@toolz.curry
def create_line_plot(coord, /, *, kws_line=None, ax=None):
    """Create a line plot from (x, y) pairs.

    Parameters
    ----------
    coord : CoordinatePairs
        The (x, y) coordinate pairs.
    kws_line : dict of keyword arguments, default=None
        The kwargs are passed to ``matplotlib.axes.Axes.plot``.
        By default, using configuration: ``VizConfig.get_kws_line__base()``.
        Optionally specify a dict of keyword arguments to update configurations.
    ax : matplotlib.axes.Axes, default=None
        Optionally supply an ``Axes`` object.
        If None, the current ``Axes`` object is retrieved.

    Returns
    -------
    matplotlib.axes.Axes
        The ``Axes`` object.

    Notes
    -----
    - Function is curried.
    - Examples are shown in the
      `Overview of fBench functions <https://fbench.readthedocs.io/en/stable/fBench-functions.html>`_.
    """  # noqa: E501
    ax = ax or plt.gca()

    settings_line = VizConfig.get_kws_line__base()
    settings_line.update(kws_line or dict())
    ax.plot(coord.x, coord.y, **settings_line)

    return ax


@toolz.curry
def create_surface_plot(coord, /, *, kws_surface=None, kws_contourf=None, ax=None):
    """Create a surface plot from X, Y, Z coordinate matrices.

    Parameters
    ----------
    coord : CoordinateMatrices
        The X, Y, Z coordinate matrices to plot.
    kws_surface : dict of keyword arguments, default=None
        The kwargs are passed to ``mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface``.
        By default, using configuration: ``VizConfig.get_kws_surface__YlOrBr_r()``.
        Optionally specify a dict of keyword arguments to update configurations.
    kws_contourf : dict of keyword arguments, default=None
        The kwargs are passed to ``mpl_toolkits.mplot3d.axes3d.Axes3D.contourf``.
        By default, using configuration: ``VizConfig.get_kws_contourf__YlOrBr_r()``.
        Optionally specify a dict of keyword arguments to update configurations.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, default=None
        Optionally supply an ``Axes3D`` object.
        If None, the current ``Axes3D`` object is retrieved.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        The ``Axes3D`` object of the surface.

    Notes
    -----
    - Function is curried.
    - Examples are shown in the
      `Overview of fBench functions <https://fbench.readthedocs.io/en/stable/fBench-functions.html>`_.
    """  # noqa: E501
    ax = ax or plt.gcf().add_subplot(projection="3d")

    # Make background and axis panes transparent
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    settings_surface = VizConfig.get_kws_surface__YlOrBr_r()
    settings_surface.update(kws_surface or dict())
    ax.plot_surface(coord.x, coord.y, coord.z, **settings_surface)

    settings_contourf = VizConfig.get_kws_contourf__YlOrBr_r()
    settings_contourf.update(kws_contourf or dict())
    settings_contourf["zdir"] = settings_contourf.get("zdir", "z")
    settings_contourf["offset"] = settings_contourf.get("offset", 0) + coord.z.min()
    ax.contourf(coord.x, coord.y, coord.z, **settings_contourf)

    return ax


def get_1d_plotter():
    """Get FunctionPlotter instances for functions with 1-vector input.

    Returns
    -------
    dict of FunctionPlotter
        Predefined FunctionPlotter instances.
    """
    return {
        "Ackley_1D": FunctionPlotter(
            func=fbench.ackley,
            bounds=((-5, 5),),
            n_grid_points=1001,
        ),
        "Rastrigin_1D": FunctionPlotter(
            func=fbench.rastrigin,
            bounds=((-5, 5),),
            n_grid_points=1001,
        ),
    }


def get_2d_plotter():
    """Get FunctionPlotter instances for functions with 2-vector input.

    Returns
    -------
    dict of FunctionPlotter
        Predefined FunctionPlotter instances.
    """
    return {
        "Ackley_2D": FunctionPlotter(
            func=fbench.ackley,
            bounds=((-5, 5), (-5, 5)),
        ),
        "Rastrigin_2D": FunctionPlotter(
            func=fbench.rastrigin,
            bounds=((-5.12, 5.12), (-5.12, 5.12)),
        ),
        "Rosenbrock_2D": FunctionPlotter(
            func=fbench.rosenbrock,
            bounds=((-2, 2), (-2, 2)),
        ),
        "Rosenbrock_2D_log1p": FunctionPlotter(
            func=toolz.compose_left(fbench.rosenbrock, np.log1p),
            bounds=((-2, 2), (-2, 2)),
        ),
        "Sphere_2D": FunctionPlotter(
            func=fbench.sphere,
            bounds=((-2, 2), (-2, 2)),
        ),
    }
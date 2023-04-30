import matplotlib.pyplot as plt
import numpy as np
import toolz
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fbench import structure, validation

__all__ = (
    "create_coordinates3d",
    "create_contour_plot",
)


@toolz.curry
def create_contour_plot(coord, contourf_kws=None, contour_kws=None, ax=None):
    """Create a contour plot from X, Y, Z coordinate matrices.

    Parameters
    ----------
    coord : CoordinateMatrices
        The X, Y, Z coordinate matrices to plot.
    contourf_kws : dict of keyword arguments
        The kwargs are passed to ``matplotlib.axes.Axes.contourf``.
    contour_kws : dict of keyword arguments
        The kwargs are passed to ``matplotlib.axes.Axes.contour``.
    ax: matplotlib.axes.Axes, default=None
        Optionally supply an Axes object.
        If None, the current Axes object is retrieved.

    Returns
    -------
    matplotlib.axes.Axes
        An Axes object with filled contours and superimposed contour lines.

    Notes
    -----
    - Function is curried.
    - Examples are shown in the
      `Overview of fBench functions <https://fbench.readthedocs.io/en/stable/fBench-functions.html)>`_.
    """  # noqa: E501
    ax = ax or plt.gca()

    contourf_kws = {} if contourf_kws is None else contourf_kws
    contourf_default_kws = dict(
        levels=100,
        cmap=cm.YlOrBr_r,
        antialiased=True,
        alpha=0.61803,
        zorder=0,
    )
    contourf_kws.update(contourf_default_kws)
    contour_plot = ax.contourf(coord.x, coord.y, coord.z, **contourf_kws)

    contour_kws = {} if contour_kws is None else contour_kws
    contour_default_kws = dict(
        levels=12,
        colors="dimgray",
        antialiased=True,
        linewidths=0.25,
        alpha=1.0,
        zorder=1,
    )
    contour_kws.update(contour_default_kws)
    ax.contour(coord.x, coord.y, coord.z, **contour_default_kws)

    plt.colorbar(
        contour_plot,
        cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15),
    )

    return ax


@toolz.curry
def create_coordinates3d(func1d, x_coord, y_coord=None, /):
    """Create X, Y, Z coordinate matrices from coordinate vectors and function.

    First, a meshgrid of (x, y)-coordinates is constructed from the coordinate vectors.
    Then, the z-coordinate for each (x, y)-point is computed using the function.

    Parameters
    ----------
    func1d : Callable[[np.ndarray], float]
        A scalar-valued function that takes a two-dimensional, real vector as input.
    x_coord : np.ndarray
        An one-dimensional array for the x-coordinates of the grid.
    y_coord : np.ndarray, default=None
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
    >>> fbench.create_coordinates3d(fbench.sphere, [-1, 0, 1])
    CoordinateMatrices(x=array([[-1,  0,  1],
           [-1,  0,  1],
           [-1,  0,  1]]), y=array([[-1, -1, -1],
           [ 0,  0,  0],
           [ 1,  1,  1]]), z=array([[2., 1., 2.],
           [1., 0., 1.],
           [2., 1., 2.]]))
    """
    x_coord = validation.check_vector(x_coord, min_elements=2)
    y_coord = validation.check_vector(y_coord, min_elements=2) if y_coord else x_coord
    x, y = np.meshgrid(x_coord, y_coord)
    z = np.apply_along_axis(func1d=func1d, axis=1, arr=np.c_[x.ravel(), y.ravel()])
    return structure.CoordinateMatrices(x, y, z.reshape(x.shape))

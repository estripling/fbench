import numpy as np
import toolz

from fbench import structure, validation

__all__ = ("create_coordinates3d",)


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

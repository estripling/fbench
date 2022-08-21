import numpy as np


def ackley(x):
    """Ackley function.

    .. math::

        f(\\mathbf{x}) =
        -20 \\exp \\left(
            -0.2 \\sqrt{ \\frac{1}{n} \\sum_{i=1}^n x_i^2 }
        \\right)
        - \\exp \\left( \\frac{1}{n} \\sum_{i=1}^n \\cos(2 \\pi x_i) \\right)
        + 20
        + e

    Parameters
    ----------
    x : array_like
        A real-valued vector to evaluate.

    Returns
    -------
    float
        Function value at x.

    References
    ----------
    .. [1] "Test functions for optimization", Wikipedia,
           `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

    Examples
    --------
    >>> round(ackley([0, 0]), 4)
    0.0

    >>> round(ackley([1, 1]), 4)
    3.6254
    """
    x = np.asarray(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt((x**2).mean()))
        - np.exp((np.cos(2 * np.pi * x)).sum() / len(x))
        + 20
        + np.e
    )


def sphere(x):
    """Sphere function.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^n x_i^2

    Parameters
    ----------
    x : array_like
        A real-valued vector to evaluate.

    Returns
    -------
    float
        Function value at x.

    References
    ----------
    .. [1] "Test functions for optimization", Wikipedia,
           `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

    Examples
    --------
    >>> sphere([0, 0])
    0.0

    >>> sphere([1, 1])
    2.0
    """
    return float((np.asarray(x) ** 2).sum())

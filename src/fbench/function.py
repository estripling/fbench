import numpy as np

from fbench import validation

__all__ = (
    "ackley",
    "rastrigin",
    "rosenbrock",
    "sphere",
)


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
    x = validation.check_vector(x, 1)
    return float(
        -20 * np.exp(-0.2 * np.sqrt((x**2).mean()))
        - np.exp((np.cos(2 * np.pi * x)).sum() / len(x))
        + 20
        + np.e
    )


def rastrigin(x):
    """Rastrigin function.

    .. math::

        f(\\mathbf{x}) =
        10n + \\sum_{i=1}^n \\left( x_i^2 - 10 \\cos(2 \\pi x_i) \\right)

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
    >>> round(rastrigin([0, 0]), 4)
    0.0

    >>> round(rastrigin([1, 2]), 4)
    5.0

    >>> round(rastrigin([4.5, 4.5]), 4)
    80.5
    """
    x = validation.check_vector(x, 1)
    return float(10 * len(x) + (x**2 - 10 * np.cos(2 * np.pi * x)).sum())


def rosenbrock(x):
    """Rosenbrock function.

    .. math::

        f(\\mathbf{x}) =
        \\sum_{i=1}^{n-1} \\left(
            100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        \\right)

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
    >>> round(rosenbrock([0, 0]), 4)
    1.0

    >>> round(rosenbrock([1, 1]), 4)
    0.0

    >>> round(rosenbrock([1, 1, 1]), 4)
    0.0

    >>> round(rosenbrock([3, 3]), 4)
    3604.0
    """
    x = validation.check_vector(x, 2)
    return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


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
    x = validation.check_vector(x, 1)
    return float((x**2).sum())

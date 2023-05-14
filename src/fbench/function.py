import numpy as np

import fbench

__all__ = (
    "ackley",
    "rastrigin",
    "rosenbrock",
    "sphere",
)


def ackley(x, /):
    """Ackley function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.

    .. math::

        f(\\mathbf{x}) =
        -20 \\exp \\left(
            -0.2 \\sqrt{ \\frac{1}{n} \\sum_{i=1}^{n} x_i^2 }
        \\right)
        - \\exp \\left( \\frac{1}{n} \\sum_{i=1}^{n} \\cos(2 \\pi x_i) \\right)
        + 20
        + e

    Parameters
    ----------
    x : array_like
        The :math:`n`-vector.

    Returns
    -------
    float
        Function value at :math:`\\mathbf{x}`.

    References
    ----------
    .. [1] "Test functions for optimization", Wikipedia,
           `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

    Examples
    --------
    >>> import fbench
    >>> round(fbench.ackley([0, 0]), 4)
    0.0

    >>> round(fbench.ackley([1, 2]), 4)
    5.4221

    >>> round(fbench.ackley([1, 2, 3]), 4)
    7.0165
    """
    x = fbench.check_vector(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt((x**2).mean()))
        - np.exp((np.cos(2 * np.pi * x)).sum() / len(x))
        + 20
        + np.e
    )


def rastrigin(x, /):
    """Rastrigin function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.

    .. math::

        f(\\mathbf{x}) =
        10n + \\sum_{i=1}^{n} \\left( x_i^2 - 10 \\cos(2 \\pi x_i) \\right)

    Parameters
    ----------
    x : array_like
        The :math:`n`-vector.

    Returns
    -------
    float
        Function value at :math:`\\mathbf{x}`.

    References
    ----------
    .. [1] "Test functions for optimization", Wikipedia,
           `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

    Examples
    --------
    >>> import fbench
    >>> round(fbench.rastrigin([0, 0]), 4)
    0.0

    >>> round(fbench.rastrigin([1, 2]), 4)
    5.0

    >>> round(fbench.rastrigin([4.5, 4.5]), 4)
    80.5

    >>> round(fbench.rastrigin([1, 2, 3]), 4)
    14.0
    """
    x = fbench.check_vector(x)
    return float(10 * len(x) + (x**2 - 10 * np.cos(2 * np.pi * x)).sum())


def rosenbrock(x, /):
    """Rosenbrock function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.

    .. math::

        f(\\mathbf{x}) =
        \\sum_{i=1}^{n-1} \\left(
            100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        \\right)

    Parameters
    ----------
    x : array_like
        The :math:`n`-vector.

    Returns
    -------
    float
        Function value at :math:`\\mathbf{x}`.

    References
    ----------
    .. [1] "Test functions for optimization", Wikipedia,
           `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

    Examples
    --------
    >>> import fbench
    >>> round(fbench.rosenbrock([0, 0]), 4)
    1.0

    >>> round(fbench.rosenbrock([1, 1]), 4)
    0.0

    >>> round(fbench.rosenbrock([1, 1, 1]), 4)
    0.0

    >>> round(fbench.rosenbrock([1, 2, 3]), 4)
    201.0

    >>> round(fbench.rosenbrock([3, 3]), 4)
    3604.0
    """
    x = fbench.check_vector(x, n_min=2)
    return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


def sphere(x, /):
    """Sphere function.

    A function :math:`f\\colon \\mathbb{R}^{n} \\rightarrow \\mathbb{R}`
    that takes an :math:`n`-vector as input and returns a scalar value.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{n} x_i^2

    Parameters
    ----------
    x : array_like
        The :math:`n`-vector.

    Returns
    -------
    float
        Function value at :math:`\\mathbf{x}`.

    References
    ----------
    .. [1] "Test functions for optimization", Wikipedia,
           `<https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

    Examples
    --------
    >>> import fbench
    >>> fbench.sphere([0, 0])
    0.0

    >>> fbench.sphere([1, 1])
    2.0

    >>> fbench.sphere([1, 2, 3])
    14.0
    """
    x = fbench.check_vector(x)
    return float((x**2).sum())

import numpy as np
import toolz

import fbench

__all__ = (
    "ackley",
    "get_optima",
    "peaks",
    "rastrigin",
    "rosenbrock",
    "sinc",
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


@toolz.curry
def get_optima(n, /, func):
    """Retrieve optima for defined functions.

    Parameters
    ----------
    n : int
        Specify the number of dimensions :math:`n`.
    func : callable
        A fBench function to retrieve its optima.
        None is returned if no optima is defined.

    Returns
    -------
    Optional[list[Optimum]]]
        Optima with specified dimension for fBench function if defined.

    Notes
    -----
    - Function is curried.
    - Optima are defined for the following functions:
        - ackley
        - peaks
        - rastrigin
        - rosenbrock
        - sphere

    Examples
    --------
    >>> import fbench
    >>> optima = fbench.get_optima(5, fbench.sphere)
    >>> optima
    [Optimum(x=array([0, 0, 0, 0, 0]), fx=0)]
    >>> optimum = optima[0]
    >>> optimum.n
    5
    """
    optima = {
        ackley: [fbench.structure.Optimum(fbench.check_vector([0] * n), 0)],
        peaks: [
            fbench.structure.Optimum(
                fbench.check_vector([0.228279999979237, -1.625531071954464]),
                -6.551133332622496,
            )
        ],
        rastrigin: [fbench.structure.Optimum(fbench.check_vector([0] * n), 0)],
        rosenbrock: [fbench.structure.Optimum(fbench.check_vector([1] * n), 0)],
        sinc: [
            fbench.structure.Optimum(
                fbench.check_vector([-4.493409471849579]),
                -0.217233628211222,
            ),
            fbench.structure.Optimum(
                fbench.check_vector([4.493409471849579]),
                -0.217233628211222,
            ),
        ],
        sphere: [fbench.structure.Optimum(fbench.check_vector([0] * n), 0)],
    }
    return optima.get(func, None)


def peaks(x, /):
    """Peaks function.

    A function :math:`f\\colon \\mathbb{R}^{2} \\rightarrow \\mathbb{R}`
    that takes an :math:`2`-vector as input and returns a scalar value.

    .. math::

       f(\\mathbf{x}) =
       3 (1 - x_{1})^{2}
         \\exp\\left( - x_{1}^{2} - (x_{2} + 1)^{2} \\right)
       - 10 \\left( \\frac{x_{1}}{5} - x_{1}^{3} - x_{2}^{5} \\right)
         \\exp\\left( - x_{1}^{2} - x_{2}^{2} \\right)
       - \\frac{1}{3} \\exp\\left( - (x_{1} + 1)^{2} - x_{2}^{2} \\right)

    Parameters
    ----------
    x : array_like
        The :math:`2`-vector.

    Returns
    -------
    float
        Function value at :math:`\\mathbf{x}`.

    Examples
    --------
    >>> import fbench
    >>> round(fbench.peaks([0, 0]), 4)
    0.981
    """
    x1, x2 = fbench.check_vector(x, n_max=2)
    f1 = 3 * (1 - x1) ** 2 * np.exp(-(x1**2) - (x2 + 1) ** 2)
    f2 = 10 * (x1 / 5 - x1**3 - x2**5) * np.exp(-(x1**2) - x2**2)
    f3 = 1 / 3 * np.exp(-((x1 + 1) ** 2) - x2**2)
    return float(f1 - f2 - f3)


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


def sinc(x, /):
    """Sinc function.

    A function :math:`f\\colon \\mathbb{R}^{1} \\rightarrow \\mathbb{R}`
    that takes an :math:`1`-vector as input and returns a scalar value.

    .. math::

        f(\\mathbf{x}) =
        \\begin{cases}
            \\frac{\\sin(x)}{x} & \\text{ if } x \\neq 0 \\\\
            1 & \\text{ if } x = 0
        \\end{cases}

    Parameters
    ----------
    x : array_like
        The :math:`1`-vector.

    Returns
    -------
    float
        Function value at :math:`\\mathbf{x}`.

    References
    ----------
    .. [1] "Sinc Function", Wolfram MathWorld,
           `<https://mathworld.wolfram.com/SincFunction.html>`_

    Examples
    --------
    >>> import fbench
    >>> fbench.sinc([0])
    1.0

    >>> round(fbench.sinc([1]), 4)
    0.8415
    """
    x = fbench.check_vector(x, n_max=1)[0]
    return float(1 if x == 0 else np.sin(x) / x)


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

import numpy as np

__all__ = ("check_vector",)


def check_vector(x, /, *, n_min=1, n_max=np.inf):
    """Validate :math:`n`-vector.

    Parameters
    ----------
    x : array_like
        The input object to be validated to represent an :math:`n`-vector.
    n_min : int, default=1
        Specify the minimum number of :math:`n`.
    n_max : int, default=inf
        Specify the maximum number of :math:`n`.

    Returns
    -------
    np.ndarray
        The :math:`n`-vector.

    Raises
    ------
    TypeError
        - If ``x`` is not vector-like.
        - If ``n`` is not between ``n_min`` and ``n_max``.

    Examples
    --------
    >>> import fbench
    >>> fbench.check_vector([0, 0])
    array([0, 0])
    """
    x = np.atleast_1d(x)

    if len(x.shape) != 1:
        raise TypeError(f"input must be a vector-like object - it has shape={x.shape}")

    if not (n_min <= len(x) <= n_max):
        raise TypeError(f"n={len(x)} is not between n_min={n_min} and n_max={n_max}")

    return x

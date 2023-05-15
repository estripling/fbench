import numpy as np
import numpy.testing as npt
import pytest

import fbench


def test_check_vector():
    x = [1, 2, 3]
    actual = fbench.check_vector(x)
    npt.assert_array_equal(actual, np.array(x))

    with pytest.raises(
        TypeError,
        match=r"input must be a vector-like object - it has shape=\(1, 2\)",
    ):
        fbench.check_vector([[1, 2]])

    with pytest.raises(TypeError, match=r"n=2 is not between n_min=3 and n_max=inf"):
        fbench.check_vector([1, 2], n_min=3)

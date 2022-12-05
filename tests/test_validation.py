import numpy as np
import numpy.testing as npt
import pytest

from fbench import exception, validation


def test_check_vector():
    x = [1, 2, 3]
    actual = validation.check_vector(x, 1)
    npt.assert_array_equal(actual, np.array(x))

    with pytest.raises(exception.NotAVectorError):
        validation.check_vector([[1, 2]], 1)

    with pytest.raises(exception.IncorrectNumberOfElements):
        validation.check_vector([1, 2], 3)

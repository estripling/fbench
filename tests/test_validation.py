import numpy as np
import numpy.testing as npt
import pytest

import fbench


def test_check_vector():
    x = [1, 2, 3]
    actual = fbench.check_vector(x, min_elements=1)
    npt.assert_array_equal(actual, np.array(x))

    with pytest.raises(fbench.exception.NotAVectorError):
        fbench.check_vector([[1, 2]], min_elements=1)

    with pytest.raises(fbench.exception.IncorrectNumberOfElements):
        fbench.check_vector([1, 2], min_elements=3)

import numpy as np
import numpy.testing as npt
import pytest

import fbench


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0.0),
        ([0, 0], 0.0),
        ([1, 1], 3.6254),
        ([2, 2], 6.5936),
    ],
)
def test_ackley(x, expected):
    actual = fbench.ackley(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0, 0], 0.9810),
        ([1, 1], 2.4338),
        ([2, 2], 0.1328),
    ],
)
def test_peaks(x, expected):
    actual = fbench.peaks(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0.0),
        ([0, 0], 0.0),
        ([1, 1], 2.0),
        ([1, 2], 5.0),
        ([2, 1], 5.0),
        ([2, 2], 8.0),
        ([4, 4.5], 56.25),
        ([4.5, 4], 56.25),
        ([5.12, 5.12], 57.85),
    ],
)
def test_rastrigin(x, expected):
    actual = fbench.rastrigin(x)
    assert round(actual, 2) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0, 0], 1.0),
        ([1, 1], 0.0),
        ([1, 1, 1], 0.0),
        ([1, 1, 1, 1], 0.0),
        ([1, 2], 100.0),
        ([2, 1], 901.0),
        ([2, 2], 401.0),
        ([4, 4.5], 13234.0),
        ([4.5, 4], 26418.5),
        ([5.12, 5.12], 44514.35),
    ],
)
def test_rosenbrock(x, expected):
    actual = fbench.rosenbrock(x)
    assert round(actual, 2) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 1.0),
        ([1], 0.8415),
    ],
)
def test_sinc(x, expected):
    actual = fbench.sinc(x)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([0], 0),
        ([0, 0], 0),
        ([1, 1], 2),
        ([2, 2], 8),
    ],
)
def test_sphere(x, expected):
    actual = fbench.sphere(x)
    assert actual == expected


@pytest.mark.parametrize(
    "func, n, idx, expected_x, expected_fx",
    [
        (fbench.ackley, 1, 0, np.zeros(1), 0),
        (fbench.ackley, 2, 0, np.zeros(2), 0),
        (fbench.ackley, 5, 0, np.zeros(5), 0),
        (fbench.rastrigin, 5, 0, np.zeros(5), 0),
        (fbench.rosenbrock, 5, 0, np.ones(5), 0),
        (fbench.sphere, 5, 0, np.zeros(5), 0),
    ],
)
def test_get_optima(func, n, idx, expected_x, expected_fx):
    actual = fbench.get_optima(n, func)
    assert all(isinstance(opt, fbench.structure.Optimum) for opt in actual)
    opt = actual[idx]
    npt.assert_array_almost_equal(opt.x, expected_x)
    assert opt.fx == expected_fx
    assert opt.n == n

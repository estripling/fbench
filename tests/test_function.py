import pytest

import fbench


@pytest.mark.parametrize(
    "arg, expected",
    [
        ([0], 0.0),
        ([0, 0], 0.0),
        ([1, 1], 3.6254),
        ([2, 2], 6.5936),
    ],
)
def test_ackley(arg, expected):
    actual = fbench.ackley(x=arg)
    assert round(actual, 4) == expected


@pytest.mark.parametrize(
    "arg, expected",
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
def test_rastrigin(arg, expected):
    actual = fbench.rastrigin(x=arg)
    assert round(actual, 2) == expected


@pytest.mark.parametrize(
    "arg, expected",
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
def test_rosenbrock(arg, expected):
    actual = fbench.rosenbrock(x=arg)
    assert round(actual, 2) == expected


@pytest.mark.parametrize(
    "arg, expected",
    [
        ([0], 0),
        ([0, 0], 0),
        ([1, 1], 2),
        ([2, 2], 8),
    ],
)
def test_sphere(arg, expected):
    actual = fbench.sphere(x=arg)
    assert actual == expected

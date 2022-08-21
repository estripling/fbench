import pytest

from fbench import core


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
    actual = core.ackley(x=arg)
    assert round(actual, 4) == expected


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
    actual = core.sphere(x=arg)
    assert actual == expected

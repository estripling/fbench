import pytest

import fbench as fb


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
    actual = fb.ackley(x=arg)
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
    actual = fb.sphere(x=arg)
    assert actual == expected

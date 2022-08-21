import pytest

from fbench import core


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

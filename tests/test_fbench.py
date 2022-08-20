from importlib.metadata import version

import fbench


def test_version():
    assert fbench.__version__ == version("fbench")

from importlib import metadata

import fbench


def test_version():
    assert fbench.__version__ == metadata.version("fbench")

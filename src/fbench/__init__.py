from importlib.metadata import version

__version__ = version("fbench")

from .core import *

del core

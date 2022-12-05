from importlib.metadata import version

__version__ = version("fbench")

from .core import *
from .validation import *

del core, validation

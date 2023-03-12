from importlib import metadata

__version__ = metadata.version("fbench")

from .function import *
from .validation import *

del (
    function,
    validation,
)

from importlib import metadata

__version__ = metadata.version("fbench")

from . import structure, visualization
from .function import *
from .validation import *

del (
    metadata,
    function,
    validation,
)

from importlib import metadata

__version__ = metadata.version("fbench")

from .function import *
from .validation import *
from .visualization import *

del (
    metadata,
    function,
    validation,
    visualization,
)

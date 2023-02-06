from importlib import metadata

__version__ = metadata.version("fbench")

from .core import *
from .validation import *

del (
    core,
    validation,
)

from __future__ import absolute_import
import gc

from sympy.core import cache

from devito.base import *  # noqa
from devito.backends import init_backend
from devito.finite_difference import *  # noqa
from devito.dimension import *  # noqa
from devito.grid import *  # noqa
from devito.function import Forward, Backward  # noqa
from devito.types import _SymbolCache  # noqa
from devito.logger import error, warning, info  # noqa
from devito.parameters import *  # noqa
from devito.symbolics import *  # noqa
from devito.tools import *  # noqa


def clear_cache():
    cache.clear_cache()
    gc.collect()

    for key, val in list(_SymbolCache.items()):
        if val() is None:
            del _SymbolCache[key]


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions


# Initialize the Devito backend
init_configuration()
init_backend(configuration['backend'])

"""
The ``yask`` Devito backend uses the YASK stencil optimizer to generate,
JIT-compile, and run kernels.
"""

from collections import OrderedDict
import os

from devito.exceptions import InvalidOperator
from devito.logger import yask as log


def exit(emsg):
    """
    Handle fatal errors.
    """
    raise InvalidOperator("YASK Error [%s]. Exiting..." % emsg)


log("Backend initialization...")
try:
    import yask_compiler as yc
    # YASK compiler factories
    cfac = yc.yc_factory()
    nfac = yc.yc_node_factory()
    ofac = yc.yask_output_factory()
except ImportError:
    exit("Python YASK compiler bindings")
try:
    # Set directories for generated code
    path = os.environ['YASK_HOME']
except KeyError:
    exit("Missing YASK_HOME")

# YASK conventions
namespace = OrderedDict()
namespace['kernel-hook'] = 'devito_hook'
namespace['kernel-real'] = 'devito_kernel'
namespace['kernel-filename'] = 'yask_stencil_code.hpp'
namespace['path'] = path
namespace['kernel-path'] = os.path.join(path, 'src', 'kernel')
namespace['kernel-path-gen'] = os.path.join(namespace['kernel-path'], 'gen')
namespace['kernel-output'] = os.path.join(namespace['kernel-path-gen'],
                                          namespace['kernel-filename'])
namespace['time-dim'] = 't'

# TODO: this should be moved into /configuration/
arch = 'hsw'
isa = 'cpp'

log("Backend successfully initialized!")


# The following used by backends.backendSelector
from devito.pointdata import PointData  # noqa
from devito.yask.interfaces import DenseData, TimeData  # noqa
from devito.yask.operator import Operator  # noqa

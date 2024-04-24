r""" Base classes for Bayesian experimental design methods. """

__version__ = '0.0.1'

from . import core
from . import eig
from . import fwd_collection

from .core import (
    BED_base_explicit,
    BED_base_nuisance,
    # BED_base_implicit
    )

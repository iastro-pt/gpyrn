# -*- coding: utf-8 -*-
#package version
__version__ = '1.0'


from .meanfunc import Constant, Linear
from .covfunc import SquaredExponential, QuasiPeriodic

from .meanfield import inference

import pytest

import numpy as np
from gpyrn.covfunc import SquaredExponential, Periodic, QuasiPeriodic


def test_QP_equals_prod():
    η1, η2, η3, η4 = 1, 10, 20, 0.5
    k1 = SquaredExponential(η1, η2) * Periodic(1, η3, η4)
    k2 = QuasiPeriodic(η1, η2, η3, η4)

    t = np.sort(np.random.uniform(0, 100, size=50))
    T = t[:, None] - t[None, :]
    assert np.allclose(k1(T), k2(T))

import pytest

import numpy as np
from gpyrn.meanfunc import Constant, Linear


def test_Constant():
    m = Constant(0.0)
    assert m.pars[0] == 0.0
    assert np.all(m(np.random.rand(10)) == 0.0)

    m = Constant(10.0)
    assert m.pars[0] == 10.0
    assert np.all(m(np.random.rand(3)) == 10.0)

    # constant value is required
    with pytest.raises(TypeError):
        m = Constant()

    m = Constant(5.0) + Constant(10.0)
    assert np.all(m(np.random.rand(3)) == 15.0)

    m = Constant(2) * Constant(10.0)
    assert np.all(m(np.random.rand(3)) == 20.0)


def test_Linear():
    m = Linear(0.0, 1.0)
    assert m.pars[0] == 0.0
    assert m.pars[1] == 1.0
    assert np.all(m(np.random.rand(10)) == 1.0)

    m = Linear(1.0, 2.0)
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.all(m(t) == np.polyval(m.pars, t - t.mean()))

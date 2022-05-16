import pytest

import numpy as np
from gpyrn.meanfield import inference
from gpyrn import covfunc, meanfunc


def test_create_inference():
    t, y, yerr = np.random.rand(3, 10)
    gprn = inference(1, t, y, yerr)
    assert gprn.time is t
    assert gprn.N == t.size
    assert gprn.q == 1
    assert gprn.p == 1

    t, y1, ye1, y2, ye2 = np.random.rand(5, 10)
    gprn = inference(1, t, y1, ye1, y2, ye2)
    assert np.allclose(gprn.y, np.c_[y1, y2].T)
    assert gprn.q == 1
    assert gprn.p == 2


def test_create_inference_exception():
    # no time array provided
    with pytest.raises(TypeError):
        _ = inference(1)

    # wrong number of outputs
    with pytest.raises(AssertionError):
        _ = inference(1, np.random.rand(10))

    # mismatched shapes
    t, y1, ye1 = np.random.rand(3, 10)
    y2, ye2 = np.random.rand(2, 20)
    with pytest.raises(AssertionError):
        _ = inference(1, t, y1, ye1, y2, ye2)


def test_set_components():
    t, y, yerr = np.random.rand(3, 10)
    gprn = inference(1, t, y, yerr)

    node = covfunc.SquaredExponential(1, 1)
    weight = covfunc.SquaredExponential(1, 1)
    mean = meanfunc.Constant(0)
    jitter = 0.0
    gprn.set_components(node, weight, mean, jitter)
    assert gprn.nodes[0] is node

    gprn.set_components([node], [weight], mean, jitter)
    gprn.set_components([node], [weight], [mean], [jitter])

    _ = gprn.ELBO

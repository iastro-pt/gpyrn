import numpy as np
from ._utils import _array_input


class covFunction:
    """
    Base class for covariance functions (kernels) used for nodes and weights in
    the GPRN.
    """
    def __init__(self, *args):
        self.pars = np.array(args, dtype=float)

    def __call__(self, r, t1=None, t2=None):
        raise NotImplementedError

    def _dkdxidj(self, r):
        raise NotImplementedError

    def __repr__(self):
        if hasattr(self, '_param_names'):
            pars = ', '.join(
                [f'{p}={v}' for p, v in zip(self._param_names, self.pars)])
        else:
            pars = ', '.join(map(str, self.pars))
        return f"{self.__class__.__name__}({pars})"

    def get_parameters(self):
        return self.pars

    @_array_input
    def set_parameters(self, p):
        msg = f'too few parameters for kernel {self.__class__.__name__}'
        assert len(p) >= self.pars.size, msg
        if len(p) > self.pars.size:
            p = list(p)
            self.pars = np.array(p[:self.pars.size], dtype=float)
            for _ in range(self.pars.size):
                p.pop(0)
            return np.array(p)
        else:
            self.pars = p

    def __add__(self, b):
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Multiplication(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(covFunction):
    """ To allow operations between two kernels """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.kerneltype = 'complex'
        self.pars = np.r_[self.k1.pars, self.k2.pars]


class Sum(_operator):
    """ Sum of two covariance functions """
    def __call__(self, r):
        return self.k1(r) + self.k2(r)

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)


class Multiplication(_operator):
    """ Product of two covariance functions """
    def __call__(self, r):
        return self.k1(r) * self.k2(r)

    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)


class _unary_operator(covFunction):
    """ To allow operations for one kernel """
    def __init__(self, k):
        msg = f'kernel {k} is not twice differentiable'
        if not hasattr(k, '_twice_differentiable') or not k._twice_differentiable:
            raise ValueError(msg)

        self.k = k
        self.kerneltype = 'complex_unary'
        self.pars = self.k.pars

        self._param_names = self.k._param_names
        self._tag = 'd' + self.k._tag


class Derivative(_unary_operator):
    def __call__(self, r):
        return self.k._dkdxidj(r)

    def __repr__(self):
        self.k.pars = self.pars
        return "d {0}".format(self.k)


class Constant(covFunction):
    """
    This kernel returns the square of its constant argument c
    $$
      K_{ij} = c^2
    $$

    Args:
        c: Constant value
    """
    _param_names = 'c',
    _tag = 'C'

    def __init__(self, c: float):
        super(Constant, self).__init__(c)

    def __call__(self, r):
        c = self.pars[0]
        return np.full_like(r, c**2)


class WhiteNoise(covFunction):
    """
    White noise (diagonal) kernel
    $$
      K_{ij} = w^2 \, \delta_{ij}
    $$

    Args:
        w: White noise amplitude
    """
    _param_names = 'wn',
    _tag = 'WN'

    def __init__(self, w:float):
        super(WhiteNoise, self).__init__(w)

    def __call__(self, r):
        w = self.pars[0]
        if r.ndim == 2 and r[0, :].shape == r[:, 0].shape:
            return w**2 * np.diag(np.diag(np.ones_like(r)))
        return np.full_like(r, w**2)


class SquaredExponential(covFunction):
    r"""
    Squared Exponential kernel, also known as radial basis function
    $$
      K_{ij} = \theta^2 \, \exp \left[ - \frac{(t_i - t_j)^2}{2 \ell^2} \right]
    $$

    Args:
        theta: Amplitude
        ell: Length-scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'SE'
    _twice_differentiable = True

    def __init__(self, theta: float, ell: float):
        super(SquaredExponential, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-0.5 * r**2 / self.pars[1]**2)

    def _dkdxi(self, r):
        θ, ell = self.pars
        return θ**2 * (-r) * np.exp(-0.5 * (-r)**2 / ell**2) / ell**2

    def _dkdxj(self, r):
        # covariance between an observation at xi
        # and a derivative observation at xj
        θ, ell = self.pars
        return θ**2 * r * np.exp(-0.5 * r**2 / ell**2) / ell**2

    def _dkdxidj(self, r):
        term1 = self.pars[0]**2 / self.pars[1]**4
        term2 = self.pars[1]**2 - r**2
        return term1 * term2 * np.exp(-0.5 * r**2 / self.pars[1]**2)


class Periodic(covFunction):
    r"""
    Periodic kernel, also known as the exponential sine squared
    $$
      K_{ij} = \theta^2 \, \exp \left[ - \frac{2 \sin^2 \left( \frac{\pi (t_i - t_j)}{P} \right)}{\ell^2} \right]
    $$

    Args:
        theta: Amplitude
        P: Period
        ell: Lenght scale

    Note: Parameterization
        Note that the periodic kernel is sometimes parameterized differently,
        namely using $\Gamma = 2/\ell^2$.  
    """
    _param_names = 'theta', 'P', 'ell'
    _tag = 'P'
    _twice_differentiable = True

    def __init__(self, theta: float, P: float, ell: float):
        super(Periodic, self).__init__(theta, P, ell)

    def __call__(self, r):
        θ, P, ell = self.pars
        return θ**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / P)**2 / ell**2)

    def _dkdxidj(self, r):
        θ, P, ell = self.pars
        rP = np.pi * r / P
        term1 = 4 * np.pi**2 * θ**2
        term2 = ell**2 * np.cos(2 * rP) - 4 * np.sin(rP)**2 * np.cos(rP)**2
        term3 = np.exp(-2 * np.sin(rP)**2 / ell**2)
        return term1 * term2 * term3


class QuasiPeriodic(covFunction):
    r"""
    This kernel is the product between the periodic kernel and the squared
    exponential kernel, and is commonly known as the quasi-periodic kernel
    $$
      K_{ij} = \theta^2 \,
                  \exp \left[ - \frac{(t_i - t_j)^2}{2 \ell_e^2} - \frac{2 \sin^2 \left( \frac{\pi (t_i - t_j)}{P} \right)}{\ell_p^2} \right]
    $$

    Args:
        theta: Amplitude
        elle: Evolutionary length scale
        P: Kernel periodicity
        ellp: Length scale of the periodic component

    Info:
        The `QuasiPeriodic` kernel is implemented on its own for convenience,
        but it is exactly equivalent to the product of a `SquaredExponential`
        and a `Periodic` kernel with the same parameters.
    """
    _param_names = 'theta', 'le', 'P', 'lp'
    _tag = 'QP'
    _twice_differentiable = True

    def __init__(self, theta: float, elle: float, P: float, ellp: float):
        super(QuasiPeriodic, self).__init__(theta, elle, P, ellp)

    def __call__(self, r):
        θ, elle, P, ellp = self.pars
        term1 = -2 * np.sin(np.pi * np.abs(r) / P)**2 / ellp**2
        term2 = r**2 / (2 * elle**2)
        return θ**2 * np.exp(term1 - term2)

    def _dkdxidj(self, r):
        θ, elle, P, ellp = self.pars
        term1 = 2 * θ**2 / (P**2 * ellp**4 * elle**4)
        term2 = P**2 * ellp**4 * elle**2 - \
                2 * P**2 * ellp**4 * r**2 - \
                4 * np.pi * P * ellp**2 * elle**2 * r * np.sin(2 * np.pi * r / P) + \
                2 * np.pi**2 * ellp**2 * elle**4 * np.cos(2 * np.pi * r / P) - \
                8 * np.pi**2 * elle**4 * np.sin(np.pi * r / P)**2 * np.cos(np.pi * r / P)**2
        term3 = np.exp(-(ellp**2 * r**2 + 2 * elle**2 * np.sin(np.pi * r / P)**2) / (ellp**2 * elle**2))
        return term1 * term2 * term3


class RationalQuadratic(covFunction):
    """
    The rational quadratic kernel
    $$
    $$

    Args:
        theta: Amplitude of the kernel
        alpha: Amplitude of large and small scale variations
        ell: Characteristic lenght scale to define the kernel "smoothness"
    """
    _param_names = 'theta', 'alpha', 'ell'
    _tag = 'RQ'

    def __init__(self, theta: float, alpha: float, ell: float):
        super(RationalQuadratic, self).__init__(theta, alpha, ell)

    def __call__(self, r):
        θ, α, ell = self.pars
        return θ**2 * (1 + 0.5 * r**2 / (α * ell**2))**(-α)


class RQP(covFunction):
    """
    Product between the periodic kernel and the rational quadratic kernel that
    we call RQP kernel.

    Args:
        theta: Amplitude
        alpha: Alpha of the rational quadratic kernel
        elle: Aperiodic length scale
        P: Periodic repetitions of the kernel
        ellp: Periodic length scale
    """
    _param_names = 'theta', 'alpha', 'elle', 'ellp', 'P'
    _tag = 'RQP'

    def __init__(self, theta: float, alpha: float, elle: float, P: float,
                 ellp: float):
        super(RQP, self).__init__(theta, alpha, elle, P, ellp)

    def __call__(self, r):
        θ, α, ℓe, P, ℓp = self.pars
        return θ**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / P)**2 /
                             ℓp**2) * (1 + r**2 / (2 * α * ℓe**2))**(-α)


class COSINE(covFunction):
    """
    the cosine kernel

    Args:
        theta: Amplitude
        P: Period
    """
    _param_names = 'theta', 'P'
    _tag = 'COS'

    def __init__(self, theta: float, P: float):
        super(COSINE, self).__init__(theta, P)

    def __call__(self, r):
        return self.pars[0]**2 * np.cos(2 * np.pi * np.abs(r) / self.pars[1])


class Exponential(covFunction):
    r"""
    The exponential kernel
    $$
      K_{ij} = \theta^2 \, \exp \left( - \frac{|t_i - t_j|}{\ell} \right)
    $$

    Args:
        theta: Amplitude
        ell: Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'EXP'

    def __init__(self, theta: float, ell: float):
        super(Exponential, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-np.abs(r) / self.pars[1])


class Matern32(covFunction):
    """
    the Matern 3/2 kernel. This kernel arise when setting
    v=3/2 in the matern family of kernels

    Args:
        theta: Amplitude
        ell: Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'M32'

    def __init__(self, theta: float, ell: float):
        super(Matern32, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * (
            1.0 + np.sqrt(3.0) * np.abs(r) / self.pars[1]) * np.exp(
                -np.sqrt(3.0) * np.abs(r) / self.pars[1])


#### Matern 5/2 ################################################################
class Matern52(covFunction):
    """
    the Matern 5/2 kernel. This kernel arise when setting v=5/2
    in the matern family of kernels

    Args:
        theta: Amplitude
        ell: Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'M52'

    def __init__(self, theta: float, ell: float):
        super(Matern52, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * (
            1.0 +
            (3 * np.sqrt(5) * self.pars[1] * np.abs(r) + 5 * np.abs(r)**2) /
            (3 * self.pars[1]**2)) * np.exp(
                -np.sqrt(5.0) * np.abs(r) / self.pars[1])


class Linear(covFunction):
    """
    the Linear kernel

    Args:
        c: Constant
    """
    def __init__(self, c):
        super(Linear, self).__init__(c)
        self.tag = 'LIN'
        self.c = c

    def __call__(self, r, t1, t2):
        return  (t1 - self.pars[0]) * (t2 - self.pars[0])


class GammaExp(covFunction):
    """
    the gamma-exponential kernel

    Args:
        theta: Amplitude
        gamma: Shape parameter ( 0 < gamma <= 2)
        l: Lenght scale
    """
    def __init__(self, theta, gamma, l):
        super(GammaExp, self).__init__(theta, gamma, l)
        self.tag = 'GammaExp'
        self.theta = theta
        self.gamma = gamma
        self.l = l

    def __call__(self, r):
        return self.pars[0]**2 *np.exp(-(np.abs(r)/self.pars[2])**self.pars[1])


class Polynomial(covFunction):
    """
    the polynomial kernel

    Args:
        theta: Amplitude ???
        a: Real value > 0
        b: Real value >= 0
        c: Integer value
        wn: White noise amplitude
    """
    def __init__(self, theta, a, b, c):
        super(Polynomial, self).__init__(theta, a, b, c)
        self.tag = 'POLY'
        self.theta = theta
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, r, t1, t2):
        return (self.pars[1] * t1 * t2 + self.pars[2])**self.pars[3]


##### Piecewise ################################################################
class Piecewise(covFunction):
    """
    WARNING: EXPERIMENTAL KERNEL
    
    Parameters
    ----------
    """
    def __init__(self, eta):
        super(Piecewise, self).__init__(eta)
        self.eta = eta
        self.type = 'unknown'
    def __call__(self, r):
        r = r/(0.5*self.pars[0])
        piecewise = (3*np.abs(r) +1) * (1 - np.abs(r))**3
        piecewise = np.where(np.abs(r)>1, 0, piecewise)
        return piecewise


### END

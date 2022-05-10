"""
Covariance functions to use on the GPRN
"""
from gpyrn.meanfunc import array_input
import numpy as np


class covFunction():
    """
    A base class for covariance functions (kernels) used for nodes and weights
    in the GPRN.
    """
    def __init__(self, *args):
        """ Puts all kernel arguments in an array pars """
        self.pars = np.array(args, dtype=float)
        # self.pars[self.pars > 1e50] = 1e50

    def __call__(self, r, t1=None, t2=None):
        """
        r = t - t'
        Not sure if this is a good approach since will make our life harder
        when defining certain non-stationary kernels, e.g linear kernel.
        """
        raise NotImplementedError

    def __repr__(self):
        """ Representation of each kernel instance """
        if hasattr(self, '_param_names'):
            pars = ', '.join(
                [f'{p}={v}' for p, v in zip(self._param_names, self.pars)])
        else:
            pars = ', '.join(map(str, self.pars))
        return f"{self.__class__.__name__}({pars})"

    def get_parameters(self):
        return self.pars

    @array_input
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
    """ To allow the sum of kernels """
    def __call__(self, r):
        return self.k1(r) + self.k2(r)

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)


class Multiplication(_operator):
    """ To allow the multiplication of kernels """
    def __call__(self, r):
        return self.k1(r) * self.k2(r)

    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)


##### Constant #################################################################
class Constant(covFunction):
    """
    This kernel returns the square of its constant argument c

    Parameters
    ----------
    c: float
        Constant
    """
    _param_names = 'c',
    _tag = 'C'

    def __init__(self, c):
        super(Constant, self).__init__(c)

    def __call__(self, r):
        c = self.pars[0]
        return np.full_like(r, c**2)


##### White Noise ##############################################################
class WhiteNoise(covFunction):
    """
    Definition of the white noise kernel.

    Parameters
    ----------
    wn: float
        White noise amplitude
    """
    _param_names = 'wn',
    _tag = 'WN'

    def __init__(self, wn):
        super(WhiteNoise, self).__init__(wn)

    def __call__(self, r):
        wn = self.pars[0]
        if r.ndim == 2 and r[0, :].shape == r[:, 0].shape:
            return wn**2 * np.diag(np.diag(np.ones_like(r)))
        return np.full_like(r, wn**2)


##### Squared exponential ######################################################
class SquaredExponential(covFunction):
    """
    Squared Exponential kernel, also known as radial basis function or RBF
    kernel in other works.

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Length-scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'SE'

    def __init__(self, theta, ell):
        super(SquaredExponential, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-0.5 * r**2 / self.pars[1]**2)


##### Periodic #################################################################
class Periodic(covFunction):
    """
    Definition of the periodic kernel.

    Parameters
    ----------
    theta: float
        Amplitude
    P: float
        Period
    lp: float
        Lenght scale
    """
    _param_names = 'theta', 'P', 'lp'
    _tag = 'P'

    def __init__(self, theta, P, lp):
        super(Periodic, self).__init__(theta, P, lp)

    def __call__(self, r):
        θ, P, lp = self.pars
        return θ**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / P)**2 / lp**2)


##### Quasi Periodic ###########################################################
class QuasiPeriodic(covFunction):
    """
    This kernel is the product between the exponential sine squared kernel
    and the squared exponential kernel, commonly known as the quasi-periodic
    kernel

    Parameters
    ----------
    theta: float
        Amplitude
    le: float
        Evolutionary time scale
    P: float
        Kernel periodicity
    lp: float
        Length scale of the periodic component
    """
    _param_names = 'theta', 'le', 'P', 'lp'
    _tag = 'QP'

    def __init__(self, theta, le, P, lp):
        super(QuasiPeriodic, self).__init__(theta, le, P, lp)

    def __call__(self, r):
        θ, le, P, lp = self.pars
        return θ**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / P)**2 / lp**2 - \
                             r**2 / (2 * le**2))


##### Rational Quadratic #######################################################
class RationalQuadratic(covFunction):
    """
    Definition of the rational quadratic kernel.

    Parameters
    ----------
    theta: float
        Amplitude of the kernel
    alpha: float
        Amplitude of large and small scale variations
    ell: float
        Characteristic lenght scale to define the kernel "smoothness"
    """
    _param_names = 'theta', 'alpha', 'ell'
    _tag = 'RQ'

    def __init__(self, theta, alpha, ell):
        super(RationalQuadratic, self).__init__(theta, alpha, ell)

    def __call__(self, r):
        θ, α, ell = self.pars
        return θ**2 * (1 + 0.5 * r**2 / (α * ell**2))**(-α)


##### RQP kernel ###############################################################
class RQP(covFunction):
    """
    Definition of the product between the exponential sine squared kernel and
    the rational quadratic kernel that we called RQP kernel. If I am thinking
    this correctly then this kernel should tend to the QuasiPeriodic kernel as
    alpha increases, although I am not sure if we can say that it tends to the
    QuasiPeriodic kernel as alpha tends to infinity.

    Parameters
    ----------
    theta: float
        Amplitude
    alpha: float
        Alpha of the rational quadratic kernel
    ell_e: float
        Aperiodic length scale
    P: float
        Periodic repetitions of the kernel
    ell_p: float
        Periodic length scale
    """
    _param_names = 'theta', 'alpha', 'ell_e', 'ell_p', 'P'
    _tag = 'RQP'

    def __init__(self, theta, alpha, ell_e, P, ell_p):
        super(RQP, self).__init__(theta, alpha, ell_e, P, ell_p)

    def __call__(self, r):
        θ, α, ℓe, P, ℓp = self.pars
        return θ**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / P)**2 /
                             ℓp**2) * (1 + r**2 / (2 * α * ℓe**2))**(-α)


##### Cosine ###################################################################
class COSINE(covFunction):
    """
    Definition of the cosine kernel

    Parameters
    ----------
    theta: float
        Amplitude
    P: float
        Period
    """
    _param_names = 'theta', 'P'
    _tag = 'COS'

    def __init__(self, theta, P):
        super(COSINE, self).__init__(theta, P)

    def __call__(self, r):
        return self.pars[0]**2 * np.cos(2 * np.pi * np.abs(r) / self.pars[1])


##### Laplacian ##############################################################
class Laplacian(covFunction):
    """
    Definition of the Laplacian kernel

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'LAP'

    def __init__(self, theta, ell):
        super(Laplacian, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-np.abs(r) / self.pars[1])


##### Exponential ##############################################################
class Exponential(covFunction):
    """
    Definition of the exponential kernel

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'EXP'

    def __init__(self, theta, ell):
        super(Exponential, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-np.abs(r) / self.pars[1])


##### Matern 3/2 ###############################################################
class Matern32(covFunction):
    """
    Definition of the Matern 3/2 kernel. This kernel arise when setting
    v=3/2 in the matern family of kernels

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'M32'

    def __init__(self, theta, ell):
        super(Matern32, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * (
            1.0 + np.sqrt(3.0) * np.abs(r) / self.pars[1]) * np.exp(
                -np.sqrt(3.0) * np.abs(r) / self.pars[1])


#### Matern 5/2 ################################################################
class Matern52(covFunction):
    """
    Definition of the Matern 5/2 kernel. This kernel arise when setting v=5/2
    in the matern family of kernels

    Parameters
    ----------
    theta: float
        Amplitude
    ell: float
        Characteristic lenght scale
    """
    _param_names = 'theta', 'ell'
    _tag = 'M52'

    def __init__(self, theta, ell):
        super(Matern52, self).__init__(theta, ell)

    def __call__(self, r):
        return self.pars[0]**2 * (
            1.0 +
            (3 * np.sqrt(5) * self.pars[1] * np.abs(r) + 5 * np.abs(r)**2) /
            (3 * self.pars[1]**2)) * np.exp(
                -np.sqrt(5.0) * np.abs(r) / self.pars[1])


#### Linear ####################################################################
class Linear(covFunction):
    """
    Definition of the Linear kernel

    Parameters
    ----------
    theta: float
        Amplitude (should we even have an amplitude???)
    c: float
        Constant
    """
    def __init__(self, theta, c):
        super(Linear, self).__init__(theta, c)
        self.tag = 'LIN'
        self.theta = theta
        self.c = c

    def __call__(self, r, t1, t2):
        return  (t1 - self.pars[1]) * (t2 - self.pars[1])


##### Gamma-exponential ########################################################
class GammaExp(covFunction):
    """
    Definition of the gamma-exponential kernel

    Parameters
    ----------
    theta: float
        Amplitude
    gamma: float
        Shape parameter ( 0 < gamma <= 2)
    l: float
        Lenght scale

    Returns
    -------
    """
    def __init__(self, theta, gamma, l):
        super(GammaExp, self).__init__(theta, gamma, l)
        self.tag = 'GammaExp'
        self.theta = theta
        self.gamma = gamma
        self.l = l

    def __call__(self, r):
        return self.pars[0]**2 *np.exp(-(np.abs(r)/self.pars[2])**self.pars[1])


##### Polynomial ###############################################################
class Polynomial(covFunction):
    """
    Definition of the polynomial kernel

    Parameters
    ----------
    theta: float
        Amplitude ???
    a: float
        Real value > 0
    b: foat
        Real value >= 0
    c: int
        Integer value
    wn: float
        White noise amplitude

    Returns
    -------
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

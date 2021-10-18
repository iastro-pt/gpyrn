"""
Covariance functions to use on the GPRN
"""
import numpy as np

class covFunction():
    """
    Definition the covariance functions (kernels) of our GPRN, by default and
    because it simplifies my life, all kernels include a white noise term
    """
    def __init__(self, *args):
        """ Puts all kernel arguments in an array pars """
        self.pars = np.array(args, dtype=float)
        self.pars[self.pars > 1e50] = 1e50
    def __call__(self, r, t1=None, t2=None):
        """
        r = t - t'
        Not sure if this is a good approach since will make our life harder
        when defining certain non-stationary kernels, e.g linear kernel.
        """
        raise NotImplementedError

    def __repr__(self):
        """ Representation of each kernel instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

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
        super(_operator, self).__init__(k1, k2)
        self.k1 = k1
        self.k2 = k2
        self.kerneltype = 'complex'

    @property
    def pars(self):
        """ Parameters og the two kernels """
        return np.append(self.k1.pars, self.k2.pars)


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
    This kernel returns its constant argument c with white noise

    Parameters
    ----------
    c: float
        Constant
        
    Returns
    -------
    """
    def __init__(self, c, wn):
        super(Constant, self).__init__(c, wn)
        self.tag = 'C'
        self.c = c

    def __call__(self, r):
        return self.c**2 *np.ones_like(r)


##### White Noise ##############################################################
class WhiteNoise(covFunction):
    """
    Definition of the white noise kernel.

    Parameters
    ----------
    wn: float
        White noise amplitude

    Returns
    -------
    """
    def __init__(self, wn):
        super(WhiteNoise, self).__init__(wn)
        self.tag = 'WN'
        self.wn = wn

    def __call__(self, r):
        if r[0, :].shape == r[:, 0].shape:
            return self.wn**2 *np.diag(np.diag(np.ones_like(r)))
        return self.wn**2 *np.ones_like(r)


##### Squared exponential ######################################################
class SquaredExponential(covFunction):
    """
    Squared Exponential kernel, also known as radial basis function or RBF
    kernel in other works.

    Parameterstoo-many-arguments
    ----------
    theta: float
        Amplitude
    l: float
        Length-scale

    Returns
    -------
    """
    def __init__(self, theta, l):
        super(SquaredExponential, self).__init__(theta, l)
        self.tag = 'SE'
        self.theta = theta
        self.l = l

    def __call__(self, r):
        return self.pars[0]**2 *np.exp(-0.5 * r**2/self.pars[1]**2)


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

    Returns
    -------
    """
    def __init__(self, theta, P, lp):
        super(Periodic, self).__init__(theta, P, lp)
        self.tag = 'P'
        self.theta = theta
        self.lp = lp
        self.P = P

    def __call__(self, r):
        return self.pars[0]**2 *\
            np.exp(-2*np.sin(np.pi*np.abs(r)/self.pars[1])**2/self.pars[2]**2)


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

    Returns
    -------
    """
    def __init__(self, theta, le, P, lp):
        super(QuasiPeriodic, self).__init__(theta, le, P, lp)
        self.tag = 'QP'
        self.theta = theta
        self.le = le
        self.P = P
        self.lp = lp
        
    def __call__(self, r):
        return self.pars[0]**2 *np.exp(-2*np.sin(np.pi*np.abs(r)/self.pars[2])**2\
                                       /self.pars[3]**2-r**2/(2*self.pars[1]**2))


##### Rational Quadratic #######################################################
class RationalQuadratic(covFunction):
    """
    Definition of the rational quadratic kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    alpha: float
        Amplitude of large and small scale variations
    l: float
        Characteristic lenght scale to define the kernel "smoothness"
    """
    def __init__(self, amplitude, alpha, l):
        super(RationalQuadratic, self).__init__(amplitude, alpha, l)
        self.amplitude = amplitude
        self.alpha = alpha
        self.l = l
        self.params_number = 3
    def __call__(self, r):
        return self.pars[0]**2 *\
            (1+0.5*r**2/(self.pars[1]*self.pars[2]**2))**(-self.pars[1])


##### RQP kernel ###############################################################
class RQP(covFunction):
    """
    Definition of the product between the exponential sine squared kernel
    and the rational quadratic kernel that we called RQP kernel.
    If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.

    Parameters
    ----------
    theta: float
        Amplitude
    alpha: float
        Alpha of the rational quadratic kernel
    ell_e, ell_p: float
        Aperiodic and periodic lenght scales
    P: float
        Periodic repetitions of the kernel
        
    Returns
    -------
    """
    def __init__(self, theta, alpha, ell_e, P, ell_p):
        super(RQP, self).__init__(theta, alpha, ell_e, P, ell_p)
        self.tag = 'RQP'
        self.theta = theta
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return self.theta**2 *np.exp(-2*np.sin(np.pi*np.abs(r)/self.P)**2/self.ell_p**2) \
                        *(1+r**2/(2*self.alpha*self.ell_e**2))**(-self.alpha)


##### Cosine ###################################################################
class CoSINE(covFunction):
    """
    Definition of the cosine kernel

    Parameters
    ----------
    theta: float
        Amplitude
    P: float
        Period

    Returns
    -------
    """
    def __init__(self, theta, P):
        super(CoSINE, self).__init__(theta, P)
        self.tag = 'COS'
        self.theta = theta
        self.P = P

    def __call__(self, r):
        return self.pars[0]**2 *np.cos(2*np.pi*np.abs(r)/self.pars[1])


##### Laplacian ##############################################################
class Laplacian(covFunction):
    """
    Definition of the Laplacian kernel

    Parameters
    ----------
    theta: float
        Amplitude
    l: float
        Characteristic lenght scale

    Returns
    -------
    """
    def __init__(self, theta, l):
        super(Laplacian, self).__init__(theta, l)
        self.tag = 'LAP'
        self.theta = theta
        self.l = l
        
    def __call__(self, r):
        return self.pars[0]**2 *np.exp(-np.abs(r)/self.pars[1])


##### Exponential ##############################################################
class Exponential(covFunction):
    """
    Definition of the exponential kernel

    Parameters
    ----------
    theta: float
        Amplitude
    l: float
        Characteristic lenght scale

    Returns
    -------
    """
    def __init__(self, theta, l):
        super(Exponential, self).__init__(theta, l)
        self.tag = 'EXP'
        self.theta = theta
        self.l = l

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-np.abs(r)/self.pars[1])


##### Matern 3/2 ###############################################################
class Matern32(covFunction):
    """
    Definition of the Matern 3/2 kernel. This kernel arise when setting
    v=3/2 in the matern family of kernels

    Parameters
    ----------
    theta: float
        Amplitude
    l: float
        Characteristic lenght scale

    Returns
    -------
    """
    def __init__(self, theta, l):
        super(Matern32, self).__init__(theta, l)
        self.tag = 'M32'
        self.theta = theta
        self.ell = l

    def __call__(self, r):
        return self.pars[0]**2 *(1.0+np.sqrt(3.0)*np.abs(r)/self.pars[1]) \
            *np.exp(-np.sqrt(3.0)*np.abs(r)/self.pars[1])


#### Matern 5/2 ################################################################
class Matern52(covFunction):
    """
    Definition of the Matern 5/2 kernel. This kernel arise when setting v=5/2
    in the matern family of kernels

    Parameters
    ----------
    theta: float
        Amplitude
    l: float
        Characteristic lenght scale

    Returns
    -------
    """
    def __init__(self, theta, l):
        super(Matern52, self).__init__(theta, l)
        self.tag = 'M52'
        self.theta = theta
        self.ell = l

    def __call__(self, r):
        return self.pars[0]**2 *\
            (1.0+(3*np.sqrt(5)*self.pars[1]*np.abs(r)+5*np.abs(r)**2)/(3*self.pars[1]**2))\
                *np.exp(-np.sqrt(5.0)*np.abs(r)/self.pars[1])


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
        
    Returns
    -------
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


##### Polinomial ###############################################################
class Polynomial(covFunction):
    """
    Definition of the polinomial kernel

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

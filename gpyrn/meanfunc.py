"""
Mean functions to use on the GPRN
"""
from functools import wraps
import numpy as np

__all__ = ['Constant', 'Linear', 'Parabola', 'Cubic', 'Keplerian']

def array_input(f):
    """ Decorator to provide the __call__ methods with an array """
    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r
    return wrapped


class MeanModel():
    """ Class for our mean functions"""
    _parsize = 0
    def __init__(self, *pars):
        self.pars = list(pars)
        #np.array(pars, dtype=float)

    def __repr__(self):
        """ Representation of each instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    @classmethod
    def initialize(cls):
        """ Initialize instance, setting all parameters to 0. """
        return cls(*([0.]*cls._parsize))

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)


class Sum(MeanModel):
    """ Sum of two mean functions """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2

    @property
    def _parsize(self):
        return self.m1._parsize + self.m2._parsize

    @property
    def pars(self):
        """ Sum of parameters? """
        return self.m1.pars + self.m2.pars

    def initialize(self):
        return

    def __repr__(self):
        return "{0} + {1}".format(self.m1, self.m2)

    @array_input
    def __call__(self, t):
        return self.m1(t) + self.m2(t)


##### f(x) = a #################################################################
class Constant(MeanModel):
    """  A constant offset mean function """
    _parsize = 1
    def __init__(self, c):
        super(Constant, self).__init__(c)
        self.c = c

    @array_input
    def __call__(self, t):
        return np.full(t.shape, self.c)


##### f(x) = ax + b ############################################################
class Linear(MeanModel):
    """
    A linear mean function
    m(t) = slope * t + intercept
    """
    _parsize = 2
    def __init__(self, slope, intercept):
        super(Linear, self).__init__(slope, intercept)
        self.slope, self.intercept = slope, intercept

    @array_input
    def __call__(self, t):
        tmean = t.mean()
        return self.pars[0] * (t-tmean) + self.pars[1]


##### f(x) = ax**2 + bx + c ####################################################
class Parabola(MeanModel):
    """
    A 2nd degree polynomial mean function
    m(t) = quad * t**2 + slope * t + intercept
    """
    _parsize = 3
    def __init__(self, quad, slope, intercept):
        super(Parabola, self).__init__(quad, slope, intercept)
        self.quad, self.slope, self.intercept = quad, slope, intercept

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


##### f(x) = ax**3 + bx**2 + cx + d ############################################
class Cubic(MeanModel):
    """
    A 3rd degree polynomial mean function
    m(t) = cub * t**3 + quad * t**2 + slope * t + intercept
    """
    _parsize = 4
    def __init__(self, cub, quad, slope, intercept):
        super(Cubic, self).__init__(cub, quad, slope, intercept)
        self.cub = cub
        self.quad, self.slope, self.intercept = quad, slope, intercept

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


##### f(x) = a**2 * sine(2*pi*t/b + c) + d #####################################
class Sine(MeanModel):
    """
        A sinusoidal mean function
        m(t) = amplitude**2 * sine( (2*pi*t/P) + phase) + displacement
    """
    _parsize = 3
    def __init__(self, amp, P, phi, D):
        super(Sine, self).__init__(amp, P, phi, D)
        self.amp, self.P, self.phi, self.D = amp, P, phi, D

    @array_input
    def __call__(self, t):
        return self.pars[0] * np.sin((2*np.pi*t/self.pars[1]) + self.pars[2]) \
                + self.pars[3]


##### f(x) = K*(e*np.cos(w+np.cos(w+nu(x))) + d ################################
class Keplerian(MeanModel):
    """
    Keplerian function adapted from radvel
    We have RV = K[cos(w+v) + e*cos(w)] + sys_vel
    P  = period in days
    e = eccentricity
    K = RV amplitude in m/s
    w = longitude of the periastron
    phi = orbital phase
    sys_vel = offset
    """
    _parsize = 6
    def __init__(self, P, K, e, w, phi, sys_vel):
        super(Keplerian, self).__init__(P, K, e, w, phi, sys_vel)
        self.P, self.K, self.e, self.w, self.phi = P, K, e, w, phi
        self.sys_vel = sys_vel

    @array_input
    def __call__(self, t):
        P, K, e, w, phi, sys_vel = self.pars
        T0 = t[0] - (P*phi)/(2.*np.pi)
        M0 = 2*np.pi*(t-T0)/P #first guess at M
        E0 = M0 + e*np.sin(M0) + 0.5*(e**2)*np.sin(2*M0)  #first guess at E
        M1 = ( E0 - e * np.sin(E0) - M0) #goes to zero when converges
        criteria = 1e-10
        convd = np.where(np.abs(M1) > criteria)[0]  # which indices have not converged
        nd = len(convd)  # number of unconverged elements
        count = 0
        while nd > 0:
            count += 1
            E = E0
            M1p = 1 - e * np.cos(E)
            M1pp = e * np.sin(E)
            M1ppp = 1 - M1p
            d1 = -M1 / M1p
            d2 = -M1 / (M1p + d1 * M1pp / 2.0)
            d3 = -M1 / (M1p + d2 * M1pp / 2.0 + d2 * d2 * M1ppp / 6.0)
            E = E + d3
            E0 = E
            M0 = E0 - e*np.sin(E0)
            M1 = ( E0 - e * np.sin( E0 ) - M0)
            convergence_criteria = np.abs(M1) > criteria
            nd = np.sum(convergence_criteria is True)
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = K*(e*np.cos(w)+np.cos(w+nu)) + sys_vel
        return RV


##### f(x) = (x - a)**(-3) + b #################################################
class CubicSun(MeanModel):
    """
    A 3rd degree mean function
    m(t) = (t - xshift)**(-3) + yshift
    """
    _parsize = 4
    def __init__(self, xshift, yshift):
        super(Cubic, self).__init__(xshift, yshift)
        self.xshift = xshift
        self.yshift = yshift

    @array_input
    def __call__(self, t):
        return (t - self.xshift)**(-3) + self.yshift


### END

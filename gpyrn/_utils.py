"""
Collection of useful functions
"""
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import invgamma, rv_continuous
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import minimize
from random import shuffle

##### Semi amplitude calculation ##############################################
def semi_amplitude(period, Mplanet, Mstar, ecc):
    """
    Calculates the semi-amplitude (K) caused by a planet with a given
    period and mass Mplanet, around a star of mass Mstar, with a
    eccentricity ecc.

    Parameters
    ----------
    period: float
        Period in years
    Mplanet: float
        Planet's mass in Jupiter masses, tecnically is the M.sin i
    Mstar: float
        Star mass in Solar masses
    ecc: float
        Eccentricity between 0 and 1

    Returns
    -------
    float
        Semi-amplitude K
    """
    per = np.float(np.power(1/period, 1/3))
    Pmass = Mplanet / 1
    Smass = np.float(np.power(1/Mstar, 2/3))
    Ecc = 1 / np.sqrt(1 - ecc**2)
    return 28.435 * per * Pmass* Smass * Ecc


##### Keplerian function ######################################################
def keplerian(P=365, K=.1, e=0, w=np.pi, T=0, phi=None, gamma=0, t=None):
    """
    keplerian() simulates the radial velocity signal of a planet in a
    keplerian orbit around a star.

    Parameters
    ----------
    P: float
        Period in days
    K: float
        RV amplitude
    e: float
        Eccentricity
    w: float
        Longitude of the periastron
    T: float
        Zero phase
    phi: float
        Orbital phase
    gamma: float
        Constant system RV
    t: array
        Time of measurements

    Returns
    -------
    t: array
        Time of measurements
    RV: array
        RV signal generated
    """
    if t is  None:
        print()
        print('TEMPORAL ERROR, time is nowhere to be found')
        print()
    #mean anomaly
    if phi is None:
        mean_anom = [2*np.pi*(x1-T)/P  for x1 in t]
    else:
        T = t[0] - (P*phi)/(2.*np.pi)
        mean_anom = [2*np.pi*(x1-T)/P  for x1 in t]
    #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    E0 = [x + e*np.sin(x)  + 0.5*(e**2)*np.sin(2*x) for x in mean_anom]
    #mean anomaly -> M0=E0 - e*sin(E0)
    M0 = [x - e*np.sin(x) for x in E0]
    i = 0
    while i < 1000:
        #[x + y for x, y in zip(first, second)]
        calc_aux = [x2 - y for x2, y in zip(mean_anom, M0)]
        E1 = [x3 + y/(1-e*np.cos(x3)) for x3, y in zip(E0, calc_aux)]
        M1 = [x4 - e*np.sin(x4) for x4 in E0]
        i += 1
        E0 = E1
        M0 = M1
    nu = [2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(x5/2)) for x5 in E0]
    RV = [gamma + K*(e*np.cos(w)+np.cos(w+x6)) for x6 in nu] #m/s
    return t, RV


##### Phase-folding function ##################################################
def phase_folding(t, y, yerr, period):
    """
    phase_folding() allows the phase folding (duh...) of a given data
    accordingly to a given period

    Parameters
    ----------
    t: array
        Time
    y: array
        Measurements
    yerr: array
        Measurement errors
    period: float
        Period to fold the data

    Returns
    -------
    phase: array
        Phase
    folded_y: array
        Sorted measurments according to the phase
    folded_yerr:array
        Sorted errors according to the phase
    """
    #divide the time by the period to convert to phase
    foldtimes = t / period
    #remove the whole number part of the phase
    foldtimes = foldtimes % 1
    if yerr is None:
        yerr = 0 * y
    #sort everything
    phase, folded_y, folded_yerr = zip(*sorted(zip(foldtimes, y, yerr)))
    return phase, folded_y, folded_yerr


##### truncated cauchy distribution ###########################################
def truncCauchy_rvs(loc=0, scale=1, a=-1, b=1, size=None):
    """
    Generate random samples from a truncated Cauchy distribution.

    Parameters
    ----------
    loc: int
        Location parameter of the distribution
    scale: int
        Scale parameter of the distribution
    a, b: int
        Interval [a, b] to which the distribution is to be limited

    Returns
    -------
    rvs: float
        rvs of the truncated Cauchy
    """
    ua = np.arctan((a - loc)/scale)/np.pi + 0.5
    ub = np.arctan((b - loc)/scale)/np.pi + 0.5
    U = np.random.uniform(ua, ub, size=size)
    rvs = loc + scale * np.tan(np.pi*(U - 0.5))
    return rvs


##### inverse gamma distribution ###############################################
f = lambda x, lims: \
    (np.array([invgamma(a=x[0], scale=x[1]).cdf(lims[0]) - 0.01,
               invgamma(a=x[0], scale=x[1]).sf(lims[1]) - 0.01])**2).sum()

def invGamma(lower, upper, x0=[1, 5], showit=False):
    """
    Arguments
    ---------
    lower, upper : float
        The upper and lower limits between which we want 98% of the probability
    x0 : list, length 2
        Initial guesses for the parameters of the inverse gamma (a and scale)
    showit : bool
        Make a plot
    """
    limits = [lower, upper]
    result = minimize(f, x0=x0, args=limits, method='L-BFGS-B',
                      bounds=[(0, None), (0, None)], tol=1e-10)
    a, b = result.x
    if showit:
        _, ax = plt.subplots(1, 1, constrained_layout=True)
        d = invgamma(a=a, scale=b)
        x = np.linspace(0.2*limits[0], 2*limits[1], 1000)
        ax.plot(x, d.pdf(x))
        ax.vlines(limits, 0, d.pdf(x).max())
        plt.show()
    return invgamma(a=a, scale=b)


##### log sum ##################################################################
def log_sum(log_summands):
    """ log sum operation """
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        shuffle(x)
    return a


##### multivariate normal ######################################################
def multivariate_normal(r, c, method='cholesky'):
    """
    Computes multivariate normal density for "residuals" vector r and
    covariance c.

    :param array r:
        1-D array of k dimensions.

    :param array c:
        2-D array or matrix of (k x k).

    :param string method:
        Method used to compute multivariate density.
        Possible values are:
            * "cholesky": uses the Cholesky decomposition of the covariance c,
              implemented in scipy.linalg.cho_factor and scipy.linalg.cho_solve.
            * "solve": uses the numpy.linalg functions solve() and slogdet().

    :return array: multivariate density at vector position r.
    """
    # Compute normalization factor used for all methods.
    kk = len(r) * np.log(2*np.pi)
    if method == 'cholesky':
        # Use Cholesky decomposition of covariance.
        cho, lower = cho_factor(c)
        alpha = cho_solve((cho, lower), r)
        return -0.5 * (kk + np.dot(r, alpha) + 2 * np.sum(np.log(np.diag(cho))))
    if method == 'solve':
        # Use slogdet and solve
        (_, d) = np.linalg.slogdet(c)
        alpha = np.linalg.solve(c, r)
        return -0.5 * (kk + np.dot(r, alpha) + d)


##### RMS ######################################################################
def rms(array):
    """ Root mean square of array
        Parameters
        ----------
        array: array
            Measurements
            
        Returns
        -------
        rms: float
            Root mean squared error
    """
    mu = np.average(array)
    rms = np.sqrt(np.sum((array - mu)**2) / array.size)
    return rms

def wrms(array, weights):
    """ Weighted root mean square of array, given weights 
        
        Parameters
        ----------
        array: array
            Measurements
        weights: array
            weights = 1 / errors**2
            To add jitter do 1 / (errors*2 + jitter**2)
            
        Returns
        -------
        rms: float
            Weighted root mean squared error
    """
    mu = np.average(array, weights=weights)
    rms = np.sqrt(np.sum(weights * (array - mu)**2) / np.sum(weights)) 
    return rms

### END

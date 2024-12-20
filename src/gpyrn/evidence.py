"""
Computation of the evidence using the method developed by Perrakis et al. (2014)
"""
import random
from math import sqrt, log
import numpy as np
import scipy.stats
from gprn import utils

##### Original functions taken from https://github.com/exord/bayev #############
def compute_perrakis_estimate(marginal_sample, lnlikefunc, lnpriorfunc,
                              nsamples=1000, lnlikeargs=(), lnpriorargs=(),
                              densityestimation='histogram', 
                              errorestimation=False, **kwargs):
    """
    Computes the Perrakis estimate of the bayesian evidence.
    The estimation is based on n marginal posterior samples
    (indexed by s, with s = 0, ..., n-1).
    
    Parameters
    ----------
    :param array marginal_sample:
        A sample from the parameter marginal posterior distribution.
        Dimensions are (n x k), where k is the number of parameters.
    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.
    :param callable lnpriorfunc:
        Function to compute ln(prior density) on the marginal samples.
    :param nsamples:
        Number of samples to produce.
    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.
    :param tuple lnpriorargs:
        Extra arguments passed to the lnprior function.
    :param str densityestimation:
        The method used to estimate theinitial_samples marginal posterior density of each
        model parameter ("normal", "kde", or "histogram").
        
    Other parameters
    ----------------
    :param kwargs:
        Additional arguments passed to estimate_density function.
    :return:
        
    References
    ----------
    Perrakis et al. (2014; arXiv:1311.0674)
    
    Returns
    -------
    Perrakis estimate of the evidence
    """
    print('Estimating evidence...')
    if errorestimation:
        initial_sample = marginal_sample
    marginal_sample = make_marginal_samples(marginal_sample, nsamples)
    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)
    number_parameters = marginal_sample.shape[1]
    marginal_posterior_density = np.zeros(marginal_sample.shape)
    for parameter_index in range(number_parameters):
        x = marginal_sample[:, parameter_index]
        marginal_posterior_density[:, parameter_index] = \
            estimate_density(x, method=densityestimation, **kwargs)
    prod_marginal_densities = marginal_posterior_density.prod(axis=1)
    log_prior = lnpriorfunc(marginal_sample, *lnpriorargs)
    log_likelihood = lnlikefunc(marginal_sample, *lnlikeargs)
    cond = log_likelihood != 0
    log_summands = (log_likelihood[cond] + log_prior[cond] -
                    np.log(prod_marginal_densities[cond]))
    perr = log_sum(log_summands) - log(len(log_summands))
    #error estimation
    K = 10
    if errorestimation:
        batchSize = initial_sample.shape[0]//K
        meanErr = [_perrakis_error(initial_sample[0:batchSize, :],
                                   lnlikefunc, lnpriorfunc, nsamples=nsamples,
                                   densityestimation=densityestimation)]
        for i in range(K):
            meanErr.append(_perrakis_error(initial_sample[i*batchSize:(i+1)*batchSize, :],
                                           lnlikefunc, lnpriorfunc,
                                           nsamples=nsamples,
                                           densityestimation=densityestimation))
        stdErr = np.std(meanErr)
        meanErr = np.mean(meanErr)
        print(perr, stdErr)
        return perr, stdErr
    return perr


def _perrakis_error(marginal_samples, lnlikefunc, lnpriorfunc, nsamples=1000,
                    densityestimation='histogram', errorestimation=False):
    """ To use when estimating the error of the perrakis method """
    return compute_perrakis_estimate(marginal_samples, lnlikefunc, lnpriorfunc,
                                     nsamples=nsamples,
                                     densityestimation=densityestimation,
                                     errorestimation=errorestimation)


def _errorCalc(marginal_sample, lnlikefunc, lnpriorfunc, nsamples=300,
               densityestimation='histogram', **kwargs):
    print('Estimating evidence error...')
    marginal_sample = make_marginal_samples(marginal_sample, nsamples)
    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)
    number_parameters = marginal_sample.shape[1]
    #Estimate marginal posterior density for each parameter.
    marginal_posterior_density = np.zeros(marginal_sample.shape)
    for parameter_index in range(number_parameters):
        #Extract samples for this parameter._perrakis_error(
        x = marginal_sample[:, parameter_index]
        #Estimate density with method "densityestimation".
        marginal_posterior_density[:, parameter_index] = \
            estimate_density(x, method=densityestimation, **kwargs)
    #Compute produt of marginal posterior densities for all parameters
    prod_marginal_densities = marginal_posterior_density.prod(axis=1)
    #Compute lnprior and likelihood in marginal sample.
    log_prior = lnpriorfunc(marginal_sample)
    log_likelihood = lnlikefunc(marginal_sample)
    #Mask values with zero likelihood (a problem in lnlike)
    cond = log_likelihood != 0
    log_summands = (log_likelihood[cond] + log_prior[cond] -
                    np.log(prod_marginal_densities[cond]))
    perr = log_sum(log_summands) - log(len(log_summands))
    return perr


def estimate_density(x, method='histogram', **kwargs):
    """
    Estimate probability density based on a sample. Return value of density at
    sample points.
    :param array_like x: sample.
    :param str method:
        Method used for the estimation. 'histogram' estimates the density based
        on a normalised histogram of nbins bins; 'kde' uses a 1D non-parametric
        gaussian kernel; 'normal approximates the distribution by a normal
        distribution.
    Additional parameters
    :param int nbins:
        Number of bins used in "histogram method".
    :return: density estimation at the sample points.
    """
    nbins = kwargs.pop('nbins', 100)
    if method == 'normal':
        #Approximate each parameter distribution by a normal.
        return scipy.stats.norm.pdf(x, loc=x.mean(), scale=sqrt(x.var()))
    if method == 'kde':
        #Approximate each parameter distribution using a gaussian kernel estimation
        return scipy.stats.gaussian_kde(x)(x)
    if method == 'histogram':
        #Approximate each parameter distribution based on the histogram
        density, bin_edges = np.histogram(x, nbins, density=True)
        #Find to which bin each element corresponds
        density_indexes = np.searchsorted(bin_edges, x, side='left')
        #Correct to avoid index zero from being assiged to last element
        density_indexes = np.where(density_indexes > 0, density_indexes,
                                   density_indexes + 1)
        return density[density_indexes - 1]


def make_marginal_samples(joint_samples, nsamples=None):
    """
    Reshuffles samples from joint distribution of k parameters to obtain samples
    from the _marginal_ distribution of each parameter.
    :param np.array joint_samples:
        Samples from the parameter joint distribution. Dimensions are (n x k),
        where k is the number of parameters.
    :param nsamples:
        Number of samples to produce. If 0, use number of joint samples.
    :type nsamples:
        int or None
    """
    if nsamples > len(joint_samples) or nsamples is None:
        nsamples = len(joint_samples)
    marginal_samples = joint_samples[-nsamples:, :].copy()
    number_parameters = marginal_samples.shape[-1]
    # Reshuffle joint posterior samples to obtain _marginal_ posterior samples
    for parameter_index in range(number_parameters):
        random.shuffle(marginal_samples[:, parameter_index])
    return marginal_samples


def log_sum(log_summands):
    """ log_sum operation """
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        random.shuffle(x)
    return a


def compute_harmonicmean(lnlike_post, posterior_sample=None, lnlikefunc=None,
                         lnlikeargs=(), **kwargs):
    """
    Computes the harmonic mean estimate of the marginal likelihood.
    The estimation is based on n posterior samples
    (indexed by s, with s = 0, marginal likelihood error..., n-1), but can be done directly if the
    log(likelihood) in this sample is passed.
    :param array lnlike_post:
        log(likelihood) computed over a posterior sample. 1-D array of length n.
        If an emply array is given, then compute from posterior sample.
    :param array posterior_sample:
        A sample from the parameter posterior distribution.
        Dimensions are (n x k), where k is the number of parameters. If None
        the computation is done using the log(likelihood) obtained from the
        posterior sample.
    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.
    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.
    Other parameters
    ----------------
    :param int size:
        Size of sample to use for computation. If none is given, use size of
        given array or posterior sample.
    References
    ----------
    Kass & Raftery (1995), JASA vol. 90, N. 430, pp. 773-795
    """
    if len(lnlike_post) == 0 and posterior_sample is not None:
        samplesize = kwargs.pop('size', len(posterior_sample))
        if samplesize < len(posterior_sample):
            posterior_subsample = np.random.choice(posterior_sample,
                                                   size=samplesize,
                                                   replace=False)
        else:
            posterior_subsample = posterior_sample.copy()
        #Compute log likelihood in posterior sample.
        log_likelihood = lnlikefunc(posterior_subsample, *lnlikeargs)
    elif len(lnlike_post) > 0:
        samplesize = kwargs.pop('size', len(lnlike_post))
        log_likelihood = np.random.choice(lnlike_post, size=samplesize,
                                          replace=False)
    hme = -log_sum(-log_likelihood) + log(len(log_likelihood))
    return hme


def run_hme_mc(log_likelihood, nmc, samplesize):
    """ Harmonic mean """
    hme = np.zeros(nmc)
    for i in range(nmc):
        hme[i] = compute_harmonicmean(log_likelihood, size=samplesize)
    return hme


def compute_cj_estimate(posterior_sample, lnlikefunc, lnpriorfunc,
                        param_post, nsamples, qprob=None, lnlikeargs=(),
                        lnpriorargs=(), lnlike_post=None, lnprior_post=None):
    """
    Computes the Chib & Jeliazkov estimate of the bayesian evidence.
    The estimation is based on an posterior sample with n elements
    (indexed by s, with s = 0, ..., n-1), and a sample from the proposal
    distribution used in MCMC (qprob) of size nsample. Note that if qprob is
    None, it is estimated as a multivariate Gaussian.
    :param array posterior_sample:
        A sample from the parameter posterior distribution. Dimensions are
        (n x k), where k is the number of parameters.
    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.
    :param callable lnpriorfunc:
        Function to compute ln(prior density) on the marginal samples.
    :param array param_post:
        Posterior parameter sample used to obtained fixed point needed by the
        algorithm.
    :param int nsamples:
        Size of sample drawn from proposal distribution.
    :param object or None qprob:
        Proposal distribution function. If None, it will be estimated as a
        multivariate Gaussian. If not None, it must possess the methods pdf and
        rvs. See scipy.stats.rv_continuous.
    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.
    :param tuple lnpriorargs:
        Extra arguments passed to the lnprior function.
    :param array lnlike_post:
        log(likelihood) computed over a posterior sample. 1-D array of length n.
    :param array lnprior_post:
        log(prior) computed over a posterior sample. 1-D array of length n.
    :raises AttributeError:
        if instace qprob does not have method 'pdf' or 'rvs'.
    :raises TypeError:
        if methods 'pdf' or 'rvs' from instance qprob are not callable.
    :returns: Natural logarithm of estimated Bayesian evidence.
    References
    ----------
    Chib & Jeliazkov (2001): Journal of the Am. Stat. Assoc.; Mar 2001; 96, 453
    """
    #Find fixed point on which to estimate posterior ordinate.
    if lnlike_post is not None:
        #Pass values of log(likelihood) in posterior sample.
        arg_fp = [lnlike_post, ]
    else:
        #Pass function that computes log(likelihood).
        arg_fp = [lnlikefunc, ]
    if lnlike_post is not None:
        #Pass values of log(prior) in posterior sample.
        arg_fp.append(lnprior_post)
    else:
        #Pass function that computes log(prior).
        arg_fp.append(lnpriorfunc)
    fp, lnpost0 = get_fixed_point(posterior_sample, param_post, lnlikefunc, lnpriorfunc,
                                  lnlikeargs=lnlikeargs,
                                  lnpriorargs=lnpriorargs)
    #If proposal distribution is not given, define as multivariate Gaussian.
    if qprob is None:
        #Get covariance from posterior sample
        k = np.cov(posterior_sample.T)
        qprob = utils.MultivariateGaussian(fp, k)
    else:
        #Check that qprob has the necessary attributes
        for method in ('pdf', 'rvs'):
            try:
                att = getattr(qprob, method)
            except AttributeError:
                raise AttributeError('qprob does not have method '
                                     '\'{}\''.format(method))
            if not callable(att):
                raise TypeError('{} method of qprob is not '
                                'callable'.format(method))
    #Compute proposal density in posterior sample
    q_post = qprob.pdf(posterior_sample)
    #If likelihood over posterior sample is not given, compute it
    if lnlike_post is None:
        lnlike_post = lnlikefunc(posterior_sample, *lnlikeargs)
    #Idem for prior
    if lnprior_post is None:
        lnprior_post = lnpriorfunc(posterior_sample, *lnpriorargs)
    #Compute Metropolis ratio with respect to fixed point over posterior sample
    lnalpha_post = metropolis_ratio(lnprior_post + lnlike_post, lnpost0)
    #Sample from the proposal distribution with respect to fixed point
    proposal_sample = qprob.rvs(nsamples)
    #Compute likelihood and prior on proposal_sample
    lnprior_prop = lnpriorfunc(proposal_sample, *lnpriorargs)
    if np.all(lnprior_prop == -np.inf):
        raise ValueError('All samples from proposal density have zero prior'
                         'probability. Increase nsample.')
    #Now compute likelihood only on the samples where prior != 0.
    lnlike_prop = np.full_like(lnprior_prop, -np.inf)
    ind = lnprior_prop != -np.inf
    lnlike_prop[ind] = lnlikefunc(proposal_sample[ind, :], *lnlikeargs)
    #Get Metropolis ratio with respect to fixed point over proposal sample
    lnalpha_prop = metropolis_ratio(lnpost0, lnprior_prop + lnlike_prop)
    #Compute estimate of posterior ordinate (see Eq. 9 from reference)
    num = log_sum(lnalpha_post + q_post) - log(len(posterior_sample))
    den = log_sum(lnalpha_prop) - log(len(proposal_sample))
    lnpostord = num - den
    #Return log(Evidence) estimation
    return lnpost0 - lnpostord


def metropolis_ratio(lnpost0, lnpost1):
    """
    Compute Metropolis ratio for two states.
    :param float or array lnpost0:
        Value of ln(likelihood * prior) for inital state.
    :param float or array lnpost1:
        Value of ln(likelihood * prior) for proposal state.
    :raises ValueError: if lnpost0 and lnpost1 have different lengths.
    :return: log(Metropolis ratio)
    """
    if (hasattr(lnpost0, '__iter__') and hasattr(lnpost1, '__iter__') and
            len(lnpost0) != len(lnpost1)):
        raise ValueError('lnpost0 and lnpost1 have different lenghts.')
    return np.minimum(lnpost1 - lnpost0, 0.0)


def get_fixed_point(posterior_samples, param_post, lnlike, lnprior, lnlikeargs=(), lnpriorargs=()):
    """
    Find the posterior point closest to the model of the lnlike distribution.
    :param array posterior_samples:
        A sample from the parameters posterior distribution. Array dimensions
        must be (n x k), where n is the number of elements in the sample and
        k is the number of parameters.
    :param array or None param_post:
        A sample from the marginal posterior distribution of the parameter
        chosen to identify the high-density point to use as fixed point. This is
        typically one of the columns of posterior_samples, but could be any
        1-D array of size n. If None, then a multivariate Gaussian kernel
        estimate of the joint posterior distribution is used.
    :param array or callable lnlike:
        Function to compute log(likelihood). If an array is given, this is
        simply the log(likelihood) values at the posterior samples, and the
         best value will be chosen from this array.
    :param array or callable lnprior:
        Function to compute log(prior). If an array is given, this is
        simply the log(prior) values at the posterior samples, and the
        best value will be chosen from this array.
    :param tuple lnlikeargs:
        Extra arguments passed to lnlike functions.
    :param tuple lnpriorargs:
        Extra arguments passed to lnprior functions.
    :raises IndexError: if either lnlike or lnprior are arrays with length not
        matching the number of posterior samples.
    :return:
        the fixed point in parameter space and the value of
        log(prior * likelihood) evaluated at this point.
    """
    if param_post is not None:
        #Use median of param_post as fixed point.
        param0 = np.median(param_post)
        #Find argument closest to median.
        ind0 = np.argmin(np.abs(param_post - param0))
        fixed_point = posterior_samples[ind0, :]
        #Compute log(likelihood) at fixed_point
        if hasattr(lnlike, '__iter__'):
            if len(lnlike) != len(posterior_samples):
                raise IndexError('Number of elements in lnlike array and in '
                                 'posterior sample must match.')
            lnlike0 = lnlike[ind0]
        else:
            #Evaluate lnlike function at fixed point.
            lnlike0 = lnlike(fixed_point, *lnlikeargs)
        #Compute log(prior) at fixed_point
        if hasattr(lnprior, '__iter__'):
            if len(lnprior) != len(posterior_samples):
                raise IndexError('Number of elements in lnprior array and in '
                                 'posterior sample must match.')
            lnprior0 = lnprior[ind0]
        else:
            #Evaluate lnlike function at fixed point.
            lnprior0 = lnprior(fixed_point, *lnpriorargs)
        return fixed_point, lnlike0 + lnprior0
    raise NotImplementedError


### END

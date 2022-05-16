from functools import partial
from itertools import chain
import time as time_module
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve as cho_solve_jax
jax.config.update("jax_enable_x64", True)

from . import _gp, covfunc, meanfunc
from ._utils import Array, _array_input

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from emcee import EnsembleSampler, backends
from emcee.utils import sample_ellipsoid

# from numba import njit

from ._plots import plot_prediction

# def _cholNugget(matrix, maximum=1000):
#     """
#     Returns the cholesky decomposition to a given matrix, if it is not
#     positive definite, a nugget is added to its diagonal.

#     Parameters
#     ----------
#     matrix: array
#         Matrix to decompose
#     maximum: int
#         Number of times a nugget is added.

#     Returns
#     -------
#     L: array
#         Matrix containing the Cholesky factor
#     nugget: float
#         Nugget added to the diagonal
#     """
#     nugget = 0  # our nugget starts as zero
#     try:
#         nugget += 1e-15
#         L = cholesky(matrix, lower=True, overwrite_a=True)
#         return L, nugget
#     except LinAlgError:
#         n = 0 # number of tries
#         while n < maximum:
#             try:
#                 L = cholesky(matrix + nugget * np.identity(matrix.shape[0]),
#                              lower=True, overwrite_a=True)
#                 return L, nugget
#             except LinAlgError:
#                 nugget *= 10.0
#             finally:
#                 n += 1
#         raise LinAlgError("Not positive definite, even with nugget.")


def comp_results(a, b):
    if not np.allclose(a, b):
        print(a,b)
        raise Exception



# @njit
@jax.jit
def _cholNugget(matrix):
    """
    Returns the cholesky decomposition to a given matrix.

    Parameters
    ----------
    matrix: array
        Matrix to decompose

    Returns
    -------
    L: array
        Matrix containing the Cholesky factor
    nugget: float
        Nugget added to the diagonal
    """
    return jnp.linalg.cholesky(matrix), 0.0
    #  + 1.25e-6 * np.eye(matrix.shape[0])), 1.25e-6


class inference:
    """
    Mean-field variational inference for GPRNs.
    See Nguyen & Bonilla (2013) for more information.

    Args:
        q: int
            Number of latent node functions f(x)
        time: array
            Time coordinates
        *args: arrays
            The observed data in the following order:
                y1, y1error, y2, y2error, ...
    """
    def __init__(self, q: int, time: Array, *args):
        self.q = q
        self.time = time
        self.N = self.time.size

        # check if the input was correct
        msg = 'Number of observed data arrays should be even: y1, y1error, ...'
        assert len(args) > 0 and len(args) % 2 == 0, msg
        msg = 'Output arrays should all have the same dimensions as time'
        assert np.all(np.array([len(a) for a in args]) == self.N), msg

        # number of outputs
        self.p = int(len(args) / 2)
        # total number of weights
        self.qp = self.q * self.p
        self.d = self.N * self.q * (self.p + 1)

        # to organize the data, we now join everything
        self.tt = np.tile(time, self.p) # "extended" time
        self.y = np.concatenate([args[::2]])
        self.yerr = np.concatenate([args[1::2]])
        self.yerr2 = self.yerr**2

        self._components_set = False
        self._frozen_mask = np.array([])
        self._mu, self._var = None, None
        self._mu_var_iters = 0
        self.update_muvar_after = 50
        self.elbo_max_iter = 5000

    def set_components(self, nodes, weights, means, jitters):
        """
        Set the different GPRN components: nodes, weights, means, and jitters

        Args:
            nodes: `covFunction` or list of `covFunction`
                The q GPRN nodes. The number should match what was provided when
                creating this `inference`
            weights: `covFunction` or list of `covFunction`
                The q x p GPRN weights. The number should match the number of
                nodes times the number of datasets
            means: `meanFunction` or list of `meanFunction`
                The p GPRN mean functions
            jitters: float or list
                The p jitter values
        """
        if isinstance(nodes, covfunc.covFunction):
            nodes = [nodes]
        # check number
        if len(nodes) != self.q:
            msg = 'Wrong number of nodes provided, '
            msg += f'expected {self.q} got {len(nodes)}'
            raise ValueError(msg)

        if isinstance(weights, covfunc.covFunction):
            weights = [weights]
        # check number
        if len(weights) != self.qp:
            msg = 'Wrong number of weights provided, '
            msg += f'expected {self.qp} got {len(weights)}'
            raise ValueError(msg)

        if isinstance(means, (int, float, meanfunc.meanFunction)):
            means = [means]

        if isinstance(jitters, (int, float)):
            jitters = [jitters]

        self.nodes = nodes
        self.weights = weights
        self.means = means
        self.jitters = np.array(jitters, dtype=np.float)
        self._components_set = True

    def get_parameters(self, nodes=None, weights=None, means=None,
                       jitters=None, include_frozen=False):
        """
        Get the values of all the GPRN parameters
        """
        nones = [
            nodes is None, weights is None, means is None, jitters is None
        ]
        if not self._components_set and all(nones):
            msg = 'Cannot get parameters. '
            msg += 'Provide arguments or run set_components before.'
            raise ValueError(msg)

        if self._components_set:
            p = []
            for node in self.nodes:
                p.append(node.get_parameters())
            for weight in self.weights:
                p.append(weight.get_parameters())
            for mean in self.means:
                p.append(mean.get_parameters())
            for jitter in self.jitters:
                p.append(np.array([jitter]))
        else:
            p = []
            if nodes is not None:
                for node in nodes:
                    p.append(node.get_parameters())
            if weights is not None:
                for weight in weights:
                    p.append(weight.get_parameters())
            if means is not None:
                for mean in means:
                    p.append(mean.get_parameters())
            if jitters is not None:
                for jitter in jitters:
                    p.append(np.array([jitter]))

        if include_frozen:
            return np.concatenate(p).ravel()
        else:
            return np.concatenate(p).ravel()[~self.frozen_mask]

    @_array_input
    def set_parameters(self, parameters: Array):
        """
        Set values for all the GPRN parameters
        """
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg
        all_parameters = self.get_parameters(include_frozen=True)
        n_free_parameters = self.n_parameters - self.frozen_mask.sum()

        if parameters.size == self.n_parameters:
            # all parameters provided, even if some may be frozen
            # we ignore the frozen ones
            parameters[self.frozen_mask] = all_parameters[self.frozen_mask]

        elif parameters.size == n_free_parameters:
            # only non-frozen parameters were provided, fill in the frozen ones
            for i, par in enumerate(all_parameters):
                if self.frozen_mask[i]:
                    parameters = np.insert(parameters, i, par)

        else:
            # wrong numer of parameters provided
            NP = parameters.size
            ep = self.n_parameters
            fp = n_free_parameters
            msg = f'Wrong number of parameters provided: got {NP}, '
            if ep == fp:
                msg += f'expected {ep}'
            else:
                msg += f'expected {ep} (all) or {fp} (not frozen)'
            raise ValueError(msg)

        it = [self.nodes, self.weights, self.means]
        for component in chain.from_iterable(it):
            parameters = component.set_parameters(parameters)
        self.jitters = parameters

    @property
    def n_parameters(self):
        """ Total number of parameters """
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg
        n = 0
        it = [self.nodes, self.weights, self.means]
        for component in chain.from_iterable(it):
            n += component.pars.size
        n += self.jitters.size
        return n

    @property
    def parameters_dict(self):
        """ Dictionary with parameters names and values """
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg

        p = {}
        for i, node in enumerate(self.nodes, start=1):
            for par, val in zip(node._param_names, node.pars):
                p[f'node{i}.{par}'] = val
        for i, weight in enumerate(self.weights, start=1):
            for par, val in zip(weight._param_names, weight.pars):
                p[f'weight{i}.{par}'] = val
        for i, mean in enumerate(self.means, start=1):
            for par, val in zip(mean._param_names, mean.pars):
                p[f'mean{i}.{par}'] = val
        for i, jit in enumerate(self.jitters, start=1):
            p[f'jitter{i}'] = jit
        return p

    def freeze_parameter(self, index=None, name=None):
        """
        Freeze (do not fit for) a given parameter, by index or name.

        Args:
            index: int, array
                Index (or indices) of the parameters to freeze
            name: str
                Name of the parameter to freeze. Including a "*" in `name`
                allows to freeze more than one parameter at once. For example,
                name='node1*' will freeze all parameters of the first node.
        """
        self.frozen_mask
        if index is None and name is None:
            raise ValueError('Provide either index or name')
        if name is None:
            self._frozen_mask[index] = True
        elif index is None:
            if '*' in name:
                names = list(self.parameters_dict.keys())
                name = name.replace('*', '')
                for index, known_name in enumerate(names):
                    if name in known_name:
                        self._frozen_mask[index] = True
            else:
                msg = f'Name "{name}" not found in parameters_dict'
                assert name in self.parameters_dict, msg
                index = list(self.parameters_dict.keys()).index(name)
                self._frozen_mask[index] = True

    def freeze_all_parameters(self):
        """ Freeze (do not fit for) all parameters """
        self._frozen_mask = np.ones(self._frozen_mask.size, dtype=bool)

    fix_parameter = freeze_parameter
    fix_all_parameters = freeze_all_parameters

    def thaw_parameter(self, index=None, name=None):
        """
        Thaw (free) a given parameter, by index or name.

        Args:
            index: int, array
                Index (or indices) of the parameters to thaw
            name: str
                Name of the parameter to thaw. Including a "*" in `name`
                allows to freeze more than one parameter at once. For example,
                name='node1*' will thaw all parameters of the first node.
        """
        self.frozen_mask
        if index is None and name is None:
            raise ValueError('Provide either index or name')
        if name is None:
            self._frozen_mask[index] = False
        elif index is None:
            if '*' in name:
                names = list(self.parameters_dict.keys())
                name = name.replace('*', '')
                for index, known_name in enumerate(names):
                    if name in known_name:
                        self._frozen_mask[index] = False
            else:
                msg = f'Name "{name}" not found in parameters_dict'
                assert name in self.parameters_dict, msg
                index = list(self.parameters_dict.keys()).index(name)
                self._frozen_mask[index] = False

    def thaw_all_parameters(self):
        """ Thaw (free) all parameters """
        self._frozen_mask = np.zeros(self._frozen_mask.size, dtype=bool)

    free_parameter = thaw_parameter
    free_all_parameters = thaw_all_parameters

    @property
    def frozen_mask(self):
        """ Boolean mask for the frozen parameters """
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg
        if self._frozen_mask.size == 0:
            self._frozen_mask = np.full(self.n_parameters, False, dtype=bool)
        return self._frozen_mask

    @frozen_mask.setter
    def frozen_mask(self, mask):
        msg = 'Do not set frozen_mask, use thaw_parameter/freeze_parameter'
        raise NotImplementedError(msg)


    def _mean(self, means, time=None):
        """
        Returns the values of the mean functions

        Args:
            means : list of instances of meanFunction
            time : array, optional

        Returns:
            m: array
                Mean function evaluated at `time` or `self.time`
        """
        if time is None:
            N = self.time.size
            m = np.zeros_like(self.tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i * N:(i + 1) * N] = meanfun(self.time)
        else:
            N = time.size
            tt = np.tile(time, self.p)
            m = np.zeros_like(tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i * N:(i + 1) * N] = meanfun(time)
        return m

    def _KMatrix(self, kernel, time=None):
        """
        Returns the covariance matrix created by evaluating a given kernel at
        inputs time. For stability, a 1e-6 nugget is added to the diagonal.

        Args:
            kernel : instance of covFunc
            time : array, optional
        
        Returns:
            K: array
                Matrix of a covariance function
        """
        r = time[:, None] - time[None, :]
        K = kernel(r) + 1e-6 * np.eye(time.size)
        return K

    def _tinyNuggetKMatrix(self, kernel, time=None):
        """
        To be used in Prediction(). Returns the covariance matrix created by
        evaluating a given kernel at inputs time with the tiniest stability
        nugget possible.

        Returns:
            K: array Matrix of a covariance function
        """
        r = time[:, None] - time[None, :]
        K = kernel(r) + 1.25e-12 * np.diag(np.diag(np.ones_like(r)))
        return K

    def _predictKMatrix(self, kernel, time):
        """
        To be used in Prediction()

        Args:
            kernel : instance of covfunc
            time : array, optional

        Returns:
            K: array
                Matrix of a covariance function
        """
        if time.size == 1:
            r = time - self.time[None, :]
        else:
            r = time[:, None] - self.time[None, :]
        return kernel(r)

    def _u_to_fhatW(self, u):
        """
        Transform from the concatenated u values to the corresponding nodes (f)
        and weights (w).

        Args:
            u: array

        Returns:
            f: array
                Samples of the nodes
            w: array
                Samples of the weights
        """
        f = u[:self.q * self.N].reshape((1, self.q, self.N))
        w = u[self.q * self.N:].reshape((self.p, self.q, self.N))
        return f, w

    def _initMuVar(self, nodes, weights, jitter):
        a1 = [n.pars[0] for n in nodes]
        a2 = [w.pars[0] for w in weights]
        mean1, mean2 = [], []
        var1, var2 = [], []
        for _, n in enumerate(a1):
            m = [
                np.sqrt(np.abs(j) * n / i) * np.sign(j)
                for i, j in zip(a2, self.y)
            ]
            mean1.append(np.mean(m, axis=0))
            mean2.append(
                [np.sqrt(np.abs(j) * i / n) for i, j in zip(a2, self.y)])

            var1.append([np.mean(jitter) * np.ones_like(self.time)])
            var2.append([jitt * np.ones_like(self.time) for jitt in jitter])

        mu = np.concatenate((mean1, mean2), axis=None)
        var = np.concatenate((var1, var2), axis=None)
        return mu, var

    def _randomMuVar(self):
        mu = np.random.randn(self.d, 1)
        var = np.random.rand(self.d, 1)
        return mu, var

    def _sample_from_gp(self, kernel, time=None):
        """
        Returns random samples from a given kernel

        Args:
            kernel: covFunction
            time: array
        """
        if time is None:
            time = self.time
        mean = np.zeros_like(time)
        K = self._tinyNuggetKMatrix(kernel, time)
        normal = multivariate_normal(mean, K, allow_singular=True).rvs()
        return normal

    def sample(self, time=None):
        nodes, weights, means, jitters = self._get_components()
        node_samples = np.array([self._sample_from_gp(node) for node in nodes])
        weight_samples = np.array(
            [self._sample_from_gp(weight) for weight in weights])
        print(node_samples.shape)
        print(weight_samples.shape)
        return node_samples, weight_samples

    def _get_components(self, nodes=None, weights=None, means=None,
                        jitters=None):
        # if nothing was given, componentes must be already set
        all_none = all([i is None for i in (nodes, weights, means, jitters)])
        msg = 'GPRN components not set, use set_components'
        if all_none and not self._components_set:
            raise ValueError(msg)

        nodes = self.nodes if nodes is None else nodes
        weights = self.weights if weights is None else weights
        means = self.means if means is None else means
        jitters = self.jitters if jitters is None else jitters
        return nodes, weights, means, jitters


    @property
    def ELBO(self):
        """ The evidence lower bound for the GPRN """
        return self.ELBOcalc()[0]

    def ELBOcalc(self, nodes=None, weights=None, means=None, jitters=None,
                 max_iter=None, mu=None, var=None):
        """
        Calculate the evidence lower bound

        Args:
            nodes: list of `covFunction` instances, optional
                Kernel(s) for the node(s)
            weights: list of `covFunction` instances, optional
                Kernel(s) for the weight(s)
            means: list of `meanFunction` instances, optional
                Mean functions
            jitters: list of floats, optional
                Jitter terms
            max_iter: int, default self.elbo_max_iter
                Maximum number of iterations allowed in ELBO calculation
            mu: array or str, optional
                Variational means or 'init', 'random', or 'previous'
            var: array or str, optional
                Variational variances or 'init', 'random', or 'previous'

        Returns:
            ELBO: array
                Value of the ELBO per iteration
            mu: array
                Optimized variational means
            var: array
                Optimized variational variance (diagonal of sigma)
            """
        # deal with inputs or get attributes
        nodes, weights, means, jitters = self._get_components(
            nodes, weights, means, jitters)

        # initial variational parameters
        if mu is None or var is None:
            mu = var = 'init'

        if mu == 'previous' or var == 'previous':
            # should_update = self._mu_var_iters > self.update_muvar_after
            # if should_update:
            # mu, var = self._randomMuVar()
            # self._mu_var_iters = 0
            if self._mu is not None:
                mu, var = self._mu, self._var
                # self._mu_var_iters += 1
            else:
                mu, var = self._initMuVar(nodes, weights, jitters)

        elif mu == 'random' and var == 'random':
            mu, var = self._randomMuVar()

        elif mu == 'init' and var == 'init':
            mu, var = self._initMuVar(nodes, weights, jitters)

        if max_iter is None:
            max_iter = 10000

        j2 = np.array(jitters)**2
        Kf = np.array([self._KMatrix(i, self.time) for i in nodes])
        Kw = np.array([self._KMatrix(j, self.time) for j in weights])
        Lf = np.array([_cholNugget(j)[0] for j in Kf])
        Lw = np.array([_cholNugget(j)[0] for j in Kw])
        y = np.concatenate(self.y) - self._mean(means)
        y = np.array(np.array_split(y, self.p))

        # To add new elbo values inside
        ELBO, *_ = self.ELBOaux(Kf, Kw, Lf, Lw, y, j2, mu, var)
        elboArray = np.array([ELBO])
        iterNumber = 0

        if max_iter is None:
            max_iter = self.elbo_max_iter

        while iterNumber < max_iter:
            # Optimize mu and var analytically
            ELBO, mu, var, _, _ = self.ELBOaux(Kf, Kw, Lf, Lw, y, j2, mu, var)
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            # Stoping criteria:
            if iterNumber > 3:
                means = np.mean(elboArray[-3:])
                criteria = np.abs(np.std(elboArray[-3:]) / means)
                if criteria < 1e-3 and criteria != 0:
                    self._mu = mu
                    self._var = var
                    return ELBO, mu, var, iterNumber

        print('\nMax iterations reached')
        return ELBO, mu, var, iterNumber

    def ELBOaux(self, Kf, Kw, Lf, Lw, y, jitt2, mu, var):
        """
        Evidence Lower bound to use in ELBOcalc()

        Args:
            Kf: array
                Covariance matrices of the node functions 
            Kw: array
                Covariance matrices of the weight function
            Lf: array
                Lower matrix calculated with Cholesky of Kf 
            Lw: array
                Lower matrix calculated with Cholesky of Kw
            y: array
                Measurements - means
            jitt2: array
                Squared jitter terms
            mu: array
                Variational means
            var: array
                Variational variances

        Returns:
            ELBO: float
                Evidence lower bound
            new_mu: array
                New variational means
            new_var: array
                New variational variances
        """
        #to separate the variational parameters between the nodes and weights
        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())
        sigmaF, muF, sigmaW, muW = self._updateSigMu(Kf, Kw, Lf, Lw, y, jitt2,
                                                     muF, varF, muW, varW)

        # new mean and var for the nodes
        muF = muF.reshape(1, self.q, self.N)
        varF = np.zeros_like(varF)
        for i in range(self.q):
            varF[:, i, :] = np.diag(sigmaF[i, :, :])

        # new mean and var for the weights
        varW = np.zeros_like(varW)
        for j in range(self.q):
            for i in range(self.p):
                varW[i, j, :] = np.diag(sigmaW[j, i, :, :])

        new_mu = np.concatenate((muF, muW))
        new_var = np.concatenate((varF, varW))

        # Entropy
        Ent = self._entropy(sigmaF, sigmaW)
        # Expected log prior
        LogP = self._expectedLogPrior(Kf, Kw, Lf, Lw, sigmaF, muF, sigmaW, muW)
        # Expected log-likelihood
        LogL = self._expectedLogLike(y, jitt2, sigmaF, muF, sigmaW, muW)
        # Evidence Lower Bound
        ELBO = (LogL + LogP + Ent) / self.q
        return ELBO, new_mu, new_var, sigmaF, sigmaW

    # @partial(jax.jit, static_argnums=(0,))
    def _updateSigMu(self, Kf, Kw, Lf, Lw, y, jitt2, muF, varF, muW, varW):
        """
        Efficient closed-form updates fot variational parameters. This
        corresponds to eqs. 16, 17, 18, and 19 of Nguyen & Bonilla (2013)

        Args:
            Kf: array
                Covariance matrices of the node functions
            Kw: array
                Covariance matrices of the weight function
            y: array
                Measurements - means
            jitt2: array
                Squared jitter terms
            muF: array
                Initial variational mean of each node
            varF: array
                Initial variational variance of each node
            muW: array
                Initial variational mean of each weight
            varW: array
                Initial variational variance of each weight

        Returns:
            sigma_f: array
                Updated variational covariance of each node
            mu_f: array
                Updated variational mean of each node
            sigma_w: array
                Updated variational covariance of each weight
            mu_w: array
                Updated variational mean of each weight
        """

        compare_results = False

        Kw = Kw.reshape(self.q, self.p, self.N, self.N)
        Lw = Lw.reshape(self.q, self.p, self.N, self.N)
        muF = np.squeeze(muF)

        # creation of Sigma_fj and mu_fj

        sigma_f = np.empty((self.q, self.N, self.N))
        mu_f = np.empty((self.q, self.N))

        # shape: p x N
        variance = jitt2[:, None] + self.yerr2

        # muW  shape: p x q x N
        # varW shape: p x q x N
        # sum is over p, need to replicate variance over q axis
        #? precomputed for all nodes j (eq 20)
        diagonal_vector = np.sum((muW * muW + varW) / variance[:, None, :],
                                 axis=0)


        for j in range(self.q):
            # Woodbury matrix identity
            sigma_f[j] = Kf[j] - Kf[j] @ np.linalg.solve(np.diag((1 / diagonal_vector[j])) + Kf[j], Kf[j])
            # # Kf_invf = np.linalg.inv(Kf[j])
            # Kf_inv = cho_solve((Lf[j], True), np.eye(self.N))
            # # assert np.allclose(Kf_invf, Kf_inv)
            # #? einsum gives a writeable view
            # Kf_inv_diagonal = np.einsum('jj->j', Kf_inv)
            # Kf_inv_diagonal += diagonal_vector[j]
            # sigma_f[j] = np.linalg.inv(Kf_inv)

            # muW  shape: p x q x N
            # muF shape: q x N
            # sum is over q, except j
            # y shape: p x N
            # print(muW * muF)
            # print('a')
            # print(np.delete(muW * muF, j, axis=1))
            # print('b')
            residuals = y - np.sum(np.delete(muW * muF, j, axis=1), axis=1)
            # residuals shape: p x N --> p x q x N
            pred = np.sum(residuals * muW[:, j, :] / variance,
                          axis=0)
            mu_f[j] = sigma_f[j] @ pred

        if compare_results:
            sigma_f_og, mu_f_og = [], []
            for j in range(self.q):
                diagFj, auxCalc = 0, 0
                for i in range(self.p):
                    diagFj = diagFj + (muW[i,j,:]*muW[i,j,:]+varW[i,j,:]) / (jitt2[i] + self.yerr2[i,:])
                    sumNj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            sumNj += muW[i, k, :] * muF[k, :].reshape(self.N)
                    auxCalc = auxCalc + ((y[i, :] - sumNj)*muW[i,j,:]) / (jitt2[i] + self.yerr2[i, :])
                    # R = y[i, :] - sumNj
                    # print(R)
                    # print(RR[j][i])
                    # input()

                comp_results(diagFj, diagonal_vector[j])
                CovF = np.diag(1 / diagFj) + Kf[j]
                sigF = Kf[j] - Kf[j] @ np.linalg.solve(CovF, Kf[j])

                # print(auxCalc)
                # input()

                sigma_f_og.append(sigF)
                mu_f_og.append(sigF @ auxCalc)
                # muF = np.array(mu_f_og)
            sigma_f_og = np.array(sigma_f_og)
            mu_f_og = np.array(mu_f_og)
            comp_results(sigma_f, sigma_f_og)
            comp_results(mu_f, mu_f_og)

        # #creation of Sigma_wij and mu_wij
        sigma_w = np.empty((self.q, self.p, self.N, self.N))
        mu_w = np.empty((self.p, self.q, self.N))
        #! why mu_w is not q x p x N ???

        # Kw shape: q x p x N x N
        # sigma_f shape: q x N x N
        # mu_f shape: q x N
        # print(sigma_f.shape)
        # print(mu_f.shape)

        # print(np.einsum('ijj->ij', sigma_f).shape)
        #? this one is also a view but the + creates a copy
        diagonal_vector = mu_f * mu_f + np.einsum('ijj->ij', sigma_f)
        # print(diagonal_vector.shape)
        # print(variance.shape)
        # input()
        # diagonal_vector = diagonal_vector / variance

        # mu_w = np.empty((self.q, self.N))

        for j in range(self.q):
            residuals = y - np.sum(np.delete(mu_f * muW, j, axis=1), axis=1)

            for i in range(self.p):
                sigma_w[j, i] = Kw[j, i] - Kw[j, i] @ np.linalg.solve(np.diag((variance[i] / diagonal_vector[j])) + Kw[j,i], Kw[j,i])
                # # Kw_invf = np.linalg.inv(Kw[j, i])
                # Kw_inv = cho_solve((Lw[j, i], True), np.eye(self.N))
                # # print(Kw_invf, Kw_inv)
                # # assert np.allclose(Kw_invf, Kw_inv, rtol=0.01)

                # Kw_inv_diagonal = np.einsum('jj->j', Kw_inv)
                # Kw_inv_diagonal += diagonal_vector[j] / variance[i]
                # sigma_w[j, i] = np.linalg.inv(Kw_inv)

                #
                # sum is over q, except j
                # y shape: p x N
                # residuals shape: p x N --> p x q x N
                pred = residuals[i] * mu_f[j, :] / variance[i]
                mu_w[i, j] = sigma_w[j, i] @ pred


        if compare_results:
            sigma_w_og, mu_w_og = [], np.zeros_like(muW)
            for j in range(self.q):
                for i in range(self.p):
                    mu_fj = mu_f_og[j]
                    var_fj = np.diag(sigma_f_og[j])
                    Diag_ij = (mu_fj * mu_fj + var_fj) / (jitt2[i] + self.yerr2[i, :])
                    Kw_ij = Kw[j, i, :, :]
                    CovWij = np.diag(1 / Diag_ij) + Kw_ij
                    sigWij = Kw_ij - Kw_ij @ np.linalg.solve(CovWij, Kw_ij)
                    sigma_w_og.append(sigWij)
                    sumNj = np.zeros(self.N)
                    for k in range(self.q):
                        if k != j:
                            sumNj += mu_f_og[k].reshape(self.N) * np.array(muW[i, k, :])
                    auxCalc = ((y[i, :] - sumNj) * mu_f_og[j, :]) / (jitt2[i] + self.yerr2[i, :])
                    mu_w_og[i, j, :] = sigWij @ auxCalc

            sigma_w_og = np.array(sigma_w_og).reshape(self.q, self.p, self.N, self.N)
            # mu_w_og = np.array(muW)

            comp_results(sigma_w, sigma_w_og)
            comp_results(mu_w, mu_w_og)

        # input('all good!')
        return sigma_f, mu_f, sigma_w, mu_w

    @partial(jax.jit, static_argnums=(0,))
    def _expectedLogLike(self, y, jitt2, sigma_f, mu_f, sigma_w, mu_w):
        """
        Calculates the expected log-likelihood in mean-field inference,
        corresponds to eq.14 in Nguyen & Bonilla (2013)

        Args:
            y: array
                Measurements - means
            jitt2: array
                Squared jitter terms
            sigma_f: array
                Variational covariance for each node
            mu_f: array
                Variational mean for each node
            sigma_w: array
                Variational covariance for each weight
            mu_w: array
                Variational mean for each weight

        Returns:
            logl: float
                Expected log-likelihood
        """
        compare_results = False


        # shape: p x N
        variance = jitt2[:, None] + self.yerr2

        logl = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * variance))

        if compare_results:
            logl_og = 0
            for p in range(self.p):
                for n in range(self.N):
                    logl_og += np.log(2 * np.pi * (jitt2[p] + self.yerr2[p, n]))
            logl_og *= -0.5
            comp_results(logl, logl_og)

        # mu_f shape: 1 x q x N  (T: )
        # mu_w shape: p x q x N

        # shape: p x N                Nxqxp   Nxq
        Ωnu = jnp.einsum('ijk,ij->ik', mu_w.T, mu_f[0].T).T
        resid = self.y - Ωnu
        term2 = -0.5 * jnp.sum(resid**2 / variance)
        logl += term2

        if compare_results:
            sumN = []
            for n in range(self.N):
                for p in range(self.p):
                    Ydiff = y[p, n] - mu_f[0, :, n] @ mu_w[p, :, n].T
                    bottom = jitt2[p] + self.yerr2[p, n]
                    sumN.append((Ydiff.T * Ydiff) / bottom)
            term2_og = -0.5 * np.sum(sumN)
            comp_results(term2, term2_og)
            logl_og += term2_og


        # mu_f shape: 1 x q x N
        # mu_w shape: p x q x N
        #
        # sigma_f shape: q x N x N
        # sigma_w shape: q x p x N x N

        sigma_f_diagonals = jnp.einsum('ijj->ij', sigma_f)
        sigma_w_diagonals = jnp.einsum('ijkk->ijk', sigma_w)

        value = 0.0
        for i in range(self.p):
            for j in range(self.q):
                _1 = sigma_f_diagonals[j].T @ (mu_w[i, j]**2 / variance[i])
                _2 = sigma_w_diagonals[j, i].T @ (mu_f[0, j]**2 / variance[i])
                _3 = sigma_f_diagonals[j].T @ (sigma_w_diagonals[j, i] / variance[i])
                value += (_1 + _2 + _3)
        logl += -0.5 * value


        if compare_results:
            value_og = 0
            for p in range(self.p):
                for q in range(self.q):
                    value_og += np.sum(
                        (np.diag(sigma_f[q, :, :]) * mu_w[p, q, :] * mu_w[p, q, :]
                         + np.diag(sigma_w[q, p, :, :]) * mu_f[:, q, :] * mu_f[:, q, :]
                         + np.diag(sigma_f[q, :, :]) * np.diag(sigma_w[q, p, :, :])) / (jitt2[p] + self.yerr2[p, :])
                         )
            comp_results(value, value_og)
            logl_og += -0.5 * value_og

            comp_results(logl, logl_og)

            # input('all good')
        return logl

    @partial(jax.jit, static_argnums=(0,))
    def _expectedLogPrior(self, Kf, Kw, Lf, Lw, sigma_f, mu_f, sigma_w, mu_w):
        """
        Calculates the expection of the log prior wrt q(f,w) in mean-field
        inference, corresponds to eq.15 in Nguyen & Bonilla (2013)

        Args:
            Kf: array
                Covariance matrices of the node functions 
            Kw: array
                Covariance matrices of the weight function
            sigma_f: array
                Variational covariance for each node
            mu_f: array
                Variational mean for each node
            sigma_w: array
                Variational covariance for each weight
            mu_w: array
                Variational mean for each weight

        Returns:
            logp: float
                Expected log prior value
        """
        compare_results = False

        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        Kw = Kw.reshape(self.q, self.p, self.N, self.N)
        Lw = Lw.reshape(self.q, self.p, self.N, self.N)
        muW = mu_w.reshape(self.q, self.p, self.N)

        first_term = 0.0  #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0.0  #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        sumSigmaF = jnp.zeros_like(sigma_f[0])

        for j in range(self.q):
            Lfj = Lf[j]
            logKf = jnp.sum(jnp.log(jnp.diag(Lfj)))

            mu_reshaped = mu_f[0, j, :]
            muKmu = mu_reshaped.T @ cho_solve_jax((Lfj, True), mu_reshaped)

            if compare_results:
                muK = np.linalg.solve(Lfj, mu_f[:, j, :].reshape(self.N))
                muKmu_og = muK @ muK
                comp_results(muKmu, muKmu_og)

            sumSigmaF = sumSigmaF + sigma_f[j]

            trace = jnp.trace(cho_solve_jax((Lfj, True), sumSigmaF))

            if compare_results:
                trace_og = np.trace(np.linalg.solve(Kf[j], sumSigmaF))
                comp_results(trace, trace_og)

            first_term += -logKf - 0.5 * (muKmu + trace)

            for i in range(self.p):
                muKmu = muW[j, i].T @ cho_solve_jax((Lw[j, i, :, :], True), muW[j, i])
                trace = jnp.trace(cho_solve_jax((Lw[j,i,:,:], True), sigma_w[j,i,:,:]))


                if compare_results:
                    muK = np.linalg.solve(Lw[j,i,:,:], muW[j,i])
                    muKmu_og = muK @ muK
                    trace_og = np.trace(np.linalg.solve(Kw[j,i,:,:], sigma_w[j,i,:,:]))
                    comp_results(muKmu, muKmu_og)
                    comp_results(trace, trace_og)


                second_term += -jnp.sum(jnp.log(jnp.diag(Lw[j,i,:,:]))) - 0.5*(muKmu + trace)

        const = -0.5 * self.N * self.q * (self.p + 1) * jnp.log(2 * jnp.pi)
        logp = first_term + second_term + const

        return logp

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, sigma_f, sigma_w):
        """
        Calculates the entropy in mean-field inference, corresponds to eq.14 
        in Nguyen & Bonilla (2013)

        Args:
            sigma_f: array
                Variational covariance for each node
            sigma_w: array
                Variational covariance for each weight
        
        Returns:
            entropy: float
                Final entropy value
        """
        entropy = 0.0
        for j in range(self.q):
            L1 = _cholNugget(sigma_f[j])[0]
            entropy += jnp.sum(jnp.log(jnp.diag(L1)))
            for i in range(self.p):
                L2 = _cholNugget(sigma_w[j, i, :, :])[0]
                entropy += jnp.sum(jnp.log(jnp.diag(L2)))
        const = 0.5 * self.q * (self.p + 1) * self.N * (1 + jnp.log(2 * jnp.pi))
        return entropy + const

    def nELBO(self, parameters, max_iter=None):
        """ Return the negative ELBO for given values of the parameters """
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg
        self.set_parameters(parameters)

        start = time_module.time()
        elbo, _, _, _ = self.ELBOcalc(self.nodes, self.weights, self.means,
                                      self.jitters, max_iter=max_iter,
                                      mu='previous', var='previous')
        end = time_module.time()

        spaces = 20 * ' '
        print(f'ELBO={elbo:7.2f} (took {1e3*(end-start):5.2f} ms){spaces}',
              end='\r', flush=True)
        # print()
        return -elbo


    def optimize(self, vars=None, **kwargs):
        """
        Optimize (maximize) the ELBO. If provided, `vars` controls the
        parameters which are free during the optimization.

        Args:
            vars : str or list, optional
                If provided, this defines the parameters included in the
                optimization process. Options are
                    vars = 'parameter_name'
                        optimize *only* parameter_name, all others are fixed
                    vars = '-parameter_name'
                        optimize all parameters *except* parameter_name
                    vars = [list of parameter_names]
                        optimize parameter_names and hold the others fixed
            **kwargs : dict
                Keyword arguments passed directly to scipy.optimize.minimize
        """
        if vars is not None:
            if isinstance(vars, str):
                if '-' in vars:
                    vars = vars.replace('-', '')
                    self.thaw_parameter(name='*')  # thaw all
                    self.freeze_parameter(name=vars)  # freeze vars
                else:
                    self.freeze_parameter(name='*')  # freeze all
                    self.thaw_parameter(name=vars)  # thaw vars
            elif isinstance(vars, list):
                self.freeze_parameter(name='*')  # freeze all
                for var in vars:
                    self.thaw_parameter(name=var)  # except all vars
            else:
                msg = f'`vars` should be str or list, got {type(vars)}'
                raise ValueError(msg)

        kwargs.setdefault('method', 'Nelder-Mead')
        res = minimize(self.nELBO, self.get_parameters(), **kwargs)
        self.set_parameters(res.x)
        return res

    def mcmc(self, priors, p0=None, vars=None, niter=500, **kwargs):
        """
        Sample the posterior distribution for the GPRN parameters. If provided,
        `vars` controls the parameters which are free in the MCMC.

        Args:
            priors: dict
                Dictionary with prior distributions for all (free) parameters
            p0: array, optional
                Initial values for the parameters. If not provided, a random
                sample from the corresponding prior will be used.
            vars : str or list, optional
                If provided, this defines the parameters included in the
                sampling process. Options are
                    vars = 'parameter_name'
                        sample *only* parameter_name, all others are fixed
                    vars = '-parameter_name'
                        sample all parameters *except* parameter_name
                    vars = [list of parameter_names]
                        sample parameter_names and hold the others fixed
            niter: int
                Number of MCMC iterations 
            **kwargs : dict
                Keyword arguments passed directly to emcee.EnsembleSampler
        """
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg

        if vars is not None:
            if isinstance(vars, str):
                if '-' in vars:
                    vars = vars.replace('-', '')
                    self.thaw_parameter(name='*')  # thaw all
                    self.freeze_parameter(name=vars)  # freeze vars
                else:
                    self.freeze_parameter(name='*')  # freeze all
                    self.thaw_parameter(name=vars)  # thaw vars
            elif isinstance(vars, list):
                self.freeze_parameter(name='*')  # freeze all
                for var in vars:
                    self.thaw_parameter(name=var)  # except all vars
            else:
                msg = f'`vars` should be str or list, got {type(vars)}'
                raise ValueError(msg)

        all_parameter_names = np.array(list(self.parameters_dict.keys()))
        free_parameter_names = all_parameter_names[~self.frozen_mask]

        def prior_rvs():
            rvs = []
            for name in free_parameter_names:
                rvs.append(priors[name].rvs())
            return np.array(rvs)

        def logprior(parameters):
            _logprior = 0.0
            for par, name in zip(parameters, free_parameter_names):
                _logprior += priors[name].logpdf(par)
            return _logprior

        def logposterior(parameters):
            _logprior = logprior(parameters)
            if np.isneginf(_logprior):
                return -np.inf, -np.inf
            elbo = - self.nELBO(parameters, max_iter=100)
            return _logprior + elbo, elbo

        ndim = len(free_parameter_names)
        nwalkers = 2 * ndim

        print(f'Setting up sampler (parameters: {ndim}, walkers: {nwalkers})')

        if p0 is None:
            p0 = []
            for i in range(nwalkers - len(p0)):
                p0.append(prior_rvs())
            p0 = np.array(p0)
        else:
            sigma = []
            for name in free_parameter_names:
                try:
                    sigma.append(priors[name].std())
                except TypeError:
                    sigma.append(priors[name].std)

            p0 = sample_ellipsoid(p0, np.diag(sigma) / 100, size=nwalkers)
            for i, p in enumerate(p0):
                if np.isneginf(logprior(p)):
                    p0[i] = prior_rvs()

        print('initial values for parameters are set')
        _start = time_module.time()
        _ = [logposterior(p) for p in p0]
        _end = time_module.time()
        print()
        print(f'evaluation for initial values took {_end - _start:.0f} sec')
        print('- adjust your expectations accordingly')

        # Set up the backend
        filename = "gprn.h5"
        be = backends.HDFBackend(filename)
        be.reset(nwalkers, ndim)

        sampler = EnsembleSampler(nwalkers, ndim, logposterior, backend=be)

        # track the average autocorrelation time estimate
        index = 0
        autocorr = np.empty(niter)
        old_tau = np.inf

        for sample in sampler.sample(p0, iterations=niter, progress=True):
            if sampler.iteration % 10 == 0:
                print(sample.log_prob.max())
            # check convergence every 100 steps
            if sampler.iteration % 10:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print('MCMC converged!')
                break
            old_tau = tau

        return sampler


    def _Prediction(self, nodes=None, weights=None, means=None, jitters=None,
                   tstar=None, mu=None, var=None, separate=False):
        """
        Prediction for mean-field inference

        Args:
            nodes: array
                Node functions 
            weights: array
                Weight function
            means: array
                Mean functions
            jitters: array
                Jitter terms
            tstar: array
                Predictions time
            mu: array, optional
                Variational means, use self._mu if not provided
            var: array, optional
                Variational variances, use self._var if not provided
            separate: bool
                Whether to return predictive for nodes and weights separately

        Returns:
            predictive mean(s): array
                Predicted means
            predictive variance(s): array
                Predictive variances
        """
        if nodes is None:
            nodes = self.nodes
        if weights is None:
            weights = self.weights
        if means is None:
            means = self.means
        if jitters is None:
            jitters = self.jitters

        if tstar is None:
            tstar = self.time

        if mu is None and var is None:
            if self._mu is None and self._var is None:
                mu, var = self._initMuVar(nodes, weights, jitters)
            else:
                mu, var = self._mu, self._var

        muF, muW = self._u_to_fhatW(mu.flatten())
        varF, varW = self._u_to_fhatW(var.flatten())
        meanVal = self._mean(means, tstar)
        meanVal = np.array(np.array_split(meanVal, self.p))
        y = np.concatenate(self.y) - self._mean(means)
        y = np.array(np.array_split(y, self.p))
        weights = np.array(weights).reshape(self.q, self.p)
        jitt2 = np.array(jitters)**2
        nPred, nVar = [], []
        wPred, wVar = [], []
        for q in range(self.q):
            gpObj = _gp.GP(self.time, muF[:, q, :])
            n, nv = gpObj.prediction(nodes[q], tstar,
                                     muF[:, q, :].reshape(self.N),
                                     varF[:, q, :].reshape(self.N))
            nPred.append(n)
            nVar.append(nv)
            for p in range(self.p):
                gpObj = _gp.GP(self.time, muW[p, q, :])
                w, wv = gpObj.prediction(weights[q, p], tstar,
                                         muW[p, q, :].reshape(self.N),
                                         varW[p, q, :].reshape(self.N))
                wPred.append(w)
                wVar.append(wv)
        nPred, nVar = np.array(nPred), np.array(nVar)

        wPredd = np.array(wPred).reshape(self.q, self.p, tstar.size)
        wVarr = np.array(wVar).reshape(self.q, self.p, tstar.size)
        predictives = np.zeros((tstar.size, self.p))
        predictivesVar = np.zeros((tstar.size, self.p))
        for p in range(self.p):
            predictives[:, p] += meanVal[p]
            for q in range(self.q):
                predictives[:, p] += nPred[q] * wPredd[q, p]
                predictivesVar[:,p] += wPredd[q,p]*wPredd[q,p]*nVar[q]\
                                       +wVarr[q,p]*(nVar[q] +nPred[q]*nPred[q])\
                                       +jitt2[p]
        wPred, wVar = np.array(wPred), np.array(wVar)

        if separate:
            predictives = np.array(predictives)
            sepPredictives = np.array([nPred, wPred], dtype=object)
            return predictives, predictivesVar, sepPredictives
        return predictives, predictivesVar

    def predict(self, tstar=None, nn=1000):
        """
        Get the GPRN prediction

        Args:
            tstar: array, optional
                Times at which to get prediction. If not provided, uses a
                linspace around `self.time`
            nn: int, default 1000
                Number of points in prediction, if `tstar` is not provided
        """
        if tstar is None:
            mi, ma = self.time.min(), self.time.max()
            tptp = self.time.ptp()
            tstar = np.linspace(mi - 0.2 * tptp, ma + 0.2 * tptp, nn)

        # a, v = self._Prediction()
        aa, vv, bb = self._Prediction(tstar=tstar, separate=True)
        ss = np.sqrt(vv)
        return tstar, aa, ss, bb


    plot_prediction = plot_prediction

    def plot_structure(self):
        raise NotImplementedError
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg

        import daft
        pgm = daft.PGM()

        # observed datasets as "plates"
        pgm.add_plate([1.5, 0.2, 2, 3.2], label=r"exposure $i$", shift=-0.1)
        pgm.add_plate([2, 0.5, 1, 1], label=r"pixel $j$", shift=-0.1)
        pgm.render()
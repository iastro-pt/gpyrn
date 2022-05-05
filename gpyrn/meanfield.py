from itertools import chain
import time as time_module
from gpyrn.meanfunc import array_input
from gpyrn import _gp, covfunc

import numpy as np
from scipy.linalg import cholesky, LinAlgError
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from gpyrn import _gp


class inference(object):
    """
    Mean-field variational inference for GPRNs.
    See Nguyen & Bonilla (2013) for more information.

    Parameters
    ----------
    num_nodes: int
        Number of latent node functions f(x), called f hat in the article
    time: array
        Time coordinates
    *args: arrays
        The observed data in the following order:
            y1, y1error, y2, y2error, ...
    """

    def __init__(self, num_nodes, time, *args):
        # number of node functions; f(x) in Wilson et al. (2012)
        self.num_nodes = num_nodes
        self.q = num_nodes
        # array of the time
        self.time = time
        # number of observations, N in Wilson et al. (2012)
        self.N = self.time.size

        # check if the input was correct
        msg = 'Number of observed data arrays should be even: y1, y1error, ...'
        assert len(args) > 0 and len(args) % 2 == 0, msg

        # the data, it should be given as data1, data1error, data2, ...
        self.args = args
        # number of outputs y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args) / 2)
        # total number of weights, we will have q*p weights in total
        self.qp = self.q * self.p
        self.d = self.time.size * self.q * (self.p + 1)
        # to organize the data we now join everything
        self.tt = np.tile(time, self.p)  # "extended" time

        ys = []
        ystd = []
        yerrs = []
        for i, j in enumerate(args):
            if i % 2 == 0:
                ys.append(j)
                ystd.append(np.std(j))
            else:
                yerrs.append(j)
        self.ystd = np.array(ystd).reshape(self.p, 1)
        self.y = np.array(ys).reshape(self.p, self.N)  # matrix p*N of outputs
        self.yerr = np.array(yerrs).reshape(self.p, self.N)  # matrix p*N of errors
        self.yerr2 = self.yerr**2

        self._components_set = False
        self._frozen_mask = np.array([])
        self._mu, self._var = None, None
        self._mu_var_iters = 0
        self.update_muvar_after = 50
        self.elbo_max_iter = 5000

    def set_components(self, nodes, weights, means, jitters):
        if isinstance(nodes, covfunc.covFunction):
            nodes = [nodes]
        self.nodes = nodes

        if isinstance(weights, covfunc.covFunction):
            weights = [weights]
        self.weights = weights

        if isinstance(means, (int, float)):
            means = [means]
        self.means = means
        
        if isinstance(jitters, (int, float)):
            jitters = [jitters]
        self.jitters = np.array(jitters, dtype=np.float)

        self._components_set = True

    def get_parameters(self, nodes=None, weights=None, means=None,
                       jitters=None, include_frozen=False):
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

    @array_input
    def set_parameters(self, parameters):
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
        self._frozen_mask = np.ones(self._frozen_mask.size, dtype=bool)

    fix_parameter = freeze_parameter
    fix_all_parameters = freeze_all_parameters

    def thaw_parameter(self, index=None, name=None):
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
        self._frozen_mask = np.zeros(self._frozen_mask.size, dtype=bool)

    free_parameter = thaw_parameter
    free_all_parameters = thaw_all_parameters

    @property
    def frozen_mask(self):
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg
        if self._frozen_mask.size == 0:
            self._frozen_mask = np.full(self.n_parameters, False, dtype=bool)
        return self._frozen_mask

    @frozen_mask.setter
    def frozen_mask(self, mask):
        msg = 'Do not set frozen_mask, use thaw_parameter/freeze_parameter'
        raise NotImplementedError(msg)


##### mean functions definition ################################################


    def _mean(self, means, time=None):
        """
        Returns the values of the mean functions

        Parameters
        ----------
        means : list of instances of meanfunc
        time : array, optional

        Returns
        -------
        m: array
            Value of the mean evaluated at `time` or `self.time`
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


##### To create matrices and samples ###########################################


    def _KMatrix(self, kernel, time=None):
        """
        Returns the covariance matrix created by evaluating a given kernel at
        inputs time. For stability issues with the GPRN a 1e-6 nugget is needed.

        Parameters
        ----------
        kernel : instance of covfunc
        time : array, optional

        Returns
        -------
        K: array
            Matrix of a covariance function
        """
        r = time[:, None] - time[None, :]
        K = kernel(r) + 1e-6 * np.diag(np.diag(np.ones_like(r)))
        # K[np.abs(K) < 1e-12] = 0.
        return K


    def _tinyNuggetKMatrix(self, kernel, time=None):
        """
        To be used in Prediction(). Returns the covariance matrix created by
        evaluating a given kernel at inputs time with the tiniest stability
        nugget possible.

        Returns
        -------
        K: array Matrix of a covariance function
        """
        r = time[:, None] - time[None, :]
        K = kernel(r) + 1.25e-12 * np.diag(np.diag(np.ones_like(r)))
        return K


    def _predictKMatrix(self, kernel, time):
        """
        To be used in Prediction()

        Parameters
        ----------
        kernel : instance of covfunc
        time : array, optional

        Returns
        -------
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
        Given an array of values, divides it in the corresponding nodes (f hat)
        and weights (w) parts
        
        Parameters
        ----------
        u: array
        
        Returns
        -------
        f: array
            Samples of the nodes
        w: array
            Samples of the weights
        """
        f = u[:self.q * self.N].reshape((1, self.q, self.N))
        w = u[self.q * self.N:].reshape((self.p, self.q, self.N))
        return f, w


    def _cholNugget(self, matrix, maximum=1000):
        """
        Returns the cholesky decomposition to a given matrix, if it is not
        positive definite, a nugget is added to its diagonal.
        
        Parameters
        ----------
        matrix: array
            Matrix to decompose
        maximum: int
            Number of times a nugget is added.
        
        Returns
        -------
        L: array
            Matrix containing the Cholesky factor
        nugget: float
            Nugget added to the diagonal
        """
        nugget = 0 #our nugget starts as zero
        try:
            nugget += 1e-15
            L = cholesky(matrix, lower=True, overwrite_a=True)
            return L, nugget
        except LinAlgError:
            n = 0 #number of tries
            while n < maximum:
                try:
                    L = cholesky(matrix + nugget*np.identity(matrix.shape[0]),
                                 lower=True, overwrite_a=True)
                    return L, nugget
                except LinAlgError:
                    nugget *= 10.0
                finally:
                    n += 1
            raise LinAlgError("Not positive definite, even with nugget.")


    def _initMuVar(self, nodes, weights, jitter):
        a1 = [n.pars[0] for n in nodes]
        a2 = [w.pars[0] for w in weights]
        mean1, mean2 = [], []
        var1, var2 = [], []
        for _, n in enumerate(a1):
            m = [np.sqrt(np.abs(j)*n/i)*np.sign(j) for i,j in zip(a2,self.y)]
            mean1.append(np.mean(m, axis=0))
            mean2.append([np.sqrt(np.abs(j)*i/n) for i,j in zip(a2,self.y)])
            var1.append([np.mean(jitter)*np.ones_like(self.time)])
            var2.append([jitt*np.ones_like(self.time) for jitt in jitter])
        mu = np.concatenate((mean1, mean2), axis=None)
        var = np.concatenate((var1, var2), axis=None)
        return mu, var


    def _randomMuVar(self):
        mu = np.random.randn(self.d, 1)
        var = np.random.rand(self.d, 1)
        return mu, var


    def sampleIt(self, latentFunc, time=None):
        """
        Returns samples from the kernel
        
        Parameters
        ----------
        latentFunc: func
            Covariance function
        time: array
            Time array
        
        Returns
        -------
        norm: array
            Sample of K 
        """
        if time is None:
            time = self.time
        mean = np.zeros_like(time)
        K = self._tinyNuggetKMatrix(latentFunc, time)
        normal = multivariate_normal(mean, K, allow_singular=True).rvs()
        return normal

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


##### Mean-Field Inference functions ##########################################

    def ELBOcalc(self, nodes=None, weights=None, means=None, jitters=None,
                 max_iter=None, mu=None, var=None):
        """
        Function to use to calculate the evidence lower bound

        Parameters
        ----------
        nodes: list of `covFunction` instances
            Kernel(s) for the node(s)
        weights: list of `covFunction` instances
            Kernel(s) for the weight(s)
        means: list of `meanFunction` instances
            Mean functions
        jitters: list of floats
            Jitter terms
        max_iter: int, default self.elbo_max_iter
            Maximum number of iterations allowed in ELBO calculation
        mu: array or str, optional
            Variational means or 'init', 'random', or 'previous'
        var: array or str, optional
            Variational variances or 'init', 'random', or 'previous'

        Returns
        -------
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
        if mu is None and var is None:
            mu = var = 'init'

        if mu == 'previous' and var == 'previous':
            if self._mu is not None:
                mu, var = self._mu, self._var
            else:
                mu, var = self._initMuVar(nodes, weights, jitters)
        elif mu == 'random' and var == 'random':
            mu, var = self._randomMuVar()
        elif mu == 'init' and var == 'init':
            mu, var = self._initMuVar(nodes, weights, jitters)

        jitt2 = np.array(jitters)**2
        Kf = np.array([self._KMatrix(i, self.time) for i in nodes])
        Kw = np.array([self._KMatrix(j, self.time) for j in weights])
        Lf = np.array([self._cholNugget(j)[0] for j in Kf])
        Lw = np.array([self._cholNugget(j)[0] for j in Kw])
        y = np.concatenate(self.y) - self._mean(means)
        y = np.array(np.array_split(y, self.p))

        # To add new elbo values inside
        ELBO, *_ = self.ELBOaux(Kf, Kw, Lf, Lw, y, jitt2, mu, var)
        elboArray = np.array([ELBO])
        iterNumber = 0

        if max_iter is None:
            max_iter = self.elbo_max_iter

        while iterNumber < max_iter:
            # Optimize mu and var analytically
            ELBO, mu, var, _, _ = self.ELBOaux(Kf, Kw, Lf, Lw, y, jitt2, mu, var)
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            # Stoping criteria:
            if iterNumber > 5:
                means = np.mean(elboArray[-5:])
                criteria = np.abs(np.std(elboArray[-5:]) / means)
                if criteria < 1e-3 and criteria != 0:
                    self._mu = mu
                    self._var = var
                    return ELBO, mu, var, iterNumber

        print('\nMax iterations reached')
        return ELBO, mu, var, iterNumber

    def ELBOaux(self, Kf, Kw, Lf, Lw, y, jitt2, mu, var):
        """
        Evidence Lower bound to use in ELBOcalc()

        Parameters
        ----------
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

        Returns
        -------
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
        sigmaF, muF, sigmaW, muW = self._updateSigMu(Kf, Kw, y, jitt2,
                                                     muF, varF, muW, varW)
        #new mean and var for the nodes
        muF = muF.reshape(1, self.q, self.N)
        varF =  np.zeros_like(varF)
        for i in range(self.q):
            varF[:,i,:] = np.diag(sigmaF[i,:,:])
        #new mean and varfor the weights
        varW =  np.zeros_like(varW)
        for j in range(self.q):
            for i in range(self.p):
                varW[i,j,:] = np.diag(sigmaW[j, i,:, :])
        new_mu = np.concatenate((muF, muW))
        new_var = np.concatenate((varF, varW))
        #Entropy
        Ent = self._entropy(sigmaF, sigmaW)
        #Expected log prior
        LogP = self._expectedLogPrior(Kf, Kw, Lf, Lw, sigmaF, muF, sigmaW, muW)
        #Expected log-likelihood
        LogL = self._expectedLogLike(y, jitt2, sigmaF, muF, sigmaW, muW)
        ELBO = (LogL + LogP + Ent) /self.q  #Evidence Lower Bound
        return ELBO, new_mu, new_var, sigmaF, sigmaW


    def _updateSigMu(self, Kf, Kw, y, jitt2, muF, varF, muW, varW):
        """
        Efficient closed-form updates fot variational parameters. This
        corresponds to eqs. 16, 17, 18, and 19 of Nguyen & Bonilla (2013)

        Parameters
        ----------
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
            
        Returns
        -------
        sigma_f: array
            Updated variational covariance of each node
        mu_f: array
            Updated variational mean of each node
        sigma_w: array
            Updated variational covariance of each weight
        mu_w: array
            Updated variational mean of each weight
        """

        Kw = Kw.reshape(self.q, self.p, self.N, self.N)
        muF = np.squeeze(muF)
        #creation of Sigma_fj and mu_fj
        sigma_f, mu_f = [], []
        for j in range(self.q):
            diagFj, auxCalc = 0, 0
            for i in range(self.p):
                diagFj = diagFj + (muW[i,j,:]*muW[i,j,:]+varW[i,j,:]) \
                                                /(jitt2[i] + self.yerr2[i,:])
                sumNj = np.zeros(self.N)
                for k in range(self.q):
                    if k != j:
                        sumNj += muW[i,k,:]*muF[k,:].reshape(self.N)
                auxCalc = auxCalc + ((y[i,:]-sumNj)*muW[i,j,:])/(jitt2[i]+self.yerr2[i,:])
            CovF = np.diag(1 / diagFj) + Kf[j]
            sigF = Kf[j] - Kf[j] @ np.linalg.solve(CovF, Kf[j])
            sigma_f.append(sigF)
            mu_f.append(sigF @ auxCalc)
            muF = np.array(mu_f)
        sigma_f = np.array(sigma_f)
        mu_f = np.array(mu_f)
        #creation of Sigma_wij and mu_wij
        sigma_w, mu_w = [], np.zeros_like(muW)
        for j in range(self.q):
            for i in range(self.p):
                mu_fj = mu_f[j]
                var_fj = np.diag(sigma_f[j])
                Diag_ij = (mu_fj*mu_fj+var_fj)/(jitt2[i]+self.yerr2[i,:])
                Kw_ij = Kw[j,i,:,:]
                CovWij = np.diag(1 / Diag_ij) + Kw_ij
                sigWij = Kw_ij - Kw_ij @ np.linalg.solve(CovWij, Kw_ij)
                sigma_w.append(sigWij)
                sumNj = np.zeros(self.N)
                for k in range(self.q):
                    if k != j:
                        sumNj += mu_f[k].reshape(self.N)*np.array(muW[i,k,:])
                auxCalc = ((y[i,:]-sumNj)*mu_f[j,:])/(jitt2[i]+self.yerr2[i,:])
                muW[i,j,:] = sigWij @ auxCalc
        sigma_w = np.array(sigma_w).reshape(self.q, self.p, self.N, self.N)
        mu_w = np.array(muW)
        return sigma_f, mu_f, sigma_w, mu_w


    def _expectedLogLike(self, y, jitt2, sigma_f, mu_f, sigma_w, mu_w):
        """
        Calculates the expected log-likelihood in mean-field inference,
        corresponds to eq.14 in Nguyen & Bonilla (2013)

        Parameters
        ----------
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
            
        Returns
        -------
        logl: float
            Expected log-likelihood value
        """
        logl = 0
        for p in range(self.p):
            for n in range(self.N):
                logl += np.log(2*np.pi*(jitt2[p] + self.yerr2[p,n]))
        logl *= -0.5
        sumN = []
        for n in range(self.N):
            for p in range(self.p):
                Ydiff = y[p,n] - mu_f[0,:,n] @mu_w[p,:,n].T
                bottom = jitt2[p]+self.yerr2[p,n]
                sumN.append((Ydiff.T * Ydiff)/bottom)
        logl += -0.5 * np.sum(sumN)
        value = 0
        for p in range(self.p):
            for q in range(self.q):
                value += np.sum((np.diag(sigma_f[q,:,:])*mu_w[p,q,:]*mu_w[p,q,:] +\
                                np.diag(sigma_w[q,p,:,:])*mu_f[:,q,:]*mu_f[:,q,:] +\
                                np.diag(sigma_f[q,:,:])*np.diag(sigma_w[q,p,:,:]))\
                                /(jitt2[p]+self.yerr2[p,:]))
        logl += -0.5* value
        return logl


    def _expectedLogPrior(self, Kf, Kw, Lf, Lw, sigma_f, mu_f, sigma_w, mu_w):
        """
        Calculates the expection of the log prior wrt q(f,w) in mean-field
        inference, corresponds to eq.15 in Nguyen & Bonilla (2013)
        
        Parameters
        ----------
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
        
        Returns
        -------
        logp: float
            Expected log prior value
        """
        #we have Q nodes -> j in the paper; we have P y(x)s -> i in the paper
        first_term = 0 #calculation of the first term of eq.15 of Nguyen & Bonilla (2013)
        second_term = 0 #calculation of the second term of eq.15 of Nguyen & Bonilla (2013)
        Kw = Kw.reshape(self.q, self.p, self.N, self.N)
        Lw = Lw.reshape(self.q, self.p, self.N, self.N)
        muW = mu_w.reshape(self.q, self.p, self.N)
        sumSigmaF = np.zeros_like(sigma_f[0])
        for j in range(self.q):
            Lfj = Lf[j]
            logKf = np.float(np.sum(np.log(np.diag(Lfj))))
            muK =  np.linalg.solve(Lfj, mu_f[:,j, :].reshape(self.N))
            muKmu = muK @ muK
            sumSigmaF = sumSigmaF + sigma_f[j]
            trace = np.trace(np.linalg.solve(Kf[j], sumSigmaF))
            first_term += -logKf - 0.5*(muKmu + trace)
            for i in range(self.p):
                muK = np.linalg.solve(Lw[j,i,:,:], muW[j,i])
                muKmu = muK @ muK
                trace = np.trace(np.linalg.solve(Kw[j,i,:,:], sigma_w[j,i,:,:]))
                second_term += -np.float(np.sum(np.log(np.diag(Lw[j,i,:,:]))))\
                                - 0.5*(muKmu + trace)
        const = -0.5*self.N*self.q*(self.p+1)*np.log(2*np.pi)
        logp = first_term + second_term + const
        return logp


    def _entropy(self, sigma_f, sigma_w):
        """
        Calculates the entropy in mean-field inference, corresponds to eq.14 
        in Nguyen & Bonilla (2013)
        
        Parameters
        ----------
            sigma_f: array
                Variational covariance for each node
            sigma_w: array
                Variational covariance for each weight
        
        Returns
        -------
        entropy: float
            Final entropy value
        """
        entropy = 0 #starts at zero then we sum everything
        for j in range(self.q):
            L1 = self._cholNugget(sigma_f[j])
            entropy += np.sum(np.log(np.diag(L1[0])))
            for i in range(self.p):
                L2 = self._cholNugget(sigma_w[j, i, :, :])
                entropy += np.sum(np.log(np.diag(L2[0])))
        const = 0.5*self.q*(self.p+1)*self.N*(1+np.log(2*np.pi))
        return entropy + const


    def nELBO(self, parameters):
        msg = 'GPRN components not set, use set_components'
        assert self._components_set, msg
        self.set_parameters(parameters)

        start = time_module.time()
        elbo, _, _ = self.ELBOcalc(self.nodes, self.weights, self.means,
                                   self.jitters, mu='previous', var='previous')
        end = time_module.time()

        spaces = 20*' '
        print(f'ELBO={elbo:7.2f} (took {1e3*(end-start):5.2f} ms){spaces}',
              end='\r', flush=True)
        return -elbo


    def optimize(self, vars=None, **kwargs):
        """
        Optimize (maximize) the ELBO. If provided, `vars` controls the
        parameters which are free during the optimization.

        Arguments
        ---------
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
                    self.thaw_parameter(name='*')    # thaw all
                    self.freeze_parameter(name=vars) # freeze vars
                else:
                    self.freeze_parameter(name='*') # freeze all
                    self.thaw_parameter(name=vars)  # thaw vars
            elif isinstance(vars, list):
                self.freeze_parameter(name='*') # freeze all
                for var in vars:
                    self.thaw_parameter(name=var) # except all vars
            else:
                msg = f'`vars` should be str or list, got {type(vars)}'
                raise ValueError(msg)

        res = minimize(self.nELBO, self.get_parameters(), **kwargs)
        return res


    def Prediction(self, nodes=None, weights=None, means=None, jitters=None,
                   tstar=None, mu=None, var=None, separate=False):
        """
        Prediction for mean-field inference
        
        Parameters
        ----------
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

        Returns
        -------
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
            gpObj = _gp.GP(self.time, muF[:,q,:])
            n,nv = gpObj.prediction(nodes[q], tstar, muF[:,q,:].reshape(self.N),
                                    varF[:,q,:].reshape(self.N))
            nPred.append(n)
            nVar.append(nv)
            for p in range(self.p):
                gpObj = _gp.GP(self.time, muW[p,q,:])
                w,wv = gpObj.prediction(weights[q,p], tstar,
                                        muW[p,q,:].reshape(self.N),
                                        varW[p,q,:].reshape(self.N))
                wPred.append(w)
                wVar.append(wv)
        nPred, nVar = np.array(nPred), np.array(nVar)
        wPredd = np.array(wPred).reshape(self.q, self.p, tstar.size)
        wVarr = np.array(wVar).reshape(self.q, self.p, tstar.size)
        predictives = np.zeros((tstar.size, self.p))
        predictivesVar = np.zeros((tstar.size, self.p))
        for p in range(self.p):
            predictives[:,p] += meanVal[p]
            for q in range(self.q):
                predictives[:,p] += nPred[q]*wPredd[q,p]
                predictivesVar[:,p] += wPredd[q,p]*wPredd[q,p]*nVar[q]\
                                       +wVarr[q,p]*(nVar[q] +nPred[q]*nPred[q])\
                                       +jitt2[p]
        if separate:
            predictives = np.array(predictives)
            sepPredictives = np.array([nPred,wPred], dtype=object)
            return predictives, predictivesVar, sepPredictives
        return predictives, predictivesVar


### END

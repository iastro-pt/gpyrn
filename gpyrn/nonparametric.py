import numpy as np

from scipy.linalg import cholesky, LinAlgError, inv
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

class inference(object):
    """ 
    Class to perform mean field variational inference for GPRNs. 
    See Nguyen & Bonilla (2013) for more information.
    
    Parameters
    ----------
    num_nodes: int
        Number of latent node functions f(x), called f hat in the article
    time: array
        Time coordinates
    k: int 
        Mixture of k isotropic gaussian distributions
    *args: arrays
        The actual data (or components), it needs be given in order of data1, 
        data1error, data2, data2error, etc...
    """ 
    def  __init__(self, num_nodes, time, k, *args):
        #number of node functions; f(x) in Wilson et al. (2012)
        self.num_nodes = num_nodes
        self.q = num_nodes
        #array of the time
        self.time = time 
        #number of observations, N in Wilson et al. (2012)
        self.N = self.time.size
        #mixture of k isotropic gaussian distributions
        self.k  = k
        #the data, it should be given as data1, data1error, data2, ...
        self.args = args 
        #number of outputs y(x); p in Wilson et al. (2012)
        self.p = int(len(self.args)/2)
        #total number of weights, we will have q*p weights in total
        self.qp =  self.q * self.p
        self.d = self.time.size * self.q *(self.p+1)
        #to organize the data we now join everything
        self.tt = np.tile(time, self.p) #"extended" time
        ys = []
        ystd = []
        yerrs = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                ys.append(j)
                ystd.append(np.std(j))
            else:
                yerrs.append(j)
        self.ystd = np.array(ystd).reshape(self.p, 1)
        self.y = np.array(ys).reshape(self.p, self.N) #matrix p*N of outputs
        self.yerr = np.array(yerrs).reshape(self.p, self.N) #matrix p*N of errors
        self.yerr2 = self.yerr**2
        #check if the input was correct
        assert int((i+1)/2) == self.p, \
        'Given data and number of components dont match'
        
        
##### mean functions definition ###############################################
    def _mean(self, means, time=None):
        """
        Returns the values of the mean functions
        
        Parameters
        ----------
        
        Returns
        -------
        m: float
            Value of the mean
        """
        if time is None:
            N = self.time.size
            m = np.zeros_like(self.tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i*N : (i+1)*N] = meanfun(self.time)
        else:
            N = time.size
            tt = np.tile(time, self.p)
            m = np.zeros_like(tt)
            for i, meanfun in enumerate(means):
                if meanfun is None:
                    continue
                else:
                    m[i*N : (i+1)*N] = meanfun(time)
        return m
    
    
##### To create matrices and samples ###########################################
    def _kernelMatrix(self, kernel, time = None):
        """
        Returns the covariance matrix created by evaluating a given kernel 
        at inputs time
        
        Parameters
        ----------
        
        Returns
        -------
        K: array
            Matrix of a covariance function
        """
        r = time[:, None] - time[None, :]
        #K = kernel(r)
        K = kernel(r) + 1e-6*np.diag(np.diag(np.ones_like(r)))
        K[np.abs(K)<1e-12] = 0.
        return K
    
    
    def _predictKMatrix(self, kernel, time):
        """
        To be used in predict_gp()
        
        Parameters
        ----------
        
        Returns
        -------
        K: array
            Matrix of a covariance function
        """
        if time.size == 1:
            r = time - self.time[None, :]
        else:
            r = time[:,None] - self.time[None,:]
        K = kernel(r) 
        return K
    
    
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
    
    
    def _cholNugget(self, matrix, maximum=10):
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
        #print(latentFunc)
        if time is None:
            time = self.time
        mean = np.zeros_like(time)
        K = self._kernelMatrix(latentFunc, time)
        normal = multivariate_normal(mean, K, allow_singular=True).rvs()
        return normal
    
    
##### Non-Parametric Variational Inference functions ###########################
    def ELBOcalc(self, nodes, weights, meanf, jitters, iterations = 10000):
        """
        Function to use to calculate the evidence lower bound
        
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
        iterations: int
            Number of iterations 

            
        Returns
        -------
        ELBO: array
            Value of the ELBO per iteration

        """
        #initial variational parameters
        mu = np.random.rand(self.d, self.k).T
        var = np.ones_like(np.random.rand(1, self.k).T) # why?
        muF, muW = [], []
        for k in range(self.k):
            m1, m2 = self._u_to_fhatW(mu[k, :])
            muF.append(m1)
            muW.append(m2)
        muF = np.array(muF)
        muW = np.array(muW)
        
        ELBO = self.ELBOaux(nodes, weights, meanf, jitters, mu, var)
        print(ELBO)
        elboArray = np.array([ELBO]) #To add new elbo values inside
        iterNumber = 1
        while iterNumber < iterations:
            #ELBO = self.ELBOaux(nodes, weights, meanf, jitters, mu, var)
            ELBO, mu, var = self.updateMUandVAR(nodes, weights, meanf, jitters, 
                                                mu, var)
            elboArray = np.append(elboArray, ELBO)
            iterNumber += 1
            #Stoping criteria:
            if iterNumber > 5:
                means = np.mean(elboArray[-5:])
                criteria = np.abs(np.std(elboArray[-5:]) / means)
                if criteria < 1e-3 and criteria !=0:
                    return ELBO, mu, var
        print('Max iterations reached')
        return ELBO, mu, var
    
    
    def ELBOaux(self, nodes, weights, meanf, jitters, mu, var):
        """
        Evidence Lower bound to use in ELBOcalc()
        
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
        muF, muW = [], []
        for k in range(self.k):
            m1, m2 = self._u_to_fhatW(mu[k, :])
            muF.append(m1)
            muW.append(m2)
        muF = np.array(muF)
        muW = np.array(muW)
        
        #nodes and means
        Kf = np.array([self._kernelMatrix(i, self.time) for i in nodes])
        invKf = np.array([inv(i) for i in Kf])
        Lf = np.array([self._cholNugget(i)[0] for i in Kf])
        Kw = np.array([self._kernelMatrix(j, self.time) for j in weights])
        invKw = np.array([inv(j) for j in Kw])
        Lw = np.array([self._cholNugget(j)[0] for j in Kw])
        
        #Entropy
        Entropy = self._entropy(mu, var)
        #print('entropy:', Entropy)
        #Expected log-likelihood
        ExpLoglike = self._expectedLogLike(nodes, weights, meanf, jitters, 
                                           muF, muW, var)
        #print('expLL:', ExpLoglike)
        #Expected log prior
        ExpLogprior = self._expectedLogPrior(Kf, invKf, Lf, Kw, invKw, Lw, 
                                             muF, muW, var, jitters)
        #print('expLP:', ExpLogprior)
        # print('expLL+expLP:', np.sum(ExpLoglike + ExpLogprior)/self.k)
        ELBO = np.sum(ExpLoglike + ExpLogprior)/self.k -Entropy
        return ELBO
    
    
    def _entropy(self, mu, var):
        varmin = 1e-7
        beta = np.ones((self.k,1)) / self.k
        S0 = np.array(mu - np.mean(mu, axis=0)).T
        S = np.sum(S0*S0, axis=0) - 2*(S0.T @ S0)
        S = np.sum(S0*S0, axis=0) + S.T
        S[S<0] = 0 #numerical noise can cause it to negative
        var = var**2 + varmin
        s = (var[:,None] + var[None,:])
        logP = -0.5*S/s - 0.5*self.d*np.log(s)
        logP[logP<0] = 0 #numerical noise can cause it to negative 
        a = np.zeros((self.k,1))
        for i in range(self.k):
            a[i] = -np.log(self.k) + np.log(np.sum(np.exp(logP[0,i])))
        entropy = np.float(a.T @ beta)
        return entropy
    
    
    def _expectedLogLike(self, nodes, weights, means, jitters, muF, muW, var):
        new_y = np.concatenate(self.y) - self._mean(means, self.time)
        new_y = np.array(np.array_split(new_y, self.p)).T 
        jitt2 = np.array(jitters)**2 
        errs = 0
        for i in range(self.p):
            errs += jitt2[i] + self.yerr2[i]
        ### first term of equation 3.22
        Wblk = []
        for k in range(self.k):
            Wblk.append(np.squeeze(muW[k]))
        Wblk = np.array(Wblk)
        Fblk = []
        for k in range(self.k):
            Fblk.append(np.squeeze(muF[k]))
        Fblk = np.array(Fblk)
        Ymean = Wblk * Fblk
        Ydiff = (new_y.T - Ymean)**2/errs
        logl = -0.5 * np.sum(Ydiff, axis=1)
        ### second term of equation 3.22
        kvals = []
        for k in range(self.k):
            value= 0
            for q in range(self.q):
                for p in range(self.p):
                    value = (muF[k,:,q,:] * muF[k,:,q,:])/(jitt2[p]+self.yerr2[p,:])
                    value += (muW[k,p,:,:] * muW[k,p,:,:])/(jitt2[p]+self.yerr2[p,:])
                    value += var[k]**4 * self.q /(jitt2[p] + np.sum(self.yerr2[p,:]))
                    kvals.append(self.p*var[k]**2*np.sum(value))
        kvals = np.array(np.squeeze(kvals))
        logl += -0.5*kvals
        ### third term of equation 3.22
        value = 0
        for p in range(self.p):
            for n in range(self.N):
                value += np.log(2*np.pi*(jitt2[p] + self.yerr2[p,n]))
        logl += -0.5*value
        return logl
        
        
    def _expectedLogPrior(self, Kf, invKf, Lf, Kw, invKw, Lw, 
                          muF, muW, var, jitters):
        ### first term
        logKf = [2*np.sum(np.log(np.diag(i))) for i in Lf]
        logKw = [2*np.sum(np.log(np.diag(i))) for i in Lw]
        logprior = -0.5*np.sum(logKf) -0.5*np.sum(logKw)
        ### second term 
        sum_kj, sum_kw = [], []
        for k in range(self.k):
            for q in range(self.q):
                mKfm = muF[k,:,q,:] @invKf[q] @muF[k,:,q,:].T
                vartracef = var[k]**2 *np.trace(invKf[q])
                for p in range(self.p):
                    #almost certain this will fail for more than 1 node
                    mKwm = muW[k,p,q,:] @invKw[p] @muW[k,p,q,:].T
                    vartracew = var[k]**2 *np.trace(invKw[p])
            sum_kj.append(np.float(mKfm+vartracef))
            sum_kw.append(np.float(mKwm+vartracew))
        logprior += -0.5*np.array(sum_kj) -0.5*np.array(sum_kw)
        return logprior
    
    
    def updateMUandVAR(self, nodes, weights, meanf, jitters, mu, var):
        res1 = minimize(self._updateMU, x0 = mu, 
                       args = (nodes, weights, meanf, jitters, var), 
                       method='Nelder-Mead', 
                       options={'disp': False, 
                                'maxiter': 200})
        mu  = res1.x
        res2 = minimize(self._updateVAR, x0 = var, 
                        args = (nodes, weights, meanf, jitters, mu), 
                        method='Nelder-Mead', 
                        options={'disp': False, 
                                'maxiter': 200})
        var  = res2.x
        mu = mu.reshape(self.k, self.d)
        ELBO = self.ELBOaux(nodes, weights, meanf, jitters, mu, var)
        return ELBO, mu, var

    def _updateMU(self, mu, nodes, weights, meanf, jitters, var):
        mu = mu.reshape(self.k, self.d)
        e = -self.ELBOaux(nodes, weights, meanf, jitters, mu, var)
        return e

    def _updateVAR(self, var, nodes, weights, meanf, jitters, mu):
        mu = mu.reshape(self.k, self.d)
        e = -self.ELBOaux(nodes, weights, meanf, jitters, mu, var)
        return e
    
    def _squaredDistance(self, X):
        m, n = X.shape
        D = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                D[i,j] = np.linalg.norm(X[:,i] - X[:,j])**2
                D[j,i] = D[i,j]
        return D


### END

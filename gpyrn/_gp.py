"""
Auxiliary Gaussian processes functions
"""
import numpy as np
from gprn import covFunction as kernels
from scipy.linalg import cho_factor, cho_solve

##### Gaussian processes #######################################################
class GP(object):
    """ 
    Class to create our Gaussian process.
    
    Parameters
    ----------
    kernel: func
        Covariance funtion
    means: func
        Mean function 
    time: array
        Time array
    y: array
        Measurements array
    yerr: array
        Measurements errors array
    """
    def __init__(self, time, y, yerr=None):
        self.time = time            #time
        self.y = y                  #measurements
        if yerr is None:
            self.yerr = 1e-12 * np.identity(self.time.size)
        else:
            self.yerr = np.array(yerr)        #measurements errors
        self.yerr2 = self.yerr**2
        
    def _kernel_pars(self, kernel):
        """ Returns a kernel parameters """
        return kernel.pars

    def _kernel_matrix(self, kernel, time):
        """ Returns a cov matrix when evaluating a given kernel at inputs time """
        r = time[:, None] - time[None, :]
        K = kernel(r) + 1.25e-12*np.diag(np.diag(np.ones_like(r)))
        return K

    def _predict_kernel_matrix(self, kernel, time):
        """ To be used in prediction() """
        r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def new_kernel(self, kernel, new_pars):
        """
        Updates the parameters of a kernel
        
        Parameters
        ----------
        kernel: func
            Original kernel
        new_pars: list
            New hyperparameters
        
        Returns
        -------
        new_k: func
            Updated kernel
        """
        #if we are working with the sum of kernels
        if isinstance(kernel, kernels.Sum):
            k1_params = []
            for i, j in enumerate(kernel.k1.pars):
                k1_params.append(new_pars[i])
            k2_params = []
            for i, j in enumerate(kernel.k2.pars):
                k2_params.append(new_pars[len(kernel.k1.pars)+i])
            new_k1 = type(kernel.k1)(*k1_params)
            new_k2 = type(kernel.k2)(*k2_params)
            return new_k1+new_k2
        #if we are working with the product of kernels
        elif isinstance(kernel, kernels.Multiplication):
            k1_params = []
            for i, _ in enumerate(kernel.k1.pars):
                k1_params.append(new_pars[i])
            k2_params = []
            for j, _ in enumerate(kernel.k2.pars):
                k2_params.append(new_pars[len(kernel.k1.pars)+j])
            new_k = type(kernel.k1)(*k1_params) * type(kernel.k1)(*k2_params)
            return new_k
        #if we are working with a "single" kernel
        else:
            new_k = type(kernel)(*new_pars)
            return new_k


    def prediction(self, kernel, time, m, v):
        """ 
        Conditional predictive distribution of the Gaussian process
        
        Parameters
        ----------
        kernel: func
            Covariance function
        time: array
            Time array

        Returns
        -------
        y_mean: array
            Mean vector
        y_var: array
            Variance vector
        """
        cov = self._kernel_matrix(kernel, self.time) + np.diag(v) #K
        L1 = cho_factor(cov)
        sol = cho_solve(L1, m)
        #Kstar
        Kstar = self._predict_kernel_matrix(kernel, time)
        #Kstarstar
        Kstarstar =  self._kernel_matrix(kernel, time)
        y_mean = np.dot(Kstar, sol) #mean
        kstarT_k_kstar = []
        for i, _ in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        return y_mean, y_var

### END

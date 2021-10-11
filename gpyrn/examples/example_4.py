import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
plt.rcParams['figure.figsize'] = [8, 4]
from scipy import stats
from multiprocessing import Pool

import emcee
import corner

from gpyrn import meanfield
from gpyrn import covfunc
from gpyrn import meanfunc

time = np.linspace(0, 100, 25)
y1 = 20*np.sin(2*np.pi*time / 31)
y1err = np.random.rand(25)
GPRN = meanfield.inference(1, time, y1,y1err)

#Priors
n_eta1 = stats.loguniform(0.1, 50)
n_eta3 = stats.uniform(20, 40-20)
n_eta4 = stats.loguniform(0.1, 5)
w_eta1 = stats.loguniform(0.1, 50)
w_eta2 = stats.uniform(0, 100)
jitt = stats.uniform(0, 1)

def priors():
    return np.array([n_eta1.rvs(),n_eta3.rvs(), n_eta4.rvs(),
                     w_eta1.rvs(), w_eta2.rvs(), jitt.rvs()])

def logPosterior(thetas):
    n1,n3,n4, w1,w2, j = thetas
    logprior = n_eta1.logpdf(n1)
    logprior += n_eta3.logpdf(n3)
    logprior += n_eta4.logpdf(n4)
    logprior += w_eta1.logpdf(w1)
    logprior += w_eta2.logpdf(w2)
    logprior += jitt.logpdf(j)
    if np.isinf(logprior):
        return -np.inf

    nodes = [covfunc.Periodic(n1, n3, n4)]
    weight = [covfunc.SquaredExponential(w1, w2)]
    means = [meanfunc.Constant(0)]
    jitter = [j]
    elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                               iterations=5000, mu='init', var='init')
    logposterior = logprior + elbo
    return logposterior

ndim = priors().size 
nwalkers = 2*ndim

pool = Pool(8)
sampler = emcee.EnsembleSampler(nwalkers, ndim, logPosterior, pool=pool)
                                
p0=[priors() for i in range(nwalkers)]
sampler.run_mcmc(p0, 5000, progress=True)

#chains plot
fig, axes = plt.subplots(6, figsize=(7, 12), sharex=True)
samples = sampler.get_chain()
labels = ["$\eta_1$", "$\eta_3$", "$\eta_4$",
          "$\\theta_1$", "$\\theta_2$", "jitter"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");
plt.savefig('chains.png', bbox_inches='tight')
plt.close('all')

#corner plot
flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
fig = corner.corner(flat_samples,labels=labels, color="k", bins = 50,
                    quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
                    show_titles=True, plot_density=True, plot_contours=True,
                    fill_contours=True, plot_datapoints=False)
plt.savefig('corner.png', bbox_inches='tight')
plt.close('all')
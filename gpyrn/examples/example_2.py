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
from matplotlib.ticker import AutoMinorLocator

from gpyrn import meanfield
from gpyrn import covfunc
from gpyrn import meanfunc

time = np.linspace(0, 100, 25)
y1 = 20*np.sin(2*np.pi*time / 31)
y1err = np.random.rand(25)

y2 = 25*np.sin(2*np.pi*time / 31 + 0.5*np.pi)
y2err = np.random.rand(25)

plt.figure()
plt.errorbar(time, y1, y1err, fmt='ob', markersize=7, label='y1')
plt.errorbar(time, y2, y2err, fmt='or', markersize=7, label='y2')
plt.xlabel('Time (days)')
plt.ylabel('Measurements')
plt.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
plt.grid(which='major', alpha=0.5)
plt.savefig('data2.png', bbox_inches='tight')

############## 2 datasets 
gprn = meanfield.inference(1, time, y1, y1err, y2, y2err)

nodes = [covfunc.Periodic(5, 31, 0.5)]
weight = [covfunc.SquaredExponential(5, 5), covfunc.SquaredExponential(10, 10)]
means = [meanfunc.Constant(0), meanfunc.Constant(0)]
jitter = [0.5, 0.5]

elbo, m, v = gprn.ELBOcalc(nodes, weight, means, jitter, 
                           iterations=5000, mu='init', var='init')
print('ELBO =', elbo)

tstar = np.linspace(time.min(), time.max(), 1000)

a, _, _, bb = gprn.Prediction(nodes, weight, means, jitter, tstar, m, v, separate=True)

fig = plt.figure(constrained_layout=True, figsize=(7, 7))
axs = fig.subplot_mosaic( [['predictive 1', 'node'],
                           ['predictive 1', 'node'],
                           ['predictive 2', 'weight 1'],
                           ['predictive 2', 'weight 2'],],)

axs['predictive 1'].set(xlabel='', ylabel='y1')
axs['predictive 1'].errorbar(time, y1, y1err, fmt= '.k')
axs['predictive 1'].plot(tstar, a[:,0].T, '-r')
axs['predictive 1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 1'].grid(which='major', alpha=0.5)
axs['predictive 1'].grid(which='minor', alpha=0.2)

axs['predictive 2'].set(xlabel='', ylabel='y2')
axs['predictive 2'].errorbar(time, y2, y2err, fmt= '.k')
axs['predictive 2'].plot(tstar, a[:,1].T, '-r')
axs['predictive 2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 2'].grid(which='major', alpha=0.5)
axs['predictive 2'].grid(which='minor', alpha=0.2)

axs['weight 1'].set(xlabel='', ylabel='1st weight')
axs['weight 1'].plot(tstar, bb[1][0].T, '-b')

axs['node'].set(xlabel='', ylabel='Node')
axs['node'].plot(tstar, bb[0].T, '-b')

axs['weight 2'].set(xlabel='', ylabel='2nd weight')
axs['weight 2'].plot(tstar, bb[1][1].T, '-b')

fig.savefig('componentsPlots.png', bbox_inches='tight')


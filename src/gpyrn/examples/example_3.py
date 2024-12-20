import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
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
plt.savefig('data3.png', bbox_inches='tight')

gprn = meanfield.inference(2, time, y1, y1err, y2, y2err)

nodes = [covfunc.Periodic(1, 31, 0.5),  covfunc.Matern52(1, 100)]
weight = [covfunc.SquaredExponential(20, 5), covfunc.SquaredExponential(0.1, 10),
          covfunc.SquaredExponential(10, 5), covfunc.SquaredExponential(1, 10)]
means = [meanfunc.Constant(0), meanfunc.Constant(0)]
jitter = [0.5, 0.5]

elbo, m, v = gprn.ELBOcalc(nodes, weight, means, jitter, iterations=5000, mu='init', var='init')
print('ELBO =', elbo)

tstar = np.linspace(time.min(), time.max(), 1000)
a, _, _, b = gprn.newPrediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

fig = plt.figure(constrained_layout=True, figsize=(7, 10))
axs = fig.subplot_mosaic( [['predictive 1', 'node 1'],
                           ['predictive 1', 'node 2'],
                           ['predictive 1', 'weight 1'],
                           ['predictive 2', 'weight 2'],
                           ['predictive 2', 'weight 3'],
                           ['predictive 2', 'weight 4'],],)

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

axs['node 1'].set(xlabel='', ylabel='1st Node')
axs['node 1'].plot(tstar, b[0][0].T, '-b')
axs['node 2'].set(xlabel='', ylabel='2nd Node')
axs['node 2'].plot(tstar, b[0][1].T, '-b')
axs['weight 1'].set(xlabel='', ylabel='1st weight')
axs['weight 1'].plot(tstar, b[1][0].T, '-b')
axs['weight 2'].set(xlabel='', ylabel='2nd weight')
axs['weight 2'].plot(tstar, b[1][1].T, '-b')

axs['weight 3'].set(xlabel='', ylabel='3rd weight')
axs['weight 3'].plot(tstar, b[1][2].T, '-b')
axs['weight 4'].set(xlabel='', ylabel='4th weight')
axs['weight 4'].plot(tstar, b[1][3].T, '-b')

fig.savefig('componentsPlots2.png', bbox_inches='tight')


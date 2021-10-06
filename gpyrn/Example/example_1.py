import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
plt.rcParams['figure.figsize'] = [7, 4]
from matplotlib.ticker import AutoMinorLocator


from gpyrn import meanfield

time = np.linspace(0, 100, 25)
y1 = 20*np.sin(2*np.pi*time / 31) + np.random.randn(25)
y1err = np.random.rand(25)

y2 = 25*np.sin(2*np.pi*time / 31 + 0.5*np.pi) + np.random.randn(25)
y2err = np.random.rand(25)


plt.figure()
plt.errorbar(time, y1, y1err, fmt='ob', markersize=7, label='y1')
plt.errorbar(time, y2, y2err, fmt='or', markersize=7, label='y2')
plt.xlabel('Time (days)')
plt.ylabel('Measurements')
plt.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
plt.grid(which='major', alpha=0.5)
plt.savefig('data.pdf', bbox_inches='tight')
plt.close('all')

from gpyrn import covfunc, meanfunc
############## 1 dataset 
gprn = meanfield.inference(1, time, y1, y1err)

nodes = [covfunc.Periodic(15, 31, 0.5)]
weight = [covfunc.SquaredExponential(1.1, 20)]
means = [meanfunc.Constant(0)]
jitter = [0.5]

elbo, m, v = gprn.ELBOcalc(nodes, weight, means, jitter, 
                           iterations=5000, mu='init', var='init')
print('ELBO =', elbo)

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
a, b, c = gprn.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])


plt.figure()
plt.errorbar(time, y1, y1err, fmt='ob', markersize=7, label='y1')
plt.plot(tstar, a[0], '--k', linewidth=2, label='predictive')
plt.fill_between(tstar,  bmax1.T, bmin1.T, color="grey", alpha=0.25)
plt.xlabel('Time (days)')
plt.ylabel('Measurements')
plt.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
plt.grid(which='major', alpha=0.5)
plt.savefig('dataAndPrediction.pdf', bbox_inches='tight')
plt.close('all')

############## 2 datasets 
gprn = meanfield.inference(1, time, y1, y1err, y2, y2err)

nodes = [covfunc.Periodic(15, 31, 0.5)]
weight = [covfunc.SquaredExponential(1.1, 500), covfunc.SquaredExponential(1.2, 100)]
means = [meanfunc.Constant(0), meanfunc.Constant(0)]
jitter = [0.5, 0.5]

elbo, m, v = gprn.ELBOcalc(nodes, weight, means, jitter, 
                           iterations=5000, mu='init', var='init')
print('ELBO =', elbo)

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))


a, b, c = gprn.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])
bmin2, bmax2 = a[1]-np.sqrt(b[1]), a[1]+np.sqrt(b[1])

aa, bb, cc = gprn.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))

val1Pred, val2Pred = [], []
for i, j in enumerate(values):
    val1Pred.append(a[0][j])
    val2Pred.append(a[1][j])

residuals1 = y1 - np.array(val1Pred)
residuals2 = y2 - np.array(val2Pred)

fig = plt.figure(constrained_layout=True, figsize=(7, 7))
axs = fig.subplot_mosaic( [['predictive 1', 'mean 1'],
                           ['predictive 1', 'weight 1'],
                           ['residuals 1', 'node'],
                           ['predictive 2', 'node'],
                           ['predictive 2', 'weight 2'],
                           ['residuals 2', 'mean 2'],],)

axs['predictive 1'].set(xlabel='', ylabel='y1')
axs['predictive 1'].errorbar(time, y1, y1err, fmt= '.k')
axs['predictive 1'].plot(tstar, a[0].T, '-r')
axs['predictive 1'].fill_between(tstar,  bmax1.T, bmin1.T, color="red", alpha=0.25)
axs['residuals 1'].errorbar(time, residuals1, y1err, fmt= '.k')
axs['residuals 1'].axhline(y=0, linestyle='--', color='b')
axs['residuals 1'].set(xlabel='', ylabel='')
axs['predictive 1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 1'].grid(which='major', alpha=0.5)
axs['predictive 1'].grid(which='minor', alpha=0.2)
axs['residuals 1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['residuals 1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['residuals 1'].grid(which='major', alpha=0.5)
axs['residuals 1'].grid(which='minor', alpha=0.2)


axs['predictive 2'].set(xlabel='', ylabel='y2')
axs['predictive 2'].errorbar(time, y2, y2err, fmt= '.k')
axs['predictive 2'].plot(tstar, a[1].T, '-r')
axs['predictive 2'].fill_between(tstar,  bmax2.T, bmin2.T, color="red", alpha=0.25)
axs['residuals 2'].errorbar(time, residuals2,  y2err, fmt= '.k')
axs['residuals 2'].axhline(y=0, linestyle='--', color='b')
axs['residuals 2'].set(xlabel='Time (days)')
axs['predictive 2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive 2'].grid(which='major', alpha=0.5)
axs['predictive 2'].grid(which='minor', alpha=0.2)
axs['residuals 2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['residuals 2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['residuals 2'].grid(which='major', alpha=0.5)
axs['residuals 2'].grid(which='minor', alpha=0.2)

axs['mean 1'].set(xlabel='', ylabel='y1 mean')
axs['mean 1'].plot(tstar, means[0](tstar), '-b')
axs['weight 1'].set(xlabel='', ylabel='y1 weight')
axs['weight 1'].plot(tstar, bb[1,0].T, '-b')

axs['node'].set(xlabel='', ylabel='Node')
axs['node'].plot(tstar, bb[0,0].T, '-b')

axs['mean 2'].set(xlabel='Time (days)')
axs['mean 2'].plot(tstar, means[1](tstar), '-b')
axs['weight 2'].set(xlabel='', ylabel='y2 weight')
axs['weight 2'].plot(tstar, bb[1,1].T, '-b')

fig.savefig('componentsPlots.pdf', bbox_inches='tight')
plt.close('all')

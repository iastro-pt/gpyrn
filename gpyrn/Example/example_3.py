import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from gprn import meanField as meanfield
from gprn import covFunction as covfunc
from gprn import meanFunction as meanfunc

time = np.linspace(0, 100, 25)
y1 = 20*np.sin(2*np.pi*time / 31)
y1err = np.random.rand(25)

plt.figure()
plt.errorbar(time, y1, y1err, fmt='ob', markersize=7, label='y1')
plt.xlabel('Time (days)')
plt.ylabel('Measurements')
plt.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
plt.grid(which='major', alpha=0.5)
# plt.savefig('data2.png', bbox_inches='tight')
# plt.close('all')

gprn = meanfield.inference(2, time, y1, y1err)

nodes = [covfunc.Periodic(1, 11, 0.5),  covfunc.SquaredExponential(0.1, 10)]
weight = [covfunc.SquaredExponential(20, 5), covfunc.SquaredExponential(0.1, 1)]
means = [meanfunc.Constant(0)]
jitter = [0.5]

elbo, m, v = gprn.ELBOcalc(nodes, weight, means, jitter, iterations=5000, mu='init', var='init')
print('ELBO =', elbo)

tstar = np.linspace(time.min(), time.max(), 1000)
a, b, c = gprn.newPrediction(nodes, weight, means, jitter, tstar, m, v)
# aa, bb, cc = gprn.Prediction(nodes, weight, means, jitter, tstar, m, v, 
#                               separate=True)
plt.figure()
plt.plot(time, y1,'o')
plt.plot(tstar, a,'-')
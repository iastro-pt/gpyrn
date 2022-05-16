import os, sys
import numpy as np
import matplotlib.pyplot as plt


def equal_y_axis(ax):
    ylim = ax.get_ylim()
    m = np.max(np.abs(ylim))
    ax.set_ylim(-m, m)


def plot_prediction(gprn, nn=1000, tstar=None, over=0.2, title=None):
    if tstar is None:
        mi, ma = gprn.time.min(), gprn.time.max()
        tptp = gprn.time.ptp()
        tstar = np.linspace(mi - over * tptp, ma + over * tptp, nn)

    a, v = gprn._Prediction()
    aa, vv, bb = gprn._Prediction(tstar=tstar, separate=True)
    ss = np.sqrt(vv)

    if gprn.p == 1:
        layout = [
            ['pred1', 'd1'],
            ['resid1', 'node'],
        ]
    elif gprn.p == 2:
        layout = [
            ['pred1', 'd1'],
            ['pred1', 'd1'],
            ['resid1', 'node'],
            ['pred2', 'node'],
            ['pred2', 'd2'],
            ['resid2', 'd2'],
        ]
    elif gprn.p == 3:
        layout = [
            ['node', 'node'],
            ['node', 'node'],
            ['pred1', 'd1'], ['pred1', 'd1'],# ['pred1', 'd1'],
            ['resid1', 'd1'],
            ['pred2', 'd2'], ['pred2', 'd2'],# ['pred2', 'd2'],
            ['resid2', 'd2'],
            ['pred3', 'd3'], ['pred3', 'd3'],# ['pred3', 'd3'],
            ['resid3', 'd3'],
        ]

    fig = plt.figure(constrained_layout=False, figsize=(10, 4 * gprn.p))
    if title is not None:
        fig.suptitle(title)

    axs = fig.subplot_mosaic(layout)

    means = []
    for i in range(gprn.p):
        axs[f'pred{i+1}'].set(xlabel='', ylabel=f'y{i+1}')
        axs[f'pred{i+1}'].errorbar(gprn.time, gprn.y[i], gprn.yerr[i],
                                   fmt='ok', ms=2)

        pred = aa[:, i].T
        std = ss[:, i].T
        axs[f'pred{i+1}'].fill_between(tstar, pred - std, pred + std, alpha=0.1)
        axs[f'pred{i+1}'].plot(tstar, pred)
        axs[f'pred{i+1}'].grid(which='major', alpha=0.5)
        axs[f'pred{i+1}'].grid(which='minor', alpha=0.2)


        resid = gprn.y[i] - a[:, i]
        axs[f'resid{i+1}'].errorbar(gprn.time, resid, gprn.yerr[i], fmt='ok',
                                    ms=2)
        axs[f'resid{i+1}'].axhline(y=0.0, ls='--', color='k', alpha=0.2)
        axs[f'resid{i+1}'].set_title(f'std: {resid.std():.2f}', loc='right',
                                     fontsize=10)
        equal_y_axis(axs[f'resid{i+1}'])
        axs[f'resid{i+1}'].set_ylabel('residuals')

        try:
            means.append(gprn.means[i](tstar))
        except TypeError:
            means.append(np.zeros_like(tstar))
        # axs[f'pred{i+1}'].plot(tstar, means[-1], ls='--')

    for i in range(gprn.p):
        axs[f'd{i+1}'].set(xlabel='')
        axs[f'd{i+1}'].set_ylabel('weight', color='C0')
        axs[f'd{i+1}'].set_title('weight(s) and mean', loc='left', fontsize=10)
        # weight(s)
        for w in bb[1][i::gprn.p]:
            axs[f'd{i+1}'].plot(tstar, w, alpha=0.6)
        axs[f'd{i+1}'].tick_params(axis='y', labelcolor='C0')

        # mean
        ax2 = axs[f'd{i+1}'].twinx()
        ax2.plot(tstar, means[i], color='k', ls='--', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='k')
        ax2.set_ylabel('mean', color='k')


    # axs['d2'].set(xlabel='')
    # axs['d2'].set_ylabel('weight', color='C0')
    # axs['d2'].set_title('weight(s) and mean', loc='left', fontsize=10)
    # axs['d2'].plot(tstar, bb[1][1 * gprn.q:].T, color='C0', alpha=0.6)
    # axs['d2'].tick_params(axis='y', labelcolor='C0')

    # ax2 = axs['d2'].twinx()
    # ax2.plot(tstar, means[1], color='C1', ls='--', alpha=0.6)
    # ax2.tick_params(axis='y', labelcolor='C1')
    # ax2.set_ylabel('mean', color='C1')

    # node
    axs['node'].set(xlabel='')
    _s = '' if gprn.q == 1 else 's'
    axs['node'].set_title('node' + _s, loc='left', fontsize=10)
    axs['node'].plot(tstar, bb[0].T, '-')

    equal_y_axis(axs['node'])
    ax0 = axs['pred1']
    for ax in axs.values():
        ax.sharex(ax0)

    for i in range(gprn.p - 1):
        plt.setp(axs[f'pred{i+1}'].get_xticklabels(), visible=False)
        plt.setp(axs[f'resid{i+1}'].get_xticklabels(), visible=False)
        plt.setp(axs[f'd{i+1}'].get_xticklabels(), visible=False)
    try:
        plt.setp(axs[f'pred{i+2}'].get_xticklabels(), visible=False)
    except KeyError:
        pass

    for ax in axs.values():
        ax.axvspan(gprn.time[0], gprn.time[-1], color='k', alpha=0.05,
                   zorder=-1)

    fig.tight_layout()

    return fig, axs

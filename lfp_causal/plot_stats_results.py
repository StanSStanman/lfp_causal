import xarray as xr
import numpy as np
import os.path as op
import matplotlib.pyplot as plt


def plot_avg_stat_res(stats_dir, regressors, treshold=0.05):
    assert isinstance(stats_dir, str)
    assert isinstance(regressors, list), \
        AssertionError('regressors must be a list of str')
    print('Directory:', stats_dir)

    mi = xr.open_dataset(op.join(stats_dir, 'mi_results.nc'))
    pv = xr.open_dataset(op.join(stats_dir, 'pv_results.nc'))

    for r in regressors:
        times = mi.times.values
        labels = mi.roi.values
        mi_data = mi[r].values.T
        pv_data = pv[r].values.T

        fig, ax = plt.subplots(1, 1)
        fig.suptitle(r)
        for m, p, l, i in zip(mi_data, pv_data, labels, range(len(labels))):
            _p = m.copy()
            _p[p > treshold] = np.nan
            ax.plot(times, m, label=l, color='C%i' % i, linestyle='--')
            ax.plot(times, _p, color='C%i' % i, linewidth=2.5)
            ax.axvline(0, -1, 1, color='k', linestyle='--')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    stats_dir = '/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/{0}_{1}'
    freqs = [(8, 15), (15, 30), (25, 50), (40, 70)]
    regressors = ['Correct', 'Reward',
                  'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
                  '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                  'learn_5t', 'learn_2t', 'early_late_cons',
                  'P(R|C)', 'P(R|nC)', 'P(R|Cho)',
                  'dP', 'log_dP', 'delta_dP',
                  'surprise', 'surprise_bayes', 'rpe']

    for f in freqs:
        plot_avg_stat_res(stats_dir.format(f[0], f[1]), regressors)

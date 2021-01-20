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


def plot_tf_stat_res(stats_dir, regressors, treshold=0.05):
    assert isinstance(stats_dir, str)
    assert isinstance(regressors, list), \
        AssertionError('regressors must be a list of str')
    print('Directory:', stats_dir)

    mi = xr.open_dataset(op.join(stats_dir, 'mi_results.nc'))
    pv = xr.open_dataset(op.join(stats_dir, 'pv_results.nc'))

    for r in regressors:
        times = mi.times.values
        freqs = mi.freqs.values
        labels = mi.roi.values
        mi_data = mi[r].values.transpose(2, 0, 1)
        pv_data = pv[r].values.transpose(2, 0, 1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.suptitle(r)
        for m, p, l, i, ax in zip(mi_data, pv_data, labels,
                                  range(len(labels)), (ax1, ax2, ax3)):
            _p = m.copy()
            m[p > treshold] = np.nan
            _p[p < treshold] = np.nan
            ax.set_title(l)
            ax.pcolormesh(times, freqs, m)
            ax.pcolormesh(times, freqs, _p, alpha=0.5)
            # ax.plot(times, m, label=l, color='C%i' % i, linestyle='--')
            # ax.plot(times, _p, color='C%i' % i, linewidth=2.5)
            ax.axvline(0, -1, 1, color='w', linestyle='--')
        # plt.colorbar()
        plt.show()


if __name__ == '__main__':
    monkey = 'freddie'
    condition = 'easy'
    event = 'trig_off'

    stats_dir = op.join('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/',
                        monkey, condition, event, '{0}_{1}')
    freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    # freqs = [(5, 120)]
    regressors = ['Correct', 'Reward',
                  'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
                  'RnR|C', 'RnR|nC',
                  '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                  'learn_5t', 'learn_2t', 'early_late_cons',
                  'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
                  'dP', 'log_dP', 'delta_dP',
                  'surprise', 'surprise_bayes', 'rpe',
                  'q_pcorr', 'q_pincorr', 'q_dP',
                  'q_entropy', 'q_rpe', 'q_absrpe',
                  'q_shann_surp', 'q_bayes_surp']

    # for f in freqs:
    #     plot_avg_stat_res(stats_dir.format(f[0], f[1]), regressors)
    #     # plot_tf_stat_res(stats_dir.format(f[0], f[1]), regressors)

    for r in regressors:
        for f in freqs:
            plot_avg_stat_res(stats_dir.format(f[0], f[1]), [r])
            # plot_tf_stat_res(stats_dir.format(f[0], f[1]), [r])

    # regressors = ['early_late_cons']
    #
    # stats_dir = '/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/ffx/{0}_{1}'
    # freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    # for f in freqs:
    #     plot_avg_stat_res(stats_dir.format(f[0], f[1]), regressors)
    #
    # stats_dir = '/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/{0}_{1}_tf'
    # freqs = [(5, 120)]
    # for f in freqs:
    #     plot_tf_stat_res(stats_dir.format(f[0], f[1]), regressors)

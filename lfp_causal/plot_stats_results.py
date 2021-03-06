import xarray as xr
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt


def plot_avg_stat_res(stats_dir, regressors, treshold=0.05):
    plt.close('all')
    assert isinstance(stats_dir, str)
    assert isinstance(regressors, list), \
        AssertionError('regressors must be a list of str')
    print('Directory:', stats_dir)

    mi = xr.open_dataset(op.join(stats_dir, 'mi_results.nc'))
    pv = xr.open_dataset(op.join(stats_dir, 'pv_results.nc'))

    mi = mi.sortby('roi')
    pv = pv.sortby('roi')

    figures = []
    for r in regressors:
        times = mi.times.values
        labels = mi.roi.values
        mi_data = mi[r].values.T
        pv_data = pv[r].values.T

        fig, ax = plt.subplots(1, 1)
        fig.suptitle(r)
        for m, p, l, i in zip(mi_data, pv_data, labels, range(len(labels))):
            m = np.convolve(m, np.hanning(50), mode='same')
            _p = m.copy()
            _p[p > treshold] = np.nan
            ax.plot(times, m, label=l, color='C%i' % i, linestyle='--')
            ax.plot(times, _p, color='C%i' % i, linewidth=2.5)
            ax.axvline(0, -1, 1, color='k', linestyle='--')
        plt.legend()
        figures.append(fig)
        plt.show()
    return figures


def plot_tf_stat_res(stats_dir, regressors, treshold=0.05):
    assert isinstance(stats_dir, str)
    assert isinstance(regressors, list), \
        AssertionError('regressors must be a list of str')
    print('Directory:', stats_dir)

    mi = xr.open_dataset(op.join(stats_dir, 'mi_results.nc'))
    pv = xr.open_dataset(op.join(stats_dir, 'pv_results.nc'))

    figures = []
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
        figures.append(fig)
        plt.show()
    return figures


def plot_band_stat_res(stats_dirs, regressors, treshold=0.05):
    plt.close('all')
    assert isinstance(stats_dirs, list)
    assert isinstance(regressors, list), \
        AssertionError('regressors must be a list of str')
    print('Directories:', stats_dirs)

    mis = []
    pvs = []
    for stats_dir in stats_dirs:
        mi = xr.open_dataset(op.join(stats_dir, 'mi_results.nc'))
        pv = xr.open_dataset(op.join(stats_dir, 'pv_results.nc'))

        mi = mi.sortby('roi')
        pv = pv.sortby('roi')

        mis.append(mi)
        pvs.append(pv)

    mis = xr.concat(mis, dim='roi')
    pvs = xr.concat(pvs, dim='roi')

    rois = ['associative striatum', 'limbic striatum', 'motor striatum']
    # rois = ['unique']
    fr_names = ['alpha', 'beta', 'gamma', 'hga']

    figures = []
    for r in regressors:
        times = mi.times.values
        # labels = mi.roi.values
        # mi_data = mi[r].values.T
        # pv_data = pv[r].values.T

        fig, ax = plt.subplots(1, 3, figsize=(25, 5))
        fig.suptitle(r)
        for _ax, _roi in enumerate(rois):
            mi_data = mis[r].loc[:, _roi].values.T
            pv_data = pvs[r].loc[:, _roi].values.T
            for i, (m, p, fn) in enumerate(zip(mi_data, pv_data, fr_names)):
                m = np.convolve(m, np.hanning(150), mode='same')
                _p = m.copy()
                _p[p > treshold] = np.nan
                ax[_ax].plot(times, m, label=fn, color='C%i' % i,
                             linestyle='--')
                ax[_ax].plot(times, _p, color='C%i' % i, linewidth=2.5)
            ax[_ax].axvline(0, -1, 1, color='k', linestyle='--')
            ax[_ax].set_title(_roi)
            ax[_ax].legend()
        figures.append(fig)
        plt.show()
    return figures


if __name__ == '__main__':
    monkey = 'teddy'
    condition = 'easy'
    event = 'trig_off'
    norm = 'fbline_relchange'

    stats_dir = op.join('/media/jerry/TOSHIBA EXT/data/stats/lfp_causal/',
                        monkey, condition, event, norm, '{0}_{1}_mt')
    fig_dir = op.join('/media/jerry/TOSHIBA EXT/data/plots',
                      monkey, condition, event, norm)

    freqs = [(8, 15), (15, 30), (25, 45), (40, 70), (60, 120)]
    freqs = [(8, 12), (15, 35), (40, 65), (70, 120)]
    # freqs = [(5, 120)]
    regressors = ['Correct', 'Reward',
                  'is_R|C', 'is_nR|C', 'is_R|nC', 'is_nR|nC',
                  'RnR|C', 'RnR|nC',
                  '#R', '#nR', '#R|C', '#nR|C', '#R|nC', '#nR|nC',
                  'learn_5t', 'learn_2t', 'early_late_cons',
                  'P(R|C)', 'P(R|nC)', 'P(R|Cho)', 'P(R|A)',
                  'pra_mean',
                  'dP', 'log_dP', 'delta_dP',
                  'surprise', 'surprise_bayes', 'act_surp_bayes', 'rpe',
                  'q_pcorr', 'q_pincorr', 'q_dP',
                  'q_entropy', 'q_rpe', 'q_absrpe',
                  'q_shann_surp', 'q_bayes_surp',
                  'pra_rew', 'pra_mean', 'evl', 'expexp']
    regressors = ['q_absrpe', 'Reward', 'P(R|A)', 'delta_dP', 'rpe', 'q_rpe',
                  'expexp']
    regressors = ['Reward_30t', 'q_rpe_30t']

    # for f in freqs:
    #     plot_avg_stat_res(stats_dir.format(f[0], f[1]), regressors)
    #     # plot_tf_stat_res(stats_dir.format(f[0], f[1]), regressors)

    # for r in regressors:
    #     for f in freqs:
    #         figs = plot_avg_stat_res(stats_dir.format(f[0], f[1]), [r])
    #         # plot_tf_stat_res(stats_dir.format(f[0], f[1]), [r])
    #         # figdir = op.join(fig_dir, r.replace('|', '_'))
    #         # os.makedirs(figdir, exist_ok=True)
    #         # figname = op.join(figdir, '{0}_{1}'.format(f[0], f[1]))
    #         # plt.savefig(figname)

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

    for r in regressors:
        stats_dirs = []
        for f in freqs:
            stats_dirs.append(stats_dir.format(f[0], f[1]))
        plot_band_stat_res(stats_dirs, [r])

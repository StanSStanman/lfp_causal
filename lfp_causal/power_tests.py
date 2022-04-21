import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo

def plot_pow_time_avg(monkeys, powers, blines, conds, lbad, tbad):
    d = {m: {0: [], 1: []} for m in monkeys}
    # n = {m: {0: [], 1: []} for m in monkeys}
    for m in monkeys:
        for p, b, c, lb, tb in zip(powers, blines, conds, lbad, tbad):
            if m in p:
                print(p)
                _p = xr.load_dataset(p)
                _p = normalize_power(_p, 'fbline_relchange',
                                     (-.6, -0.1), file=b)
                _p = _p.to_array().mean('variable')
                _p = _p.loc[dict(times=slice(0., .8))]
                _p = _p.loc[dict(freqs=slice(0., 51.))]
                _p = _p[:25, :, :]

                xls = pd.read_excel(c, index_col=0)
                _r = xls['Reward'].values
                if len(lb) != 0:
                    _r = np.delete(_r, lb)
                if len(tb) != 0:
                    _r = np.delete(_r, tb)
                _r = _r[:25]

                _p['trials'] = _r
                print(_r)
                for i in [0, 1]:
                    if i in _p.trials:
                        __p = _p.loc[{'trials': i}]
                        d[m][i].append(__p)
                        # if __p.ndim == 2:
                        #     d[m][i].append(__p.drop('trials'))
                        # elif __p.ndim == 3:
                        #     d[m][i].append(__p.sum('trials'))
                        # assert 'trials' in d[m][i][-1].dims
                        # n[m][i].append(list(_p.trials).count(i))

        for i in [0, 1]:
            d[m][i] = xr.concat(d[m][i], dim='trials')
            # n[m][i] = np.sum(n[m][i])
            # d[m][i] = d[m][i].sum('trials') / n[m][i]
        t, p = ss.ttest_ind(d[m][1].values, d[m][0].values, axis=0)
        # t, p = ss.ttest_rel(d[m][1].values, d[m][0].values, axis=0)
        pc = p * np.multiply(*t.shape)
        times, freqs = d[m][0].times, d[m][0].freqs

        d[m] = {'tvals': xr.DataArray(data=t, coords=[freqs, times],
                                      dims=['freqs', 'times']),
                'pvals': xr.DataArray(data=p, coords=[freqs, times],
                                      dims=['freqs', 'times']),
                'corrected_p': xr.DataArray(data=pc, coords=[freqs, times],
                                            dims=['freqs', 'times'])}

    fig, axs = plt.subplots(1, len(monkeys), figsize=(18, 5))
    fig.subplots_adjust(left=0.2)
    cbar_ax = fig.add_axes([0.1, 0.15, 0.05, 0.7])
    mk_names = ['Monkey F', 'Monkey T']
    for i, m in enumerate(monkeys):
        _ds = -np.log10(d[m]['corrected_p'])

        # kw_pcmesh = dict(aspect='auto', origin='upper', interpolation='none',
        #                  extent=[times[0], times[-1], freqs[-1], freqs[0]])
        # kw_contour = dict(#origin=kw_pcmesh['origin'],
        #                   origin='upper',
        #                   extent=kw_pcmesh['extent'],
        #                   levels=[.05],
        #                   colors=['white'])

        # im = xr.plot.pcolormesh(_ds, x='times', y='freqs', cmap='inferno',
        #                         ax=axs[i],
        #                         shading='nearest', rasterized=True,
        #                         add_colorbar=False,
        #                         vmin=0)
        im = axs[i].pcolormesh(_ds.times, _ds.freqs, _ds,
                                shading='nearest', rasterized=True,
                                cmap='inferno', vmin=0)
        # cs = axs[i].contour(d[m]['corrected_p'], **kw_contour)
        # axs[i].set_title(m)
        axs[i].set_title(mk_names[i])
    cb = fig.colorbar(im, cax=cbar_ax, ticklocation='left')
    cb.ax.set_ylabel('-log10(pvals)')
    # for m in monkeys:
    #     # _ds = (d[m][1] - d[m][0]).mean('times')
    #     _ds = (d[m][1] - d[m][0])
    #     # _ds.values = (_ds.values - _ds.mean('times', keepdims=True).values) \
    #     #     / _ds.std('times', keepdims=True).values
    #     _ds.values = zscore(_ds.values, axis=0)
    #     _ds = _ds.mean('times')
    #     axs[-1].plot(_ds.values.squeeze(), label=m)
    # axs[-1].legend()
    fig.text(0.5, 0.04, 'Time relative to outcome (s)', ha='center',
             fontsize=20)
    fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',
             fontsize=20)
    plt.show()
    return


if __name__ == '__main__':
    monkeys = ['freddie', 'teddy']
    conditions = ['easy', 'hard']
    # monkeys = ['teddy']
    # conditions = ['hard']
    event = 'trig_off'
    norm = 'fbline_relchange'
    file = '{0}_pow_8_120.nc'.format(event)
    # file = '{0}_pow_beta_mt.nc'.format(event)
    bline = 'cue_on_pow_8_120.nc'
    # bline = 'cue_on_pow_beta_mt.nc'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['motor striatum', 'limbic striatum']

    rej_files = []
    rej_files += ['1204', '1217', '1231', '0944', # Bad sessions
                  '0845', '0847', '0911', '0939', '0946', '0963', '0984',
                  '1036', '1231', '1233', '1234', '1514', '1699',

                  '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                  '0977', '0985', '1037', '1280']
    rej_files += ['0210', '0219', '0221', '0225', '0226', '0227', '0230',
                  '0252', '0268', '0276', '0277', '0279', '0281', '0282',
                  '0283', '0285', '0288', '0290', '0323', '0362', '0365',
                  '0393', '0415', '0447', '0449', '0450', '0456', '0541',
                  '0573', '0622', '0628', '0631', '0643', '0648', '0653',
                  '0660', '0688', '0689', '0690', '0692', '0697', '0706',
                  '0710', '0717', '0718', '0719', '0713', '0726', '0732',

                  '0220', '0223', '0271', '0273', '0278', '0280', '0284',
                  '0289', '0296', '0303', '0363', '0416', '0438', '0448',
                  '0521', '0618', '0656', '0691', '0693', '0698', '0705',
                  '0707', '0711', '0712', '0716', '0720', '0731']

    rej_files += ['0900', '1512', '1555', '1682',
                  '0291', '0368', '0743']

    rej_files += ['0231', '0272', '0274', '0666', '0941', '0855', '0722',
                  '0725', '1397', '1398', '1701']

    # rej_files += ['1248', '1007', '0831', '0832', '0995', '1398', '1237',
    #               '1515', '0934', '1232', '1281', '0989', '0991', '0993',
    #
    #               '0449', '0542', '0629', '0314', '0687', '0672', '0676',
    #               '0548',  '0243']

    # acc_files = ['0885', #'0883', '0881', '0878', '1080', '1075', #'1256', '0817'
    #              ]

    all_files = []
    all_bline = []
    all_conds = []
    log_bads = []
    bad_epo = []

    for monkey in monkeys:
        # all_files = []
        # all_bline = []
        # all_conds = []
        # log_bads = []
        # bad_epo = []

        for condition in conditions:

            dirs = get_dirs('local', 'lfp_causal')
            directory = dirs['pow'].format(monkey, condition)
            epo_dir = dirs['epo'].format(monkey, condition)
            regr_dir = dirs['reg'].format(monkey, condition)
            rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                               'files_info.xlsx')

            for sect in sectors:
                fid = read_sector(rec_info, sect)
                fid = fid[fid['quality'] <= 3]
                # fid = fid[fid['neuron_type'] == 'TAN']

                # all_files = []
                # all_bline = []
                # all_conds = []
                # log_bads = []
                # bad_epo = []
                for fs in fid['file']:
                    fname = op.join(directory, fs, file)
                    bname = op.join(directory, fs, bline)
                    rname = op.join(regr_dir, '{0}.xlsx'.format(fs))
                    if op.exists(fname) and fs not in rej_files:
                    # if op.exists(fname) and fs in acc_files:
                        all_files.append(fname)
                        all_bline.append(bname)
                        all_conds.append(rname)

                        fname_epo = op.join(epo_dir,
                                            '{0}_{1}_epo.fif'.format(fs, event))
                        lb = get_log_bad_epo(fname_epo)
                        log_bads.append(lb)
                        be = get_ch_bad_epo(monkey, condition, fs,
                                            fname_info=rec_info)
                        bad_epo.append(be)
                        # plot_avg_tf([fname], [bname])

            # mu = RepeatedTimer(1, memory_usage)
            # cu = RepeatedTimer(1, cpu_usage)
            # import time
            # time.sleep(5)

            # plot_avg_tf(all_files, all_bline)
        # plot_pow_cond(all_files, all_conds, 'Reward', log_bads, bad_epo, norm,
        #               (-.6, -0.1), all_bline, c_type='d', bins=2, diff=True,
        #               plot='pcm', crop=(-0., .8), freqs=(8., 51.), maxtr=25)
        # plot_avg_tf(all_files, all_bline, plot='line')
    plot_pow_time_avg(monkeys, all_files, all_bline,
                      all_conds, log_bads, bad_epo)

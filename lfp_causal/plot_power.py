import xarray as xr
import os
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import zscore
from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)


def plot_avg_tf(powers, blines=None, plot='pcm'):
    avg_pow = None
    if blines is None:
        blines = range(len(powers))
    for p, b in zip(powers, blines):
        print('Loading', p)
        pow = xr.load_dataset(p)
        if isinstance(b, (int, float)):
            pow = normalize_power(pow, 'relchange', (-1.8, -1.3))
        elif isinstance(b, str):
            pow = normalize_power(pow, 'fbline_relchange', (-.55, -0.05),
                                  file=b)

        pow = pow.loc[dict(times=slice(0., .801))]
        pow = pow.loc[dict(freqs=slice(8, 51))]

        pow = pow.to_array().squeeze(axis=0)
        times = pow.times.values
        freqs = pow.freqs.values

        # Cut at 25 trials
        pow = pow[:25, :, :]

        # pow = pow.mean('trials')

        if avg_pow is None:
            avg_pow = pow
        else:
            avg_pow = np.concatenate((avg_pow, pow), axis=0)

        # plt.pcolormesh(times, freqs, pow.mean('trials'), shading='nearest')
        # plt.colorbar()
        # plt.show()
    avg_pow = avg_pow.mean('trials')
    # avg_pow = avg_pow.mean(0)
    vmin, vmax = np.percentile(avg_pow, [2.5, 97.5]).round(3)
    # avg_pow.plot(cmap='viridis')
    fig, ax = plt.subplots(1, 1)
    if plot is 'line':
        ax.plot(times, avg_pow.T)
    elif plot is 'pcm':
        cm = ax.pcolormesh(times, freqs, avg_pow, vmin=vmin, vmax=vmax,
                           shading='nearest', rasterized=True)
        # ax.set_xticks(np.arange(len(times)))
        # ax.set_xticklabels(times)
        # ax.set_yticks(np.arange(len(freqs)))
        # ax.set_yticklabels(freqs)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.colorbar(cm)
    plt.show()
    return


def plot_pow_cond(powers, conditions, pick, log_bads, bad_trials, norm, tbline,
                  fbline, c_type='d', bins=2, diff=True, plot='pcm',
                  crop=None, freqs=None, maxtr=25):
    _reg = []
    for con, lb, bt in zip(conditions, log_bads, bad_trials):
        xls = pd.read_excel(con, index_col=0)
        reg = xls[pick].values

        if len(lb) != 0:
            reg = np.delete(reg, lb)
        if len(bt) != 0:
            reg = np.delete(reg, bt)

        # Cut at 25 trials
        if maxtr is not None:
            reg = reg[:maxtr]

        _reg.append(reg)

    if c_type == 'c':
        bc, bv = np.histogram(np.concatenate(_reg), bins)
        b_reg = []
        for _r in _reg:
            b_r = _r.copy()
            for _b in range(bins):
                ibr = np.where(np.logical_and(_r >= bv[_b], _r <= bv[_b + 1]))
                b_r[ibr] = _b
            b_reg.append(b_r)
    elif c_type == 'd':
        b_reg = _reg
    del _reg

    if len(b_reg) > 1:
        uni = np.unique(np.concatenate(b_reg))
    else:
        uni = np.unique(b_reg)
    if fbline is None or isinstance(fbline, (int, float)):
        fbline = np.zeros(len(powers))

    b_data = [[] for i in uni]
    for pow, br, fbl in zip(powers, b_reg, fbline):
        data = xr.load_dataset(pow)
        print([k for k in data.keys()])
        data = normalize_power(data, norm, tbline, file=fbl)
        data = data.to_array().mean('variable')

        if crop is not None:
            data = data.loc[dict(times=slice(*crop))]
        if freqs is not None:
            data = data.loc[dict(freqs=slice(*freqs))]
        if maxtr is not None:
            data = data[:maxtr, :, :]
        # Crop freqs


        # data = data.loc[dict(times=slice(-.2, .8))]
        # data = data.isel(freqs=np.arange(0, len(data.freqs), 4))
        # plt.pcolormesh(data.mean('freqs'), vmin=np.percentile(data, 5),
        #                vmax=np.percentile(data, 95))
        # plt.colorbar()
        # plt.show()
        #####
        # for ff in range(data.shape[1]):
        #     plt.pcolormesh(data[:, ff, :])#, vmin=np.percentile(data[:, ff, :], 5),
        #                    # vmax=np.percentile(data[:, ff, :], 95))
        #     plt.colorbar()
        #     plt.show()
        # plt.close()
        for _iu, _u in enumerate(uni):
            _idx = np.where(br == _u)
            b_data[_iu].append(data[_idx])
    for bd in range(len(b_data)):
        b_data[bd] = np.concatenate(b_data[bd], axis=0)

    n_axs = len(b_data)

    if diff is True:
        fig, axs = plt.subplots(1, n_axs + 1, figsize=(18, 5))
    else:
        fig, axs = plt.subplots(1, n_axs, figsize=(18, 5))

    if plot == 'line':
        for ax in range(n_axs):
            for _if, f in enumerate(data.freqs):
                axs[ax].plot(data.times, b_data[ax].mean(0)[_if], label=str(f))
            axs[ax].axvline(0, 0, 1, color='k', linestyle='--')
            axs[ax].set_title('{0} = {1}'.format(pick, str(uni[ax])))
        if diff == True:
            for _if, f in enumerate(data.freqs):
                axs[n_axs].plot(data.times,
                                b_data[-1].mean(0)[_if] -
                                b_data[0].mean(0)[_if],
                                label=str(f))
            axs[n_axs].axvline(0, 0, 1, color='k', linestyle='--')
            axs[n_axs].set_title('{0} = {1} - {2}'.
                                 format(pick, str(uni[-1]),
                                        str(uni[0])))

    elif plot == 'pcm':
        for ax in range(n_axs):
            pc = axs[ax].pcolormesh(data.times.values, data.freqs.values,
                                    b_data[ax].mean(0), linewidth=0.,
                                    shading='nearest', rasterized=True,
                                    vmin=-3.5, vmax=3.5)
            # axs[ax].set_yticks(np.arange(len(data.freqs))[::2])
            # axs[ax].set_xticks(np.arange(len(data.times.values))[::500])
            # axs[ax].set_yticklabels(data.freqs.values.round(3)[::2])
            # axs[ax].set_xticklabels(data.times.values[::500])
            axs[ax].set_title('{0} = {1}'.format(pick, str(uni[ax])))
        if diff is True:
            pc = axs[n_axs].pcolormesh(data.times.values, data.freqs.values,
                                       b_data[-1].mean(0) - b_data[0].mean(0),
                                       linewidth=0., shading='nearest',
                                       rasterized=True, vmin=-3.5, vmax=3.5)
            # axs[n_axs].set_yticks(np.arange(len(data.freqs))[::2])
            # axs[n_axs].set_xticks(np.arange(len(data.times.values))[::500])
            # axs[n_axs].set_yticklabels(data.freqs.values.round(3)[::2])
            # axs[n_axs].set_xticklabels(data.times.values[::500])
            axs[n_axs].set_title('{0} = {1} - {2}'.
                                 format(pick, str(uni[-1]),
                                        str(uni[0])))
    if plot == 'pcm':
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(pc, cax=cbar_ax)
    plt.show()

    return


def plot_pow_time_avg(monkeys, powers, blines, conds, lbad, tbad):
    d = {m: {0: [], 1: []} for m in monkeys}
    n = {m: {0: [], 1: []} for m in monkeys}
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
                        if __p.ndim == 2:
                            d[m][i].append(__p.drop('trials'))
                        elif __p.ndim == 3:
                            d[m][i].append(__p.sum('trials'))
                        # assert 'trials' in d[m][i][-1].dims
                        n[m][i].append(list(_p.trials).count(i))

        for i in [0, 1]:
            d[m][i] = xr.concat(d[m][i], dim='trials')
            n[m][i] = np.sum(n[m][i])
            d[m][i] = d[m][i].sum('trials') / n[m][i]

    fig, axs = plt.subplots(1, len(monkeys) + 1, figsize=(18, 5),
                            sharex=True, sharey=True)
    fig.subplots_adjust(left=0.2)
    cbar_ax = fig.add_axes([0.1, 0.15, 0.05, 0.7])
    for i, m in enumerate(monkeys):
        _ds = d[m][1] - d[m][0]
        im = xr.plot.pcolormesh(_ds, x='Time relative to outcome (s)',
                                y='Frequency (Hz)', ax=axs[i],
                                shading='nearest', rasterized=True,
                                vmin=-3.5, vmax=3.5, add_colorbar=False)
        # im = axs[i].pcolormesh(_ds.times, _ds.freqs, _ds,
        #                         shading='nearest', rasterized=True,
        #                         vmin=-3.5, vmax=3.5)
        axs[i].set_title(m)
    fig.colorbar(im, cax=cbar_ax, ticklocation='left')
    # for m in monkeys:
    #     # _ds = (d[m][1] - d[m][0]).mean('times')
    #     _ds = (d[m][1] - d[m][0])
    #     # _ds.values = (_ds.values - _ds.mean('times', keepdims=True).values) \
    #     #     / _ds.std('times', keepdims=True).values
    #     _ds.values = zscore(_ds.values, axis=0)
    #     _ds = _ds.mean('times')
    #     axs[-1].plot(_ds.values.squeeze(), label=m)
    axs[-1].legend()
    plt.show()
    return


if __name__ == '__main__':
    # pow_fname = ['/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/freddie/easy/pow/0880/trig_off_pow_8_120.nc']
    # bline = ['/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/freddie/easy/pow/0880/cue_on_pow_8_120.nc']
    # plot_avg_tf(pow_fname, blines=bline, plot='pcm')
    # breakpoint()
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

            # mu.stop()
            # cu.stop()
            # print(cpu_usage())





# folder = '/media/jerry/TOSHIBA EXT/data/db_lfp/lfp_causal/freddie/easy/pow/0990'
#
# data = xr.load_dataset(op.join(folder, 'trig_off_pow_8_120_sl.nc'))
#
# norm_data = normalize_power(data, 'relchange', (-1.8, -1.3))
# norm_data_cut = norm_data.loc[dict(times=slice(-.2, .8))]
# norm_data_cut = norm_data_cut.loc[dict(freqs=slice(50, 120))]
# plt.pcolormesh(norm_data_cut.times, norm_data_cut.freqs, norm_data_cut.to_array().squeeze().mean('trials'))
# plt.colorbar()
# plt.show()
#
# plt.plot(norm_data.loc[dict(freqs=slice(5, 15))].to_array().squeeze().mean('times').mean('freqs'), label='5-15Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(15, 30))].to_array().squeeze().mean('times').mean('freqs'), label='15-30Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(30, 50))].to_array().squeeze().mean('times').mean('freqs'), label='30-50Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(50, 80))].to_array().squeeze().mean('times').mean('freqs'), label='50-80Hz')
# plt.plot(norm_data.loc[dict(freqs=slice(80, 120))].to_array().squeeze().mean('times').mean('freqs'), label='80-120Hz')
# plt.legend()
# plt.show()
#
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(5, 15))].to_array().squeeze().mean('freqs'), label='5-15Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(15, 30))].to_array().squeeze().mean('freqs'), label='15-30Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(30, 50))].to_array().squeeze().mean('freqs'), label='30-50Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(50, 80))].to_array().squeeze().mean('freqs'), label='50-80Hz')
# plt.pcolormesh(norm_data.loc[dict(freqs=slice(80, 120))].to_array().squeeze().mean('freqs'), label='80-120Hz')

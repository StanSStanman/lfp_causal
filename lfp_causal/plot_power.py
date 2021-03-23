import xarray as xr
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research.get_dirs import get_dirs
from lfp_causal.compute_power import normalize_power
from lfp_causal.IO import read_sector
from lfp_causal.compute_bad_epochs import get_ch_bad_epo, get_log_bad_epo
from lfp_causal.profiling import (RepeatedTimer, memory_usage, cpu_usage)


def plot_avg_tf(powers, blines=None):
    avg_pow = None
    if blines is None:
        blines = range(len(powers))
    for p, b in zip(powers, blines):
        print('Loading', p)
        pow = xr.load_dataset(p)
        if isinstance(b, (int, float)):
            pow = normalize_power(pow, 'relchange', (-1.8, -1.3))
        elif isinstance(b, str):
            pow = normalize_power(pow, 'fbline_relchange', (-.51, -.01), file=b)
        pow = pow.loc[dict(times=slice(-1., 1.5))]
        # pow = pow.loc[dict(freqs=slice(70, 120))]
        pow = pow.to_array().squeeze().mean('trials')

        if avg_pow is None:
            avg_pow = pow
        else:
            avg_pow = (avg_pow + pow) / 2

    # avg_pow.plot(cmap='viridis')
    fig, ax = plt.subplots(1, 1)
    ax.plot(avg_pow.times, avg_pow.T)
    # cm = ax.pcolormesh(avg_pow.times, avg_pow.freqs, avg_pow)
    # plt.colorbar(cm)
    plt.show()
    return


def plot_pow_cond(powers, conditions, pick, log_bads, bad_trials, norm, tbline,
                  fbline, c_type='d', bins=2, diff=True, plot='pcm'):
    _reg = []
    for con, lb, bt in zip(conditions, log_bads, bad_trials):
        xls = pd.read_excel(con, index_col=0)
        reg = xls[pick].values

        if len(lb) != 0:
            reg = np.delete(reg, lb)
        if len(bt) != 0:
            reg = np.delete(reg, bt)
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
        # data = data.loc[dict(times=slice(-.2, .8))]
        # data = data.isel(freqs=np.arange(0, len(data.freqs), 4))
        # plt.pcolormesh(data.mean('freqs'), vmin=np.percentile(data, 5),
        #                vmax=np.percentile(data, 95))
        # plt.colorbar()
        # plt.show()
        for ff in range(data.shape[1]):
            plt.pcolormesh(data[:, ff, :])#, vmin=np.percentile(data[:, ff, :], 5),
                           # vmax=np.percentile(data[:, ff, :], 95))
            plt.colorbar()
            plt.show()
        plt.close()
        for _iu, _u in enumerate(uni):
            _idx = np.where(br == _u)
            b_data[_iu].append(data[_idx])
    for bd in range(len(b_data)):
        b_data[bd] = np.concatenate(b_data[bd], axis=0)

    n_axs = len(b_data)

    if diff is True:
        fig, axs = plt.subplots(1, n_axs + 1, figsize=(20, 5))
    else:
        fig, axs = plt.subplots(1, n_axs, figsize=(20, 5))

    if plot == 'linear':
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
            axs[ax].pcolormesh(b_data[ax].mean(0))
            axs[ax].set_yticks(range(len(data.freqs)))
            axs[ax].set_xticks(np.arange(len(data.times.values))[::500])
            axs[ax].set_yticklabels(data.freqs.values.round(3))
            axs[ax].set_xticklabels(data.times.values[::500])
            axs[ax].set_title('{0} = {1}'.format(pick, str(uni[ax])))
        if diff is True:
            axs[n_axs].pcolormesh(b_data[-1].mean(0) - b_data[0].mean(0))
            axs[n_axs].set_yticks(range(len(data.freqs)))
            axs[n_axs].set_xticks(np.arange(len(data.times.values))[::500])
            axs[n_axs].set_yticklabels(data.freqs.values.round(3))
            axs[n_axs].set_xticklabels(data.times.values[::500])
            axs[n_axs].set_title('{0} = {1} - {2}'.
                                 format(pick, str(uni[-1]),
                                        str(uni[0])))
    plt.show()

    return


if __name__ == '__main__':
    monkey = 'teddy'
    condition = 'easy'
    event = 'trig_off'
    norm = 'fbline_relchange'
    file = '{0}_pow_8_120_mt.nc'.format(event)
    bline = 'cue_on_pow_8_120_mt.nc'
    sectors = ['associative striatum', 'motor striatum', 'limbic striatum']
    # sectors = ['limbic striatum']

    dirs = get_dirs('local', 'lfp_causal')
    directory = dirs['pow'].format(monkey, condition)
    epo_dir = dirs['epo'].format(monkey, condition)
    regr_dir = dirs['reg'].format(monkey, condition)
    rec_info = op.join(dirs['ep_cnds'].format(monkey, condition),
                       'files_info.xlsx')

    # rej_files = ['0845', '0847', '0873', '0911', '0939', '0945', '0946',
    #              '0948', '0951', '0956', '0963', '0968', '1024', '1036',
    #              '1038', '1056', '1204', '1217'] + \
    #             ['0944', '0967', '0969', '0967', '0970', '0971', '0985',
    #              '1139', '1145', '1515', '1701'] + \
    #             ['0814', '0831', '0850', '0866', '0923', '0941', '1135',
    #              '1138', '1234', '1235', '1248', '1299', '1302', '1397',
    #              '1398', '1514', '1699', '1283']
    acc_files = ['0947', '0949', '0952']
    ## FREDDIE
    rej_files = ['1204', '1217', '1231', '0944', # Bad sessions
                 '0845', '0847', '0939', '0946', '0963', '1036', '1231',
                 '1233', '1234', '1514', '1699',
                 '0940', '0944', '0964', '0967', '0969', '0970', '0971',
                 '0977', '0985', '1280']
    ## TEDDY
    rej_files = ['0415', '0449', '0450',
                 '0416']

    for sect in sectors:
        fid = read_sector(rec_info, sect)
        fid = fid[fid['quality'] <= 3]
        # fid = fid[fid['neuron_type'] == 'TAN']

        all_files = []
        all_bline = []
        all_conds = []
        log_bads = []
        bad_epo = []
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
        plot_pow_cond(all_files, all_conds, 'Reward', log_bads, bad_epo, norm,
                      (-.52, -.02), all_bline, c_type='d', bins=2, diff=True,
                      plot='linear')

        # mu.stop()
        # cu.stop()
        # print(cpu_usage())

        print('done')





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
